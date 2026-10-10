"""
test_shared_history.py
-----------------------
The Redis-backed history store and inference log (api/shared_store.py).

Needs the `redis` package. Tests that need a live server run only when
REDIS_TEST_URL is set (CI starts a Redis service container and sets it;
locally: `redis-server --port 6399` then REDIS_TEST_URL=redis://localhost:6399/15).
Without it those tests SKIP — they are not silently passing.

What they pin down:
  - parity: Redis-backed features equal the in-memory path's, bit for bit
  - the reason Redis exists: with two "pods" sharing one Redis a cell warms up
    after 20 readings in total, while two in-memory stores split the readings
    and both stay cold
  - atomic cap, expiry, cell counting
  - outage -> HistoryStoreUnavailable (needs no server: connects to a closed port)
  - the shared inference log round-trips through Redis for the drift job
"""

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from api.live_features import (
    MAX_HISTORY_ROWS,
    MIN_HISTORY_ROWS,
    CellHistoryStore,
    HistoryStoreUnavailable,
    MemoryHistoryBackend,
    RedisHistoryBackend,
    make_raw_row,
)

REDIS_TEST_URL = os.environ.get("REDIS_TEST_URL")


def _redis_module():
    try:
        import redis
    except ImportError:
        pytest.skip("redis package not installed")
    return redis


def _live_client():
    redis = _redis_module()
    if not REDIS_TEST_URL:
        pytest.skip("REDIS_TEST_URL not set (no live Redis for this run)")
    return redis.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=2, socket_timeout=2)


def _readings(n, cell="C1", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    rows = []
    for i in range(n):
        kpis = {
            "latency_ms": float(20 + rng.normal(0, 2)),
            "throughput_mbps": float(100 + rng.normal(0, 5)),
            "packet_loss_pct": float(abs(rng.normal(1.0, 0.2)) + 0.05),
            "sinr_db": float(15 + rng.normal(0, 1)),
            "connected_users": float(200 + rng.normal(0, 10)),
        }
        rows.append(make_raw_row(str(ts[i]), cell, kpis))
    return rows


def _redis_store(prefix=None, ttl=3600):
    client = _live_client()
    prefix = prefix or f"test:{uuid.uuid4().hex}"
    return CellHistoryStore(RedisHistoryBackend(client, ttl_seconds=ttl, key_prefix=prefix)), client, prefix


def _cleanup(client, prefix):
    for key in client.scan_iter(f"{prefix}:*"):
        client.delete(key)


def test_redis_features_equal_in_memory_features():
    store, client, prefix = _redis_store()
    try:
        mem = CellHistoryStore()
        for row in _readings(MIN_HISTORY_ROWS + 6):
            a, b = mem.add_and_compute("C1", row), store.add_and_compute("C1", row)
            assert (a is None) == (b is None)
            if a is not None:
                assert a.keys() == b.keys()
                for k in a:
                    assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), k
    finally:
        _cleanup(client, prefix)


def test_two_replicas_sharing_redis_warm_up_but_two_memory_stores_do_not():
    rows = _readings(MIN_HISTORY_ROWS)
    # Two in-memory "pods" alternating readings: each sees half, neither warms up.
    pod_a, pod_b = CellHistoryStore(), CellHistoryStore()
    results = [(pod_a, pod_b)[i % 2].add_and_compute("C1", r) for i, r in enumerate(rows)]
    assert all(r is None for r in results)

    # Two Redis-backed pods over the same Redis: the 20th reading is scorable on either one.
    a, client, prefix = _redis_store()
    try:
        b = CellHistoryStore(RedisHistoryBackend(client, key_prefix=prefix))
        results = [(a, b)[i % 2].add_and_compute("C1", r) for i, r in enumerate(rows)]
        assert all(r is None for r in results[:-1])
        assert results[-1] is not None
        assert a.rows_available("C1") == b.rows_available("C1") == MIN_HISTORY_ROWS
    finally:
        _cleanup(client, prefix)


def test_history_is_capped_and_cells_are_counted_and_expire():
    store, client, prefix = _redis_store(ttl=1)
    try:
        for row in _readings(MAX_HISTORY_ROWS + 15):
            store.add_and_compute("C1", row)
        assert store.rows_available("C1") == MAX_HISTORY_ROWS
        store.add_and_compute("C2", _readings(1, cell="C2")[0])
        assert store.tracked_cell_count() == 2
        import time
        time.sleep(1.3)
        assert store.rows_available("C1") == 0, "key should expire after ttl without readings"
        assert store.tracked_cell_count() == 0, "expired cells must not stay counted"
    finally:
        _cleanup(client, prefix)


def test_unreachable_redis_raises_history_store_unavailable():
    redis = _redis_module()
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.3, socket_timeout=0.3)
    store = CellHistoryStore(RedisHistoryBackend(dead))
    row = _readings(1)[0]
    with pytest.raises(HistoryStoreUnavailable):
        store.add_and_compute("C1", row)
    with pytest.raises(HistoryStoreUnavailable):
        store.rows_available("C1")
    with pytest.raises(HistoryStoreUnavailable):
        store.tracked_cell_count()


def test_default_backend_is_memory_without_redis_url(monkeypatch):
    from api import shared_store
    monkeypatch.setattr(shared_store.os, "environ", {k: v for k, v in os.environ.items() if k != "REDIS_URL"})
    assert isinstance(shared_store.build_history_backend(), MemoryHistoryBackend)
    assert shared_store.get_redis_client() is None


def test_inference_log_round_trips_through_redis_oldest_first_and_capped(monkeypatch):
    from api import inference_log
    client = _live_client()
    key = f"test:{uuid.uuid4().hex}:inference_log"
    cols = ["f1", "f2"]
    monkeypatch.setattr(inference_log, "REDIS_LOG_KEY", key)
    monkeypatch.setattr(inference_log, "_redis_client", client)
    monkeypatch.setenv("INFERENCE_LOG_MAX_ROWS", "3")
    try:
        assert inference_log.read_log(cols) is None, "empty log must read as None, not an empty frame"
        for i in range(5):
            inference_log.log_inference("C1", "live", 0.1 * i, {"f1": float(i), "f2": 2.0 * i}, cols)
        df = inference_log.read_log(cols)
        assert list(df["f1"]) == [2.0, 3.0, 4.0], "keeps the newest 3, oldest first"
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        with pytest.raises(ValueError):
            inference_log.read_log(["f1", "missing"])
    finally:
        client.delete(key)


def test_inference_log_redis_write_failure_never_raises(monkeypatch):
    redis = _redis_module()
    from api import inference_log
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.3, socket_timeout=0.3)
    monkeypatch.setattr(inference_log, "_redis_client", dead)
    inference_log.log_inference("C1", "live", 0.5, {"f1": 1.0}, ["f1"])   # must not raise


def test_inference_log_redis_read_failure_raises_not_none(monkeypatch):
    redis = _redis_module()
    from api import inference_log
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.3, socket_timeout=0.3)
    monkeypatch.setattr(inference_log, "_redis_client", dead)
    with pytest.raises(Exception):
        inference_log.read_log(["f1"])    # a drift job must fail loudly, not report "no traffic"
