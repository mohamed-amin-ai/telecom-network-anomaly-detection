"""
test_live_features.py
----------------------
Tests api/live_features.py — the serving-time feature path. No fastapi or
model needed (live_features only depends on pandas and data/features.py).

Two of these guard bugs/claims found only by replaying real data:
  - train/serve parity: the API claims to compute EXACTLY the training
    features. Verified on 450 real rows to ~1e-11; kept here as a
    permanent, synthetic-data regression test.
  - stale features: when the newest reading's features are undefined
    (previous packet_loss_pct == 0 -> rate of change divides by zero), the
    old code returned the PREVIOUS row's features and scored them as
    current, silently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from api.live_features import (
    MAX_HISTORY_ROWS,
    MIN_HISTORY_ROWS,
    CellHistoryStore,
    UndefinedFeatureError,
    make_raw_row,
)
from data.features import KPI_COLS, engineer_features, get_feature_columns


def _series(n=80, seed=0, zero_packet_loss_at=()):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "cell_id": "C1",
        "latency_ms": 40 + rng.normal(0, 4, n),
        "throughput_mbps": 100 + rng.normal(0, 9, n),
        "packet_loss_pct": np.abs(rng.normal(0.6, 0.3, n)) + 0.05,   # strictly positive
        "sinr_db": 15 + rng.normal(0, 2, n),
        "connected_users": rng.integers(10, 90, n),
        "anomaly": 0,
    })
    for i in zero_packet_loss_at:
        df.loc[i, "packet_loss_pct"] = 0.0
    return df


def _replay_into_store(df, upto):
    """Feed rows [0, upto) into a fresh store's buffer, return the store."""
    store = CellHistoryStore()
    for j in range(max(0, upto - (MAX_HISTORY_ROWS - 1)), upto):
        r = df.iloc[j]
        store._backend.append("C1", make_raw_row(str(r.timestamp), "C1", {k: r[k] for k in KPI_COLS}))
    return store


def _row(df, i):
    r = df.iloc[i]
    return make_raw_row(str(r.timestamp), "C1", {k: r[k] for k in KPI_COLS})


def test_cold_start_returns_none_until_min_history():
    df = _series()
    store = CellHistoryStore()
    for i in range(MIN_HISTORY_ROWS - 1):
        assert store.add_and_compute("C1", _row(df, i)) is None
    assert store.add_and_compute("C1", _row(df, MIN_HISTORY_ROWS - 1)) is not None


def test_live_features_match_training_features_exactly():
    """The core no-train/serve-skew claim, on rows drawn from steady state."""
    df = _series(n=120)
    trained = engineer_features(df, verbose=False).set_index("timestamp")
    cols = get_feature_columns(trained.reset_index())

    checked = 0
    for i in range(MAX_HISTORY_ROWS, 120, 7):
        store = _replay_into_store(df, i)
        out = store.add_and_compute("C1", _row(df, i))
        expected = trained.loc[df.timestamp[i], cols].astype(float).values
        got = pd.Series(out)[cols].astype(float).values
        assert np.max(np.abs(got - expected)) < 1e-9, f"feature mismatch at row {i}"
        checked += 1
    assert checked >= 10


def test_zero_previous_reading_raises_instead_of_returning_stale_features():
    """packet_loss_pct == 0 at row 60 makes row 61's rate of change
    undefined. Training drops row 61; serving must not quietly hand back
    row 60's features as row 61's."""
    df = _series(n=120, zero_packet_loss_at=(60,))
    trained_ts = set(engineer_features(df, verbose=False)["timestamp"])
    assert df.timestamp[61] not in trained_ts, "precondition: training drops row 61"
    assert df.timestamp[60] in trained_ts, "precondition: row 60 itself is fine"

    store = _replay_into_store(df, 61)
    with pytest.raises(UndefinedFeatureError, match="undefined"):
        store.add_and_compute("C1", _row(df, 61))


def test_reading_that_follows_the_undefined_one_scores_normally_again():
    df = _series(n=120, zero_packet_loss_at=(60,))
    store = _replay_into_store(df, 62)          # buffer now contains row 60 (the zero) as history
    out = store.add_and_compute("C1", _row(df, 62))
    assert out is not None                        # only the row immediately after the zero is affected


def test_returned_features_belong_to_the_reading_just_added():
    """Even for a healthy row, the features returned must be that row's —
    verified by checking a feature that only depends on the current value."""
    df = _series(n=120)
    i = 90
    store = _replay_into_store(df, i)
    out = store.add_and_compute("C1", _row(df, i))
    assert out["latency_ms"] == pytest.approx(df.latency_ms[i])
    assert out["packet_loss_pct"] == pytest.approx(df.packet_loss_pct[i])
