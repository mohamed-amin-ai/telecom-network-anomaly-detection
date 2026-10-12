"""
Tests the NumPy fast path in api/live_features.py against the original engineer_features() pipeline.
The fast path must return the same features (to floating-point noise) or decline (None) so the caller
falls back to the original code, which keeps the detailed error behaviour.
"""

import datetime as dt
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from api import live_features as lf
from data.features import engineer_features

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _row(i, rng, flat=False, zero_loss=False):
    stamp = (T0 + dt.timedelta(minutes=15 * i)).isoformat()
    if i % 3 == 0:                       # mix client-style and server-style timestamps
        stamp = stamp.replace("+00:00", "Z")
    return lf.make_raw_row(stamp, "C", {
        "latency_ms": 40.0 if flat else 40 + rng.random() * 10,
        "throughput_mbps": 80 + rng.random() * 5,
        "packet_loss_pct": 0.0 if zero_loss and i % 7 == 0 else 0.5 + rng.random(),
        "sinr_db": 15 + rng.random() * 3,
        "connected_users": 50 + rng.randint(0, 20),
    })


def _reference(buf, stamp):
    df = pd.DataFrame(buf)
    df["cell_id"] = "C"
    df["anomaly"] = 0
    feat = engineer_features(df, verbose=False)
    match = feat[feat["timestamp"] == stamp]
    if match.empty:
        return None
    return match.iloc[-1].drop(labels=["timestamp", "cell_id", "anomaly"]).to_dict()


def test_fast_path_matches_engineer_features():
    rng = random.Random(11)
    compared = 0
    for trial in range(40):
        rows = [_row(i, rng, flat=(trial % 5 == 0 and i > 10), zero_loss=(trial % 4 == 0)) for i in range(50)]
        for end in range(20, 50, 3):
            buf = rows[max(0, end - lf.MAX_HISTORY_ROWS):end]
            stamp = buf[-1]["timestamp"]
            fast = lf.fast_latest_features(buf, stamp)
            if fast is None:
                continue                                  # declined: falls back to the original path
            ref = _reference(buf, stamp)
            assert ref is not None and set(ref) == set(fast)
            for k, v in ref.items():
                assert fast[k] == pytest.approx(float(v), rel=1e-9, abs=1e-9), k
            compared += 1
    assert compared > 200                                  # the fast path really is the common case


def test_declines_when_a_feature_is_undefined_or_input_is_odd():
    rng = random.Random(2)
    rows = [_row(i, rng) for i in range(30)]
    flat = [dict(r, latency_ms=40.0) for r in rows]        # flat window -> original path decides
    assert lf.fast_latest_features(flat, flat[-1]["timestamp"]) is None
    zero_prev = [dict(r) for r in rows]
    zero_prev[-2]["packet_loss_pct"] = 0.0                 # rate of change undefined
    assert lf.fast_latest_features(zero_prev, zero_prev[-1]["timestamp"]) is None
    dup = rows + [dict(rows[-1])]                          # duplicate timestamp
    assert lf.fast_latest_features(dup, dup[-1]["timestamp"]) is None
    assert lf.fast_latest_features(rows, "1999-01-01T00:00:00Z") is None


def test_undefined_feature_still_raises_with_fast_path_on(monkeypatch):
    monkeypatch.setattr(lf, "USE_FAST_FEATURES", True)
    rng = random.Random(4)
    store = lf.CellHistoryStore()
    rows = [_row(i, rng) for i in range(25)]
    rows[-2]["packet_loss_pct"] = 0.0
    for r in rows[:-1]:
        store.add_and_compute("C", r)
    with pytest.raises(lf.UndefinedFeatureError):
        store.add_and_compute("C", rows[-1])


def test_store_gives_same_features_with_and_without_fast_path(monkeypatch):
    rng = random.Random(9)
    rows = [_row(i, rng) for i in range(40)]
    results = {}
    for flag in (True, False):
        monkeypatch.setattr(lf, "USE_FAST_FEATURES", flag)
        store = lf.CellHistoryStore()
        results[flag] = [store.add_and_compute("C", r) for r in rows]
    for a, b in zip(results[True], results[False]):
        assert (a is None) == (b is None)
        if a is not None:
            assert set(a) == set(b)
            for k in a:
                assert a[k] == pytest.approx(float(b[k]), rel=1e-9, abs=1e-9), k
