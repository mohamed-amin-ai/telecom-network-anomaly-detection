"""
test_common.py
--------------
Unit tests for models/common.py — the pure split/threshold-tuning logic
shared by the real training script and the offline sweep. No xgboost or
mlflow dependency, so these run anywhere numpy/pandas/sklearn do.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from models.common import three_way_time_split, time_split, tune_threshold


def test_time_split_respects_ratio():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
        "value": range(100),
    })
    train_df, test_df = time_split(df, ratio=0.8)

    assert len(train_df) == 80
    assert len(test_df) == 20


def test_time_split_is_chronological_not_random():
    """The whole point of a time split: nothing from the test period should
    precede anything in the train period."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="15min"),
        "value": range(50),
    })
    # Shuffle input order to make sure time_split sorts before splitting,
    # rather than trusting the caller's row order.
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_df, test_df = time_split(shuffled, ratio=0.8)

    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_tune_threshold_meets_recall_floor():
    # 10 positives, 90 negatives. Construct scores so a clean separation
    # exists: negatives cluster low, positives cluster high, with one
    # ambiguous positive scored low to make the recall floor bite.
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([
        rng.uniform(0.0, 0.3, 90),   # negatives, low scores
        rng.uniform(0.6, 0.9, 9),    # 9 easy positives, high scores
        [0.05],                       # 1 hard positive, scored like a negative
    ])

    threshold, recall, precision, f1 = tune_threshold(y_true, y_prob, min_recall=0.9)

    # Recall floor is 0.9 -> at least 9/10 positives must be caught.
    assert recall >= 0.9
    # With the hard positive scored at 0.05, catching it would require a
    # threshold low enough to also catch a lot of negatives clustered
    # in [0, 0.3] — tune_threshold should NOT sacrifice precision for the
    # one unreachable point once the 0.9 floor is already satisfied by
    # catching the 9 easy ones.
    assert recall < 1.0
    assert precision > 0.5


def test_tune_threshold_falls_back_when_floor_unreachable():
    """If no threshold can achieve the requested recall (e.g. floor=1.0 but
    the model actually misses at least one positive at every threshold),
    tune_threshold must not crash — it should fall back to the
    best-recall threshold available."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.2, 0.3, 0.05])  # one positive scored very low

    threshold, recall, precision, f1 = tune_threshold(y_true, y_prob, min_recall=1.0)

    assert 0.0 <= recall <= 1.0
    assert 0.0 <= precision <= 1.0
    assert np.isfinite(f1)


def test_tune_threshold_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])

    threshold, recall, precision, f1 = tune_threshold(y_true, y_prob, min_recall=0.9)

    assert recall == 1.0
    assert precision == 1.0
    # Not exactly 1.0: tune_threshold adds a 1e-8 epsilon to the F1
    # denominator to avoid div-by-zero on the 0-precision/0-recall case.
    assert np.isclose(f1, 1.0, atol=1e-6)


def _timeline(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "value": range(n),
    })


def test_three_way_split_sizes():
    train_df, val_df, test_df = three_way_time_split(_timeline(100), train_ratio=0.8, val_fraction=0.25)
    assert len(test_df) == 20                    # last 20%
    assert len(val_df) == 20                     # 25% of the 80 pre-test rows
    assert len(train_df) == 60
    assert len(train_df) + len(val_df) + len(test_df) == 100


def test_three_way_split_is_strictly_chronological_and_non_overlapping():
    df = _timeline(100).sample(frac=1, random_state=3).reset_index(drop=True)  # shuffled input
    train_df, val_df, test_df = three_way_time_split(df, train_ratio=0.8, val_fraction=0.25)

    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()
    all_ts = pd.concat([train_df, val_df, test_df])["timestamp"]
    assert all_ts.is_unique


def test_three_way_split_test_window_matches_plain_time_split():
    """Everything else in the repo (ONNX parity, Triton client, drift
    reference) calls time_split(df, train_ratio) and expects the same
    test window train.py evaluates on. Carving out a validation slice
    must not move it."""
    df = _timeline(200)
    _, _, test_3way = three_way_time_split(df, train_ratio=0.8, val_fraction=0.2)
    _, test_2way = time_split(df, ratio=0.8)
    pd.testing.assert_frame_equal(test_3way.reset_index(drop=True), test_2way.reset_index(drop=True))
