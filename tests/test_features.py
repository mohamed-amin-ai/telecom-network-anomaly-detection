"""
test_features.py
-----------------
Unit tests for data/features.py. Uses the in-memory synthetic_kpi_df
fixture, not the real generated CSV — these test the transformation
logic itself, independent of any particular dataset.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from data.features import engineer_features, get_feature_columns, KPI_COLS, LAG_STEPS


def test_output_has_expected_lag_and_rolling_columns(synthetic_kpi_df):
    result = engineer_features(synthetic_kpi_df, verbose=False)

    for kpi in KPI_COLS:
        for lag in LAG_STEPS:
            assert f"{kpi}_lag{lag}" in result.columns
        assert f"{kpi}_roll_mean_1h" in result.columns
        assert f"{kpi}_roll_mean_4h" in result.columns
        assert f"{kpi}_roll_std_1h" in result.columns
        assert f"{kpi}_roll_std_4h" in result.columns
        assert f"{kpi}_roll_min_1h" in result.columns
        assert f"{kpi}_roll_max_1h" in result.columns
        assert f"{kpi}_zscore" in result.columns
        assert f"{kpi}_roc" in result.columns

    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns


def test_get_feature_columns_excludes_metadata(synthetic_kpi_df):
    result = engineer_features(synthetic_kpi_df, verbose=False)
    cols = get_feature_columns(result)

    assert "timestamp" not in cols
    assert "cell_id" not in cols
    assert "anomaly" not in cols
    # Sanity: matches the documented 68 for the 5 real KPI_COLS, but here
    # we just check it's non-trivial and matches result's actual columns.
    assert len(cols) == len(result.columns) - 3


def test_dropna_removes_only_warmup_rows_per_cell(synthetic_kpi_df):
    n_per_cell = synthetic_kpi_df.groupby("cell_id").size().iloc[0]
    result = engineer_features(synthetic_kpi_df, verbose=False)

    counts = result.groupby("cell_id").size()
    assert set(counts.index) == {"CELL_A", "CELL_B"}
    # Every cell should have lost the same number of warmup rows (rolling
    # window min_periods, not an amount that depends on the other cell).
    assert counts["CELL_A"] == counts["CELL_B"]
    assert counts["CELL_A"] < n_per_cell  # some warmup rows were dropped
    assert counts["CELL_A"] > 0            # but not all rows


def test_no_cross_cell_leakage_in_rolling_features(synthetic_kpi_df):
    """
    CELL_A's latency is centered around 10, CELL_B's around 1000. If the
    per-cell groupby in engineer_features were broken (e.g. rolling
    computed on the whole concatenated frame instead of per group), a
    rolling window near the start of CELL_B would pull in CELL_A's much
    smaller values, and vice versa near a cell boundary — the result
    would land far outside each cell's own real range.
    """
    result = engineer_features(synthetic_kpi_df, verbose=False)

    a_latency = result.loc[result["cell_id"] == "CELL_A", "latency_ms_roll_mean_4h"]
    b_latency = result.loc[result["cell_id"] == "CELL_B", "latency_ms_roll_mean_4h"]

    # CELL_A's rolling mean should stay close to its own base (10), never
    # anywhere near CELL_B's base (1000), and vice versa.
    assert a_latency.max() < 100
    assert b_latency.min() > 100


def test_rate_of_change_matches_manual_calculation(synthetic_kpi_df):
    result = engineer_features(synthetic_kpi_df, verbose=False)
    cell_a = result[result["cell_id"] == "CELL_A"].sort_values("timestamp").reset_index(drop=True)

    # Recompute latency_ms_roc for one arbitrary row directly from the
    # raw (pre-feature-engineering) series and compare.
    raw_a = synthetic_kpi_df[synthetic_kpi_df["cell_id"] == "CELL_A"].sort_values("timestamp").reset_index(drop=True)

    # Find a timestamp present in both raw and engineered (post-dropna) data
    ts = cell_a["timestamp"].iloc[5]
    raw_idx = raw_a.index[raw_a["timestamp"] == ts][0]
    expected_roc = (
        (raw_a["latency_ms"].iloc[raw_idx] - raw_a["latency_ms"].iloc[raw_idx - 1])
        / raw_a["latency_ms"].iloc[raw_idx - 1]
    )
    actual_roc = cell_a.loc[cell_a["timestamp"] == ts, "latency_ms_roc"].iloc[0]

    assert np.isclose(actual_roc, expected_roc, rtol=1e-6)
