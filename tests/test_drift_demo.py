import pandas as pd

from scripts.drift_demo import SCENARIOS, apply_shift, shifted_columns

COLS = ["latency_ms", "latency_ms_lag1", "latency_ms_roll_mean_1h", "latency_ms_zscore", "latency_ms_roc",
        "sinr_db", "sinr_db_lag1", "hour"]


def test_only_level_features_of_the_shifted_kpi_are_scaled():
    cols = shifted_columns(COLS, {"latency_ms": 2.0})
    assert set(cols) == {"latency_ms", "latency_ms_lag1", "latency_ms_roll_mean_1h"}


def test_apply_shift_scales_values_and_leaves_input_untouched():
    df = pd.DataFrame([[10.0] * len(COLS)], columns=COLS)
    out = apply_shift(df, {"latency_ms": 2.0})
    assert out.loc[0, "latency_ms_lag1"] == 20.0
    assert out.loc[0, "latency_ms_zscore"] == 10.0   # scale-free feature unchanged
    assert out.loc[0, "sinr_db"] == 10.0 and out.loc[0, "hour"] == 10.0
    assert df.loc[0, "latency_ms"] == 10.0


def test_control_scenario_changes_nothing():
    df = pd.DataFrame([[3.0] * len(COLS)], columns=COLS)
    assert apply_shift(df, SCENARIOS["healthy_holdout"]).equals(df)
