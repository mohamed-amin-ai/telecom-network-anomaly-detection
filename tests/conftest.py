"""
conftest.py
-----------
Shared fixtures for the test suite.

`synthetic_kpi_df` builds a small, deterministic multi-cell KPI dataframe
in memory — used by test_features.py and test_common.py so those tests
don't depend on data/raw_kpis.csv existing (they test the transformation
logic itself, independent of the specific synthetic generator).

`require_model_artifacts` is used by test_api.py: those tests need a real
trained model (models/model.json + scaler.pkl + metadata.json) to import
api/main.py at all. In CI, `dvc repro` runs before pytest so these exist.
Locally (or in this sandbox, where xgboost isn't installable), if they're
absent the API tests are skipped with a clear reason instead of failing —
so `pytest tests/test_features.py tests/test_common.py` always works
standalone, and the full suite degrades gracefully rather than crashing
on import.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).parent.parent / "models"


def _model_artifacts_present() -> bool:
    return (
        (MODEL_DIR / "model.json").exists()
        and (MODEL_DIR / "scaler.pkl").exists()
        and (MODEL_DIR / "metadata.json").exists()
    )


@pytest.fixture
def require_model_artifacts():
    if not _model_artifacts_present():
        pytest.skip(
            "models/model.json, scaler.pkl, metadata.json not found — "
            "run `dvc repro` or `python models/train.py` first. "
            "(Expected in this sandbox: xgboost isn't installable here.)"
        )


@pytest.fixture
def synthetic_kpi_df():
    """
    Two cells, 40 rows each, 15-min cadence. Long enough (> LONG_WINDOW=16
    + max(LAG_STEPS)=8) that every KPI's rolling/lag features have at least
    one valid (non-NaN) row per cell after warmup.

    Cell values are deliberately on very different scales (CELL_A ~ 10s,
    CELL_B ~ 1000s) so a cross-cell-leakage bug (rolling window spanning
    the boundary between cells) would show up as an out-of-range value
    instead of silently blending two similar ranges.
    """
    n_per_cell = 40
    rng = np.random.default_rng(0)

    def make_cell(cell_id: str, base: float, n: int) -> pd.DataFrame:
        timestamps = pd.date_range("2024-01-01", periods=n, freq="15min")
        return pd.DataFrame({
            "timestamp": timestamps,
            "cell_id": cell_id,
            "latency_ms": base + rng.normal(0, 1, n),
            "throughput_mbps": base + rng.normal(0, 1, n),
            # NOT derived from `base` — deriving packet_loss_pct from a large
            # `base` (e.g. base=1000 -> base/100=10) saturates the [0, 5]
            # clip into a constant, which zeroes rolling std and turns
            # z-score into NaN for the whole cell (see EXPERIMENTS.md).
            "packet_loss_pct": np.clip(0.5 + rng.normal(0, 0.1, n), 0, 5),
            "sinr_db": base / 10 + rng.normal(0, 1, n),
            "connected_users": np.clip(base + rng.normal(0, 1, n), 1, None).astype(int),
            "anomaly": 0,
        })

    df_a = make_cell("CELL_A", base=10.0, n=n_per_cell)
    df_b = make_cell("CELL_B", base=1000.0, n=n_per_cell)
    return pd.concat([df_a, df_b], ignore_index=True)
