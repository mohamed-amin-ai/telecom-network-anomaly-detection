"""
common.py
---------
Framework-agnostic pieces of the training pipeline: loading params/data,
the time-based split, and recall-floor threshold tuning.

Pulled out of train.py so this logic has exactly one implementation,
used both by the real XGBoost training run (train.py) and by
scripts/local_experiment_sweep.py (a scikit-learn-based sweep used when
xgboost/mlflow aren't installable, e.g. offline/air-gapped environments).
Only depends on numpy/pandas/sklearn/pyyaml — no xgboost, no mlflow —
so it stays importable anywhere those two aren't available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

DATA_DIR    = Path(__file__).parent.parent / "data"
MODEL_DIR   = Path(__file__).parent
PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_data() -> pd.DataFrame:
    feat_path = DATA_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            "features.csv not found. Run: python data/features.py"
        )
    return pd.read_csv(feat_path, parse_dates=["timestamp"])


def time_split(df: pd.DataFrame, ratio: float):
    """Split by time, not randomly — critical for time-series models."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def tune_threshold(y_true, y_prob, min_recall: float):
    """
    Find the highest threshold where recall >= min_recall.

    In anomaly detection, missing a real fault (false negative) is far more
    costly than a false alarm (false positive). So we optimize for recall
    first, then tighten precision as much as possible without dropping
    below the recall floor.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    recalls    = recalls[:-1]
    precisions = precisions[:-1]

    valid = thresholds[recalls >= min_recall]
    if len(valid) == 0:
        # Recall floor unachievable at any threshold — fall back to
        # whichever threshold gives the best recall available.
        best_idx = int(np.argmax(recalls))
    else:
        best_idx = int(np.where(thresholds == valid[-1])[0][0])

    f1 = 2 * precisions[best_idx] * recalls[best_idx] / (
        precisions[best_idx] + recalls[best_idx] + 1e-8
    )
    return (
        float(thresholds[best_idx]),
        float(recalls[best_idx]),
        float(precisions[best_idx]),
        float(f1),
    )
