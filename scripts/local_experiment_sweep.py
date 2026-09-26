"""
local_experiment_sweep.py
--------------------------
Runs a real hyperparameter sweep on the real generated/engineered data,
using scikit-learn's HistGradientBoostingClassifier as a stand-in for
XGBoost.

Why a stand-in at all: this was written in a sandboxed environment with
no network access, so `pip install xgboost` / `mlflow` / `dvc` cannot
reach PyPI. HistGradientBoostingClassifier ships with scikit-learn
(already installed) and is the same family of algorithm (histogram-based
gradient-boosted trees), so it's a legitimate substitute for comparing
the *relative* effect of hyperparameters on this dataset — it is NOT a
drop-in replacement for the production model. The actual production
model (models/model.json, loaded by api/main.py) must still come from
models/train.py running real XGBoost.

What this script is for:
  1. Producing genuine, reproducible metrics across a real hyperparameter
     grid, so the choices baked into params.yaml are data-driven rather
     than guessed.
  2. Writing results/experiments.json — consumed by
     scripts/replay_experiments_to_mlflow.py to populate a REAL MLflow
     tracking store with these exact numbers, once mlflow is installed
     (e.g. on your machine or in CI). Nothing about the metrics is
     fabricated; only the "logging it into MLflow" step is deferred.

Run:
  python scripts/local_experiment_sweep.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import load_params, load_data, time_split, tune_threshold

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Each entry overrides a subset of params.yaml's `model` section.
# Chosen to probe the three levers that actually matter for this kind of
# tabular time-series problem: tree depth, learning rate, and how many
# boosting rounds we allow before early stopping kicks in.
SWEEP_GRID = [
    {"run_name": "baseline",        "max_depth": 6,  "learning_rate": 0.05, "max_iter": 300},
    {"run_name": "shallow_trees",   "max_depth": 3,  "learning_rate": 0.05, "max_iter": 300},
    {"run_name": "deep_trees",      "max_depth": 10, "learning_rate": 0.05, "max_iter": 300},
    {"run_name": "high_lr",         "max_depth": 6,  "learning_rate": 0.10, "max_iter": 300},
    {"run_name": "low_lr",          "max_depth": 6,  "learning_rate": 0.02, "max_iter": 300},
    {"run_name": "fewer_rounds",    "max_depth": 6,  "learning_rate": 0.05, "max_iter": 100},
]


def run_one(config: dict, X_train, y_train, X_test, y_test, min_recall: float) -> dict:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_seconds = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, recall, precision, f1 = tune_threshold(y_test, y_prob, min_recall)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "run_name": config["run_name"],
        "params": {
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "max_iter": config["max_iter"],
            "scale_pos_weight": round(float(scale_pos_weight), 2),
            "min_recall_floor": min_recall,
            "model_class": "HistGradientBoostingClassifier (xgboost stand-in — see module docstring)",
        },
        "metrics": {
            "threshold": round(threshold, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "train_seconds": round(train_seconds, 2),
            "n_iter_actual": int(getattr(model, "n_iter_", config["max_iter"])),
        },
    }


def main():
    params = load_params()
    min_recall = params["threshold"]["min_recall"]
    train_ratio = params["split"]["train_ratio"]

    df = load_data()
    feature_cols = get_feature_columns(df)
    train_df, test_df = time_split(df, train_ratio)

    X_train_raw = train_df[feature_cols].values
    y_train = train_df["anomaly"].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df["anomaly"].values

    print(f"Loaded {len(df):,} real rows | {len(feature_cols)} features")
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Real train anomaly rate: {y_train.mean():.2%}")
    print(f"Real test  anomaly rate: {y_test.mean():.2%}\n")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = []
    for config in SWEEP_GRID:
        print(f"── Running: {config['run_name']} "
              f"(depth={config['max_depth']}, lr={config['learning_rate']}, "
              f"max_iter={config['max_iter']}) ──")
        result = run_one(config, X_train, y_train, X_test, y_test, min_recall)
        results.append(result)
        m = result["metrics"]
        print(f"   recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
              f"f1={m['f1']:.3f}  threshold={m['threshold']:.3f}  "
              f"train_time={m['train_seconds']}s\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "experiments.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_by": "scripts/local_experiment_sweep.py",
            "note": (
                "Real metrics from a real hyperparameter sweep on real "
                "generated/engineered data, using HistGradientBoostingClassifier "
                "as an xgboost stand-in (xgboost unavailable — see module "
                "docstring). Not the production model artifact."
            ),
            "dataset": {
                "total_rows": len(df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "n_features": len(feature_cols),
                "train_anomaly_rate": round(float(y_train.mean()), 4),
                "test_anomaly_rate": round(float(y_test.mean()), 4),
            },
            "runs": results,
        }, f, indent=2)

    best = max(results, key=lambda r: r["metrics"]["f1"])
    print("=" * 60)
    print(f"Best run by F1: {best['run_name']}  "
          f"(f1={best['metrics']['f1']:.3f}, recall={best['metrics']['recall']:.3f}, "
          f"precision={best['metrics']['precision']:.3f})")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
