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
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import load_params, load_data, three_way_time_split, tune_threshold

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


def run_one(config: dict, X_train, y_train, X_val, y_val, X_test, y_test, min_recall: float) -> dict:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        # HGB's built-in early stopping carves its validation split out at
        # RANDOM, which for time-series data puts neighbouring rows of the
        # same anomaly event on both sides (leaky). Disabled on purpose —
        # boosting rounds are an explicit swept hyperparameter instead.
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_seconds = time.time() - t0

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuned on VALIDATION only, then frozen and applied to test.
    threshold, val_recall, val_precision, val_f1 = tune_threshold(y_val, val_prob, min_recall)
    y_pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    test_recall = tp / (tp + fn) if (tp + fn) else 0.0
    test_precision = tp / (tp + fp) if (tp + fp) else 0.0
    test_f1 = (2 * test_precision * test_recall / (test_precision + test_recall)
               if (test_precision + test_recall) else 0.0)

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
            "val_pr_auc": round(float(average_precision_score(y_val, val_prob)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, test_prob)), 4),
            "threshold": round(threshold, 4),
            "val_recall": round(val_recall, 4),
            "val_precision": round(val_precision, 4),
            "recall": round(float(test_recall), 4),
            "precision": round(float(test_precision), 4),
            "f1": round(float(test_f1), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "train_seconds": round(train_seconds, 2),
        },
    }


def main():
    params = load_params()
    min_recall = params["threshold"]["min_recall"]

    df = load_data()
    feature_cols = get_feature_columns(df)
    train_df, val_df, test_df = three_way_time_split(
        df, params["split"]["train_ratio"], params["split"]["val_fraction"]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)   # fit on train only
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train, y_val, y_test = (d["anomaly"].values for d in (train_df, val_df, test_df))

    print(f"Loaded {len(df):,} real rows | {len(feature_cols)} features")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Anomaly rate  train={y_train.mean():.2%}  val={y_val.mean():.2%}  test={y_test.mean():.2%}")
    print(f"Anomalous rows  train={int(y_train.sum())}  val={int(y_val.sum())}  test={int(y_test.sum())}\n")

    results = []
    for config in SWEEP_GRID:
        print(f"── Running: {config['run_name']} "
              f"(depth={config['max_depth']}, lr={config['learning_rate']}, "
              f"max_iter={config['max_iter']}) ──")
        result = run_one(config, X_train, y_train, X_val, y_val, X_test, y_test, min_recall)
        results.append(result)
        m = result["metrics"]
        print(f"   val_PR-AUC={m['val_pr_auc']:.3f}  test_PR-AUC={m['test_pr_auc']:.3f}  |  "
              f"test recall={m['recall']:.3f} precision={m['precision']:.3f} f1={m['f1']:.3f}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "experiments.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_by": "scripts/local_experiment_sweep.py",
            "note": (
                "Real metrics from a real hyperparameter sweep on the deterministic "
                "generated/engineered dataset, using HistGradientBoostingClassifier as an "
                "xgboost stand-in (xgboost unavailable in the build sandbox). Protocol: fit on "
                "train, threshold tuned on validation, test touched only for the final numbers; "
                "configs ranked by VALIDATION PR-AUC. Not the production model artifact. The test "
                "window contains very few independent anomaly events — see EXPERIMENTS.md before "
                "reading any difference between configs as meaningful."
            ),
            "dataset": {
                "total_rows": len(df),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "n_features": len(feature_cols),
                "train_anomaly_rate": round(float(y_train.mean()), 4),
                "val_anomaly_rate": round(float(y_val.mean()), 4),
                "test_anomaly_rate": round(float(y_test.mean()), 4),
            },
            "runs": results,
        }, f, indent=2)

    best = max(results, key=lambda r: r["metrics"]["val_pr_auc"])
    print("=" * 60)
    print(f"Best run by VALIDATION PR-AUC: {best['run_name']}  "
          f"(val_pr_auc={best['metrics']['val_pr_auc']:.3f})")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
