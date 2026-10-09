"""
seed_variance_check.py
------------------------
Runs ONE model configuration (the baseline from params.yaml's defaults)
against several differently-seeded synthetic datasets, using the same
train/validation/test protocol as train.py, and reports how much the
headline metrics move.

Why this exists: the generator injects only a handful of anomaly EVENTS
(roughly 6-9 per cell), so any single draw's test window contains very
few independent events. Whether a hard event lands in the test window
changes recall/precision dramatically. This script puts a number on
that, so nobody reads a single run's metrics as a stable property of
the model. It uses the scikit-learn stand-in model (xgboost isn't
installable in the build sandbox) — the point is the SPREAD across
seeds, which is a property of the dataset/evaluation design, not of
which boosted-tree library is used.

Run:
  python scripts/seed_variance_check.py
Writes results/seed_variance.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
import data.generate_data as gen
from data.features import engineer_features, get_feature_columns
from models.common import load_params, three_way_time_split
from scripts.local_experiment_sweep import run_one

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
BASELINE = {"run_name": "baseline", "max_depth": 6, "learning_rate": 0.05, "max_iter": 300}
RESULTS_PATH = Path(__file__).parent.parent / "results" / "seed_variance.json"


def count_events(labels: np.ndarray, cell_ids: np.ndarray) -> int:
    """Number of contiguous anomaly runs, counted per cell."""
    events = 0
    for cell in np.unique(cell_ids):
        a = labels[cell_ids == cell]
        events += int(((a[1:] == 1) & (a[:-1] == 0)).sum() + (a[0] == 1))
    return events


def one_seed(seed: int, params: dict) -> dict:
    gen.SEED = seed  # build_dataset() reads this module global at call time
    df = engineer_features(gen.build_dataset(), verbose=False)
    feature_cols = get_feature_columns(df)
    train_df, val_df, test_df = three_way_time_split(
        df, params["split"]["train_ratio"], params["split"]["val_fraction"]
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train, y_val, y_test = (d["anomaly"].values for d in (train_df, val_df, test_df))

    result = run_one(BASELINE, X_train, y_train, X_val, y_val, X_test, y_test,
                     params["threshold"]["validation_recall_target"])
    m = result["metrics"]
    return {
        "seed": seed,
        "test_events": count_events(y_test, test_df["cell_id"].values),
        "val_events": count_events(y_val, val_df["cell_id"].values),
        "test_positive_rows": int(y_test.sum()),
        "val_positive_rows": int(y_val.sum()),
        "test_pr_auc": m["test_pr_auc"],
        "test_recall": m["recall"],
        "test_precision": m["precision"],
        "test_f1": m["f1"],
    }


def main():
    params = load_params()
    rows = []
    print(f"{'seed':>5} {'val ev':>6} {'test ev':>7} {'test +rows':>10} "
          f"{'PR-AUC':>7} {'recall':>7} {'prec':>6} {'F1':>6}")
    for seed in SEEDS:
        r = one_seed(seed, params)
        rows.append(r)
        print(f"{r['seed']:>5} {r['val_events']:>6} {r['test_events']:>7} {r['test_positive_rows']:>10} "
              f"{r['test_pr_auc']:>7.3f} {r['test_recall']:>7.3f} {r['test_precision']:>6.3f} {r['test_f1']:>6.3f}")

    print()
    summary = {}
    for key in ("test_pr_auc", "test_recall", "test_precision", "test_f1"):
        vals = np.array([r[key] for r in rows])
        summary[key] = {"min": round(float(vals.min()), 3), "median": round(float(np.median(vals)), 3),
                        "max": round(float(vals.max()), 3)}
        print(f"{key:>15}: min={vals.min():.3f}  median={np.median(vals):.3f}  max={vals.max():.3f}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "generated_by": "scripts/seed_variance_check.py",
            "note": "Baseline config, scikit-learn stand-in model, train/val/test protocol identical to "
                    "train.py. Measures spread across generator seeds, not model quality.",
            "validation_recall_target": params["threshold"]["validation_recall_target"],
            "runs": rows,
            "summary": summary,
        }, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
