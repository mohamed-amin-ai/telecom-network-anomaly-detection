"""
replay_experiments_to_mlflow.py
---------------------------------
Takes results/experiments.json (produced by scripts/local_experiment_sweep.py,
run against real data in an environment where mlflow/xgboost couldn't be
installed) and logs each run into a REAL MLflow tracking store.

This exists to bridge two environments honestly:
  - The sweep was executed somewhere without network access, so the
    numbers are real but were never sent to mlflow.
  - This script logs those exact real numbers as real MLflow runs, the
    moment you have `pip install mlflow` available. Nothing here is
    invented — every metric was actually measured by the sweep.

Each run is tagged model_class=HistGradientBoostingClassifier and
tagged is_xgboost_stand_in=true so they're clearly distinguishable in
the MLflow UI from real XGBoost runs produced by models/train.py.

Run:
  python scripts/replay_experiments_to_mlflow.py
"""

import json
import sys
from pathlib import Path

import mlflow

sys.path.append(str(Path(__file__).parent.parent))
from models.common import load_params

RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiments.json"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run scripts/local_experiment_sweep.py first."
        )

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    params = load_params()
    mlflow.set_experiment(params["mlflow"]["experiment_name"] + "-sweep")

    for run in data["runs"]:
        with mlflow.start_run(run_name=run["run_name"]):
            mlflow.set_tags({
                "model_class": "HistGradientBoostingClassifier",
                "is_xgboost_stand_in": "true",
                "source": "scripts/local_experiment_sweep.py",
            })
            mlflow.log_params(run["params"])
            mlflow.log_params({f"dataset_{k}": v for k, v in data["dataset"].items()})
            mlflow.log_metrics({
                k: v for k, v in run["metrics"].items()
                if isinstance(v, (int, float))
            })
        print(f"Logged run: {run['run_name']}")

    print(f"\nDone. {len(data['runs'])} real sweep runs now in MLflow "
          f"experiment '{params['mlflow']['experiment_name']}-sweep'.")
    print("Run `mlflow ui` to view them alongside real XGBoost runs from train.py.")


if __name__ == "__main__":
    main()
