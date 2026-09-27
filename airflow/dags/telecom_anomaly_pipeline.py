"""
telecom_anomaly_pipeline.py
-----------------------------
Airflow DAG: check production drift first; only retrain (and only
promote) if drift was actually detected. Runs daily by default.

    run_drift_check
          │
          ▼
   decide_whether_to_retrain (branch)
      │                   │
   [drift]            [no drift]
      ▼                   ▼
   generate_data      skip_retrain
      ▼
   engineer_features
      ▼
   train_model
      ▼
   promote_model  (internally re-checks the recall floor —
                    "drift happened" and "new model is good enough
                    to serve" are two separate gates, not one)

═══════════════════════════════════════════════════════════════════════
CAVEATS — read before trusting this DAG in production
═══════════════════════════════════════════════════════════════════════
1. Airflow itself isn't installable in the sandbox this was built in (no
   network), so this file's exact behavior against a live scheduler is
   unverified. Confidence is higher than Evidently's case in Phase 4 —
   the TaskFlow API (@dag/@task) has been stable for several years — but
   still: the first real `airflow dags list-import-errors` (or just
   opening the webserver UI and watching this DAG parse) is the actual
   verification, not this comment.

2. `generate_data` regenerates the SAME synthetic dataset every time
   (fixed seed=42) — it does not produce new information. In this demo,
   it stands in for "new labeled production data became available."
   Rerunning it on a real drift signal will retrain on byte-identical
   data and produce the same model. A real deployment would replace this
   task with a pull from wherever labeled production data actually lands
   (a data warehouse table, an object-storage prefix, etc.), not a
   synthetic generator.

3. Promoting a model version to "Production" in the MLflow registry
   (models/promote.py) currently has NO EFFECT on what api/main.py
   actually serves — see that file's own docstring. This DAG updates the
   registry; it does not restart or reconfigure the running API.
═══════════════════════════════════════════════════════════════════════
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

# Set by docker-compose.airflow.yml's volume mount — the whole repo is
# bind-mounted here so DAG tasks can import data/, models/, monitoring/
# exactly like every other script in this project does.
PROJECT_ROOT = Path("/opt/project")
sys.path.append(str(PROJECT_ROOT))

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="telecom_anomaly_pipeline",
    description="Daily drift check; retrain + conditionally promote only if drift is detected.",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["telecom-anomaly", "mlops"],
)
def telecom_anomaly_pipeline():

    @task
    def run_drift_check() -> bool:
        """Runs the existing monitoring/drift_check.py unchanged (shelled
        out to, since it's a proper CLI script with its own argparse —
        simpler and more reliable than re-implementing its logic inline).
        Returns whether drift was detected, read back from the status
        JSON it writes; False (not an error) if there's no production
        traffic logged yet to check."""
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "monitoring" / "drift_check.py"),
                "--pushgateway-url", "http://pushgateway:9091",
                "--min-current-rows", "10",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("drift_check.py failed — see logs above")

        status_path = PROJECT_ROOT / "monitoring" / "reports" / "latest_drift_status.json"
        if not status_path.exists():
            print("No drift status written (likely: not enough production traffic logged "
                  "yet). Treating as 'no drift' rather than retraining blind.")
            return False

        with open(status_path) as f:
            status = json.load(f)
        return bool(status["dataset_drift"])

    @task.branch
    def decide_whether_to_retrain(drift_detected: bool) -> str:
        return "generate_data" if drift_detected else "skip_retrain"

    @task
    def skip_retrain():
        print("No drift detected — leaving the current model as-is.")

    @task
    def generate_data():
        # See caveat #2 above: deterministic in this demo, stands in for
        # "new production data arrived" in a real deployment.
        from data.generate_data import main as generate_main
        generate_main()

    @task
    def engineer_features():
        import pandas as pd
        from data.features import engineer_features as _engineer_features

        raw_path = PROJECT_ROOT / "data" / "raw_kpis.csv"
        df_raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
        df_feat = _engineer_features(df_raw)
        df_feat.to_csv(PROJECT_ROOT / "data" / "features.csv", index=False)

    @task
    def train_model():
        from models.train import main as train_main
        train_main()

    @task
    def promote_model():
        from models.common import load_params
        from models.promote import promote_latest_model

        params = load_params()
        model_name = params["mlflow"]["registered_model_name"]
        min_recall = params["threshold"]["min_recall"]

        outcome = promote_latest_model(model_name, min_recall)
        print(f"Promotion outcome: {outcome}")
        return outcome

    drifted = run_drift_check()
    branch = decide_whether_to_retrain(drifted)

    skip = skip_retrain()
    gen = generate_data()
    feat = engineer_features()
    train = train_model()
    promote = promote_model()

    branch >> skip
    branch >> gen >> feat >> train >> promote


telecom_anomaly_pipeline()
