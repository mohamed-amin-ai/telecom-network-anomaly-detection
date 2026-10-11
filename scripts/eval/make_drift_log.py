"""Write an inference log (api/inference_log.py format) from held-out rows, optionally with a congestion
incident applied, so the Airflow DAG's drift branch has real input.   usage: make_drift_log.py healthy|shifted"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api import inference_log  # noqa: E402
from models.common import load_data, load_params, time_split  # noqa: E402
from scripts.drift_demo import SCENARIOS, apply_shift  # noqa: E402

mode = sys.argv[1]
df = load_data()
_, test_df = time_split(df, load_params()["split"]["train_ratio"])
cols = list(__import__("pandas").read_csv("models/reference_features.csv", nrows=1).columns)
sample = test_df[cols].sample(n=400, random_state=5)
if mode == "shifted":
    sample = apply_shift(sample, SCENARIOS["congestion_incident"])
inference_log.LOG_PATH.unlink(missing_ok=True)
for i, (_, row) in enumerate(sample.iterrows()):
    inference_log.log_inference(cell_id=f"EVAL_{i % 12}", feature_source="live", anomaly_score=0.1,
                                features=row.to_dict(), feature_cols=cols)
print(f"wrote {len(sample)} {mode} rows to {inference_log.LOG_PATH}")
