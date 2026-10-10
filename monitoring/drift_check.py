"""
drift_check.py
---------------
Compares the feature distribution of real production traffic (logged by
api/inference_log.py) against the training distribution (features.csv's
train split), using Evidently. Pushes a summary to a Prometheus
Pushgateway and saves a full HTML report for human inspection.

═══════════════════════════════════════════════════════════════════════
IMPORTANT — READ BEFORE RELYING ON THIS SCRIPT
═══════════════════════════════════════════════════════════════════════
This was written in a sandboxed environment with no network access, so
`evidently` itself could not be installed here to verify the exact
method names and result-dict shape against the real library. Evidently
has had substantial breaking API changes across versions (the Report/
metric_preset API used below is the 0.4.x line, pinned in
requirements.txt as evidently==0.4.33) — if you're reading this after
upgrading that pin, the API below may no longer match.

What this means practically:
  - `Report(metrics=[DataDriftPreset()])`, `.run(reference_data=,
    current_data=)`, `.save_html()`, and `.as_dict()` are the parts I'm
    most confident about — they've been Evidently's basic report
    interface for a long time.
  - The exact KEY NAMES inside `report.as_dict()["metrics"][i]["result"]`
    (e.g. whether it's "drift_share" vs "share_of_drifted_columns") are
    the part I'm least confident about. `_extract_drift_summary()` below
    therefore tries several candidate key names and, if NONE of them
    match, raises a clear error and dumps the raw dict to
    monitoring/reports/_debug_raw_result.json — inspect that file,
    find the real key names, and add them to the CANDIDATE_KEYS lists
    below, rather than assuming the numbers it prints are correct.

First real run of this script should be treated as a smoke test, not
a trusted production job, until you've confirmed the parsed numbers
against that raw dict at least once.
═══════════════════════════════════════════════════════════════════════

Usage:
  python monitoring/drift_check.py
  python monitoring/drift_check.py --pushgateway-url http://localhost:9091
  python monitoring/drift_check.py --min-current-rows 50
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import MODEL_DIR, REFERENCE_FILENAME, load_params, load_data, time_split
from api.inference_log import read_log

REPORTS_DIR = Path(__file__).parent / "reports"

# Candidate key names to try, in order, for each field we need out of
# Evidently's result dict — see the module docstring above.
CANDIDATE_KEYS = {
    "dataset_drift": ["dataset_drift"],
    "drift_share": ["drift_share", "share_of_drifted_columns"],
    "number_of_columns": ["number_of_columns"],
    "number_of_drifted_columns": ["number_of_drifted_columns"],
    "drift_by_columns": ["drift_by_columns"],
}


def _extract_drift_summary(result: dict) -> dict:
    """Walk every metric's result dict looking for the fields we need,
    trying each candidate key name. Raises RuntimeError with a debug
    dump path if any required field is never found anywhere."""
    found = {}
    for metric_entry in result.get("metrics", []):
        metric_result = metric_entry.get("result", {})
        if not isinstance(metric_result, dict):
            continue
        for field, candidates in CANDIDATE_KEYS.items():
            if field in found:
                continue
            for key in candidates:
                if key in metric_result:
                    found[field] = metric_result[key]
                    break

    required = ["dataset_drift", "drift_share", "number_of_columns", "number_of_drifted_columns"]
    missing = [f for f in required if f not in found]
    if missing:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        debug_path = REPORTS_DIR / "_debug_raw_result.json"
        with open(debug_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        raise RuntimeError(
            f"Could not find expected field(s) {missing} anywhere in "
            f"Evidently's report.as_dict() output — this script's "
            f"CANDIDATE_KEYS is likely out of date for your installed "
            f"evidently version. Raw result dumped to {debug_path} — "
            f"inspect it, find the real key names, and update "
            f"CANDIDATE_KEYS at the top of monitoring/drift_check.py. "
            f"See this file's module docstring for context."
        )
    # drift_by_columns is optional (only used for the per-column gauges)
    found.setdefault("drift_by_columns", {})
    return found


def _push_to_gateway(summary: dict, reference_rows: int, current_rows: int, pushgateway_url: str) -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()

    Gauge("data_drift_detected", "1 if Evidently flagged dataset-level drift, else 0.",
          registry=registry).set(1 if summary["dataset_drift"] else 0)
    Gauge("data_drift_share", "Share of features flagged as drifted (0-1).",
          registry=registry).set(summary["drift_share"])
    Gauge("data_drift_columns_total", "Total feature columns evaluated.",
          registry=registry).set(summary["number_of_columns"])
    Gauge("data_drift_columns_drifted", "Count of feature columns flagged as drifted.",
          registry=registry).set(summary["number_of_drifted_columns"])
    Gauge("drift_check_reference_rows", "Rows in the reference (training) dataset used.",
          registry=registry).set(reference_rows)
    Gauge("drift_check_current_rows", "Rows in the current (production log) dataset used.",
          registry=registry).set(current_rows)
    Gauge("drift_check_last_run_timestamp", "Unix timestamp of this drift check run.",
          registry=registry).set(datetime.now(timezone.utc).timestamp())

    # Per-column drift flags. Cardinality here is bounded by the FIXED
    # feature count (~68), not by anything that grows in production
    # (contrast with api/metrics.py's deliberate avoidance of a cell_id
    # label) — safe to label by column name.
    if summary["drift_by_columns"]:
        column_gauge = Gauge(
            "column_drift_detected", "1 if this feature column was flagged as drifted.",
            labelnames=["column"], registry=registry,
        )
        for column, col_result in summary["drift_by_columns"].items():
            detected = col_result.get("drift_detected") if isinstance(col_result, dict) else None
            if detected is not None:
                column_gauge.labels(column=column).set(1 if detected else 0)

    push_to_gateway(pushgateway_url, job="drift_check", registry=registry)


def load_reference(reference_path):
    """Reference (training) feature distribution.

    Preferred: models/reference_features.csv, a fixed-seed sample of the train split written by
    models/train.py and shipped in the image — it is what the Kubernetes CronJob uses, where
    data/features.csv does not exist. Fallback (local dev): the train split of data/features.csv.
    Returns (DataFrame restricted to the feature columns, feature column list)."""
    import pandas as pd

    path = Path(reference_path) if reference_path else MODEL_DIR / REFERENCE_FILENAME
    if path.exists():
        reference = pd.read_csv(path)
        return reference, list(reference.columns)
    if reference_path:
        raise FileNotFoundError(f"--reference {reference_path} does not exist")
    params = load_params()
    df = load_data()
    feature_cols = get_feature_columns(df)
    train_df, _ = time_split(df, params["split"]["train_ratio"])
    return train_df[feature_cols], feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pushgateway-url", default="http://localhost:9091")
    parser.add_argument("--min-current-rows", type=int, default=30,
                         help="Skip the check (rather than run on too little data) below this many logged rows.")
    parser.add_argument("--reference", default=None,
                         help="CSV of reference feature rows (default: models/reference_features.csv, "
                              "else the train split of data/features.csv).")
    parser.add_argument("--skip-push", action="store_true",
                         help="Compute and save the HTML report but don't push to Pushgateway (e.g. local debugging).")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reference, feature_cols = load_reference(args.reference)

    current_df = read_log(feature_cols=feature_cols)
    if current_df is None:
        print(f"No inference log found yet (expected at api/inference_log.py's LOG_PATH). "
              f"Nothing to compare against — send some real traffic to the API first "
              f"(e.g. `make demo-traffic`), then rerun this.")
        return
    if len(current_df) < args.min_current_rows:
        print(f"Only {len(current_df)} rows logged so far, below --min-current-rows "
              f"({args.min_current_rows}). Skipping rather than running a drift check "
              f"on too little data to be meaningful. Send more traffic and rerun.")
        return
    current = current_df[feature_cols]

    print(f"Reference (training) rows: {len(reference):,}")
    print(f"Current (production) rows: {len(current):,}")
    print("Running Evidently DataDriftPreset...")

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    print(f"Saved → {html_path}")

    result = report.as_dict()
    summary = _extract_drift_summary(result)

    status_path = REPORTS_DIR / "latest_drift_status.json"
    with open(status_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "reference_rows": len(reference),
            "current_rows": len(current),
            "dataset_drift": bool(summary["dataset_drift"]),
            "drift_share": float(summary["drift_share"]),
            "number_of_columns": int(summary["number_of_columns"]),
            "number_of_drifted_columns": int(summary["number_of_drifted_columns"]),
            "report_html": str(html_path),
        }, f, indent=2)
    print(f"Saved → {status_path}")

    print(f"\ndataset_drift={summary['dataset_drift']}  "
          f"drift_share={summary['drift_share']:.3f}  "
          f"drifted_columns={summary['number_of_drifted_columns']}/{summary['number_of_columns']}")

    if args.skip_push:
        print("--skip-push set, not pushing to Pushgateway.")
        return

    try:
        _push_to_gateway(summary, len(reference), len(current), args.pushgateway_url)
        print(f"Pushed metrics to {args.pushgateway_url} (job=drift_check)")
    except Exception as e:
        print(f"Could not push to Pushgateway at {args.pushgateway_url}: {e}\n"
              f"(Report was still generated and saved above — this only affects "
              f"the Grafana dashboard, not the report itself.)")


if __name__ == "__main__":
    main()
