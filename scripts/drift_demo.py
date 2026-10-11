"""
Does the Evidently drift check actually catch drift?

monitoring/drift_check.py is exercised in CI on traffic drawn from the same generator as the
training data, which proves the plumbing works but not that anything gets *detected*. This script
closes that gap: it compares the training reference against (a) held-out rows left as they are and
(b) the same rows with a known KPI degradation applied, and asserts that Evidently stays quiet
for (a) and raises dataset drift for the realistic multi-KPI incident in (b).

Writes results/drift_demo.json (+ an HTML report for the incident) and exits non-zero if the
expectations are not met.

Usage (after data/generate_data.py, data/features.py and models/train.py):
  python scripts/drift_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KPIS = ["latency_ms", "throughput_mbps", "packet_loss_pct", "sinr_db", "connected_users"]
# Features that are ratios/standardised values: a pure level shift of the KPI leaves them unchanged.
SCALE_FREE_SUFFIXES = ("_zscore", "_roc")

# name -> {KPI: multiplicative factor}. "none" is the control.
SCENARIOS = {
    "healthy_holdout": {},
    "latency_x2_only": {"latency_ms": 2.0},
    "congestion_incident": {"latency_ms": 2.0, "throughput_mbps": 0.5, "packet_loss_pct": 3.0,
                            "sinr_db": 0.5, "connected_users": 1.5},
}
SAMPLE_ROWS = 2000
SEED = 7


def shifted_columns(columns, factors: dict) -> dict:
    """Map feature column -> factor for every level-type feature of a shifted KPI."""
    out = {}
    for kpi, factor in factors.items():
        for col in columns:
            if col.startswith(kpi) and not col.endswith(SCALE_FREE_SUFFIXES):
                out[col] = factor
    return out


def apply_shift(df, factors: dict):
    """Return a copy of df with the level-type features of each KPI scaled by its factor."""
    out = df.copy()
    for col, factor in shifted_columns(df.columns, factors).items():
        out[col] = out[col] * factor
    return out


def run_scenario(reference, current):
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    from monitoring.drift_check import _extract_drift_summary

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    summary = _extract_drift_summary(report.as_dict())
    flagged = sorted(c for c, r in summary["drift_by_columns"].items()
                     if isinstance(r, dict) and r.get("drift_detected"))
    return report, {
        "dataset_drift": bool(summary["dataset_drift"]),
        "drift_share": round(float(summary["drift_share"]), 3),
        "drifted_columns": int(summary["number_of_drifted_columns"]),
        "columns": int(summary["number_of_columns"]),
        "flagged": flagged,
    }


def main() -> int:
    from models.common import load_data, load_params, time_split
    from monitoring.drift_check import load_reference

    reference, feature_cols = load_reference(None)
    df = load_data()
    _, test_df = time_split(df, load_params()["split"]["train_ratio"])
    sample = test_df[feature_cols].sample(n=min(SAMPLE_ROWS, len(test_df)), random_state=SEED)

    results, reports = {}, {}
    for name, factors in SCENARIOS.items():
        reports[name], results[name] = run_scenario(reference, apply_shift(sample, factors))
        results[name]["shift"] = factors
        r = results[name]
        print(f"{name:22s} dataset_drift={r['dataset_drift']!s:5s} "
              f"drifted {r['drifted_columns']}/{r['columns']} (share {r['drift_share']})")

    checks = {
        "healthy holdout is not flagged as dataset drift": not results["healthy_holdout"]["dataset_drift"],
        "congestion incident IS flagged as dataset drift": results["congestion_incident"]["dataset_drift"],
        "incident drifts more columns than the healthy control":
            results["congestion_incident"]["drifted_columns"] > results["healthy_holdout"]["drifted_columns"],
        "measured drift share equals drifted/total columns (not Evidently's 0.5 threshold)":
            all(abs(r["drift_share"] - r["drifted_columns"] / r["columns"]) < 0.001 for r in results.values()),
        "latency-only shift flags latency features":
            any(c.startswith("latency_ms") for c in results["latency_x2_only"]["flagged"]),
        "latency-only shift leaves sinr features alone":
            not any(c.startswith("sinr_db") for c in results["latency_x2_only"]["flagged"]),
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    reports["congestion_incident"].save_html(str(out_dir / "drift_demo_congestion_incident.html"))
    (out_dir / "drift_demo.json").write_text(json.dumps(
        {"reference_rows": len(reference), "current_rows_per_scenario": len(sample),
         "scenarios": results, "checks": checks}, indent=2, sort_keys=True))
    for text, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + text)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
