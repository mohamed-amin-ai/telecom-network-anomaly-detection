"""
realism_study.py
------------------
Answers "why are the numbers lower than the clean benchmark, and which
difficulty factor is responsible?" by adding the generator's realism
factors one at a time (cumulatively) and measuring every metric after each.

Profiles (cumulative):
  clean (legacy)        original clean-room generator: upper bound
  + nonstationarity     per-cell baselines, weekly seasonality, slow drift
  + severity & ramp     faults from barely-visible to dramatic, gradual onset
  + label noise         ticket-style labels: jittered boundaries, some faults unlogged
  + benign surges       real but non-fault episodes that resemble mild faults  (= full)

Order matters in a cumulative ablation (factors interact); this shows one
path, not a unique attribution.

Also reports the operating-point trade-off for the FULL profile: raising
the recall target buys detection at the price of false alerts, which is
how a production threshold is really chosen.

All metrics on the held-out test window, validation-tuned thresholds, the
scikit-learn stand-in model. Absolute numbers from real XGBoost will differ,
and none of this is calibrated to a real network (none was available).

Run:  python scripts/realism_study.py [--seeds 42 1 2 ...]
Writes results/realism_study.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.common import load_params
from scripts.study_common import FACTORS, evaluate_current_dataset, set_profile, spread

DEFAULT_SEEDS = [42, 1, 2, 3, 4, 5]
OPERATING_POINTS = [0.5, 0.7, 0.8, 0.9, 0.96]
RESULTS_PATH = Path(__file__).parent.parent / "results" / "realism_study.json"

PROFILES = {
    "clean (legacy)":         ("legacy", ()),
    "+ nonstationarity":      ("realistic", FACTORS[:1]),
    "+ severity & ramp":      ("realistic", FACTORS[:2]),
    "+ label noise":          ("realistic", FACTORS[:3]),
    "+ benign surges (full)": ("realistic", FACTORS[:4]),
}


def realistic_level_target(params) -> str:
    """The realistic level's operating point (chosen on dev seeds by
    scripts/recall_target_study.py), NOT params.yaml's target — which belongs
    to whichever profile is configured and may be the clean benchmark's."""
    study = RESULTS_PATH.parent / "recall_target_study.json"
    if study.exists():
        levels = json.loads(study.read_text()).get("levels", {})
        if "realistic" in levels:
            return str(levels["realistic"]["chosen_target"])
    return str(params["threshold"]["validation_recall_target"])


def run(seeds, params):
    days = params["data"]["periods_days"]
    configured = str(params["threshold"]["validation_recall_target"])
    targets = sorted({*OPERATING_POINTS, float(configured)})
    all_rows = {}
    for name, (profile, factors) in PROFILES.items():
        rows = []
        for seed in seeds:
            set_profile(profile, factors, days=days, seed=seed)
            r = evaluate_current_dataset(params, targets)
            r["seed"] = seed
            rows.append(r)
            print(f"  {name:24s} seed {seed:>3}: PR-AUC={r['test_pr_auc']:.3f}", flush=True)
        all_rows[name] = rows
    return all_rows, configured


def summarize(all_rows, configured):
    summary = {}
    for name, rows in all_rows.items():
        at = lambda key: spread([r["by_target"][configured][key] for r in rows])
        summary[name] = {
            "test_pr_auc": spread([r["test_pr_auc"] for r in rows]),
            "recall": at("recall"), "precision": at("precision"), "f1": at("f1"),
            "event_detection_rate": at("event_detection_rate"),
            "false_alerts_per_cell_day": at("false_alerts_per_cell_day"),
            "median_delay_minutes": at("median_delay_minutes"),
            "anomaly_rate": spread([r["test_anomaly_rate"] for r in rows]),
        }
    full = list(all_rows)[-1]
    tradeoff = {}
    for t in sorted({*OPERATING_POINTS, float(configured)}):
        k = str(t)
        g = lambda key: spread([r["by_target"][k][key] for r in all_rows[full]])
        tradeoff[k] = {"recall": g("recall"), "precision": g("precision"), "f2": g("f2"),
                       "event_detection_rate": g("event_detection_rate"),
                       "false_alerts_per_cell_day": g("false_alerts_per_cell_day"),
                       "median_delay_minutes": g("median_delay_minutes")}
    return summary, tradeoff


def fmt(s):
    return "n/a" if s["median"] is None else f"{s['median']:.3f} ({s['min']:.3f}-{s['max']:.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = ap.parse_args()

    params = load_params()
    configured = realistic_level_target(params)
    params["threshold"]["validation_recall_target"] = float(configured)   # evaluate at the realistic level's own operating point
    print(f"{params['data']['periods_days']}-day datasets | seeds {args.seeds} | "
          f"validation recall target {configured} (the realistic level's own, from recall_target_study.json)\n")
    all_rows, configured = run(args.seeds, params)
    summary, tradeoff = summarize(all_rows, configured)

    print(f"\n=== Cumulative ablation at validation target {configured} — median (min-max) over {len(args.seeds)} seeds ===")
    print(f"{'profile':24s} {'PR-AUC':>20s} {'recall':>20s} {'precision':>20s} {'incident detect':>20s} {'false alerts/cell-day':>22s}")
    for name, s in summary.items():
        print(f"{name:24s} {fmt(s['test_pr_auc']):>20s} {fmt(s['recall']):>20s} {fmt(s['precision']):>20s} "
              f"{fmt(s['event_detection_rate']):>20s} {fmt(s['false_alerts_per_cell_day']):>22s}")

    print(f"\n=== Operating points, FULL realistic profile — median (min-max) ===")
    print(f"{'val target':>10s} {'row recall':>20s} {'row precision':>20s} {'incident detect':>20s} {'false alerts/cell-day':>22s} {'delay (min)':>14s}")
    for t, s in tradeoff.items():
        print(f"{t:>10s} {fmt(s['recall']):>20s} {fmt(s['precision']):>20s} {fmt(s['event_detection_rate']):>20s} "
              f"{fmt(s['false_alerts_per_cell_day']):>22s} {fmt(s['median_delay_minutes']):>14s}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "generated_by": "scripts/realism_study.py", "periods_days": params["data"]["periods_days"],
            "seeds": args.seeds, "validation_recall_target": configured,
            "realism_params": params["data"]["realism"],
            "summary": summary, "operating_points_full_profile": tradeoff, "rows": all_rows,
        }, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
