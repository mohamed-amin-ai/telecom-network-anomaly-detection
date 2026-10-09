"""
calibrate_to_original.py
--------------------------
Finds benchmark settings under which a baseline model's recall/precision/F1
land near the ORIGINAL project's reported metrics (recall 0.9524,
precision 0.7692, F1 0.8511) — at the request of the project owner.

READ THIS BEFORE QUOTING THE RESULT
  This is calibration, not validation. The settings are chosen BECAUSE they
  reproduce the original numbers, so the match is by construction and is NOT
  independent evidence that the model is good. What it buys: a reference
  difficulty, so later changes (features, model, thresholds) can be measured
  against "original-level" performance. The original numbers themselves were
  one test window with 21 anomalous rows (see EXPERIMENTS.md).

Why calibration is needed: on the clean benchmark recall already matches
(~0.95) but precision is ~0.96, far above 0.77, and moving the threshold
can't fix it — pushing precision down to 0.77 pushes recall up to ~0.975.
The original point lies BELOW the clean precision-recall curve, so the data
must be somewhat harder than clean.

Two knobs are searched:
  1. the onset dial `ramp_fraction` (the realistic profile with only
     `severity_and_ramp` on, every fault at full strength, i.e.
     severity_min = 1.0): the share of each fault over which its effect
     ramps in. 0 = abrupt step change (clean-like); larger = more gradual,
     so more of a fault's early rows are hard to tell from normal;
  2. the operating point (validation recall target).

First attempt, kept for the record: the dial was `severity_min` (fault
faintness), starting from 1.0 with the default 0.3 ramp. That already
overshot — severity_min=1.0 gave recall 0.860 / precision 0.682 on the dev
seeds (distance 0.127 from the original) and 0.8 was worse — because the
ramp alone is too hard. The original point lies BETWEEN clean (ramp 0) and
that, so the dial moved to the ramp fraction itself.
Choice: minimise Euclidean distance, in (recall, precision), between the
median over DEV seeds and the original point. F1 is NOT a target — it
follows from the other two, so its closeness is a genuine consistency check.

To check the match isn't luck, the chosen setting is then evaluated on
EVAL seeds that were not used in the search.

Uses the scikit-learn stand-in following models/train.py's recipe; xgboost
isn't installable in the build sandbox, and real XGBoost numbers will differ.

Run:
  python scripts/calibrate_to_original.py --dials 0.0 0.1        # compute dev results for some ramp_fraction values
  python scripts/calibrate_to_original.py --finalize             # pick the best, evaluate on eval seeds
Writes results/calibration_to_original.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from models.common import load_params
from scripts.study_common import apply_level, evaluate_current_dataset, spread

ORIGINAL = {"recall": 0.9524, "precision": 0.7692, "f1": 0.8511}
DEV_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107]
EVAL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DIALS = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]   # 0.075 and 0.125 added after the coarse pass showed the best region
FIXED = {"severity_min": 1.0}      # every fault at full strength; only onset gradualness varies
FIRST_ATTEMPT = {"dial": "severity_min (ramp 0.3)", "dev_best": {"1.0": {"target": 0.85, "recall": 0.860, "precision": 0.682, "f1": 0.758, "distance": 0.127}, "0.8": {"target": 0.8, "recall": 0.813, "precision": 0.924, "f1": 0.868, "distance": 0.208}}}
TARGETS = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]
RESULTS_PATH = Path(__file__).parent.parent / "results" / "calibration_to_original.json"
KEYS = ("recall", "precision", "f1", "event_detection_rate", "false_alerts_per_cell_day", "median_delay_minutes")


def distance(recall: float, precision: float) -> float:
    return float(np.hypot(recall - ORIGINAL["recall"], precision - ORIGINAL["precision"]))


def run_seeds(dial: float, seeds, params):
    days = params["data"]["periods_days"]
    rows = []
    for seed in seeds:
        apply_level("calibrated", days=days, seed=seed, realism_overrides={**FIXED, "ramp_fraction": dial})
        r = evaluate_current_dataset(params, TARGETS)
        r["seed"] = seed
        rows.append(r)
        print(f"  dial {dial:<4} seed {seed:>3}: PR-AUC={r['test_pr_auc']:.3f}", flush=True)
    return rows


def dev_summary(rows) -> dict:
    """Per target: median recall/precision/F1 over the dev seeds, and the distance to the original."""
    out = {}
    for t in TARGETS:
        k = str(t)
        med = {key: float(np.median([r["by_target"][k][key] for r in rows])) for key in ("recall", "precision", "f1")}
        out[k] = {**{key: round(v, 4) for key, v in med.items()}, "distance": round(distance(med["recall"], med["precision"]), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dials", type=float, nargs="*", default=[])
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--use-dial", type=float, default=None,
                    help="owner override: use this ramp_fraction instead of the distance-rule pick")
    ap.add_argument("--use-target", type=float, default=None,
                    help="owner override: use this validation_recall_target (with --use-dial)")
    args = ap.parse_args()

    params = load_params()
    state = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {"dev_by_dial": {}}
    if "first_attempt" not in state:           # start fresh: discard the superseded severity_min-dial results
        state = {"dev_by_dial": {}, "first_attempt": FIRST_ATTEMPT}

    for dial in args.dials:
        rows = run_seeds(dial, DEV_SEEDS, params)
        summ = dev_summary(rows)
        best_t = min(summ, key=lambda k: summ[k]["distance"])
        state["dev_by_dial"][str(dial)] = {"summary": summ, "best_target": float(best_t),
                                           "best_distance": summ[best_t]["distance"],
                                           "pr_auc_median": round(float(np.median([r["test_pr_auc"] for r in rows])), 4)}
        b = summ[best_t]
        print(f"dial {dial}: best target {best_t} -> recall {b['recall']:.3f} precision {b['precision']:.3f} "
              f"F1 {b['f1']:.3f}  (distance {b['distance']:.3f}) | median PR-AUC {state['dev_by_dial'][str(dial)]['pr_auc_median']}")

    if args.finalize:
        if not state["dev_by_dial"]:
            sys.exit("No dev results yet — run with --dials first.")
        rule_dial = min(state["dev_by_dial"], key=lambda d: state["dev_by_dial"][d]["best_distance"])
        rule_target = state["dev_by_dial"][rule_dial]["best_target"]
        best_dial, target = rule_dial, rule_target
        selection = {"method": "distance_rule", "distance_rule_pick": {"ramp_fraction": float(rule_dial), "target": float(rule_target)}}
        if args.use_dial is not None:
            if args.use_target is None or str(args.use_dial) not in state["dev_by_dial"] or args.use_target not in TARGETS:
                sys.exit("--use-dial needs --use-target, a dial already in the dev results, and a target in TARGETS.")
            best_dial, target = str(args.use_dial), args.use_target
            rule_ev = run_seeds(float(rule_dial), EVAL_SEEDS, params)
            rule_at = [r["by_target"][str(rule_target)] for r in rule_ev]
            selection = {
                "method": "owner_selected_from_dev_table",
                "distance_rule_pick": {"ramp_fraction": float(rule_dial), "target": float(rule_target),
                                       "dev_distance": state["dev_by_dial"][rule_dial]["best_distance"]},
                "selected_dev_summary": state["dev_by_dial"][best_dial]["summary"][str(target)],
                "distance_rule_pick_eval_summary": {k: spread([a[k] for a in rule_at]) for k in ("recall", "precision", "f1")},
            }
        state["selection"] = selection
        print(f"\nSelected ({selection['method']}): ramp_fraction={best_dial} (severity_min fixed at 1.0), validation_recall_target={target}")
        ev = run_seeds(float(best_dial), EVAL_SEEDS, params)
        at = [r["by_target"][str(target)] for r in ev]
        eval_summary = {k: spread([a[k] for a in at]) for k in KEYS}
        eval_summary["pr_auc"] = spread([r["test_pr_auc"] for r in ev])
        eval_summary["test_events"] = spread([r["test_events"] for r in ev])
        eval_summary["test_positive_rows"] = spread([r["test_positive_rows"] for r in ev])
        state.update({
            "generated_by": "scripts/calibrate_to_original.py", "periods_days": params["data"]["periods_days"],
            "original_readme_metrics": ORIGINAL, "dev_seeds": DEV_SEEDS, "eval_seeds": EVAL_SEEDS,
            "targets": TARGETS, "dials": DIALS,
            "fixed": FIXED, "chosen_ramp_fraction": float(best_dial), "chosen_target": float(target),
            "eval_summary": eval_summary,
            "eval_distance_of_medians": round(distance(eval_summary["recall"]["median"], eval_summary["precision"]["median"]), 4),
            "eval_rows": ev,
        })
        print("\n=== EVAL seeds (not used in the search) — median (min–max) vs original ===")
        for key in ("recall", "precision", "f1"):
            s = eval_summary[key]
            print(f"  {key:10s} {s['median']:.3f} ({s['min']:.3f}–{s['max']:.3f})   original {ORIGINAL[key]:.3f}")
        for key in ("event_detection_rate", "false_alerts_per_cell_day", "median_delay_minutes"):
            s = eval_summary[key]
            print(f"  {key:26s} {s['median']} ({s['min']}–{s['max']})")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
