"""
recall_target_study.py
------------------------
For each generator difficulty level (clean / default / harsh), picks the
operating point WITHOUT looking at the data it is then reported on, and
reports row-level AND incident-level results.

The operating point is `threshold.validation_recall_target`: the recall the
decision threshold is tuned to reach on the validation slice. Whatever is
chosen is then measured on the held-out test window.

Selection rule — stated before any realistic-profile results were seen:
    choose the target that MAXIMISES THE MEDIAN ROW-LEVEL F2 ON THE DEV SEEDS
    (F2 weights recall twice as heavily as precision, matching this project's
    stated "a missed fault costs more than a false alarm"); ties -> the
    smaller target. Nothing is chosen from the EVAL seeds.

One documented deviation: the target grid originally started at 0.3. On the
realistic profiles the optimum sat on that boundary, so the grid was
extended downward (0.1, 0.2) — the rule itself is unchanged.

Seeds: DEV seeds choose; EVAL seeds (disjoint, containing the default
seed 42) report.

Uses the scikit-learn stand-in model following models/train.py's recipe
(validation-PR-AUC early stopping). xgboost isn't installable in the build
sandbox; absolute numbers from real XGBoost will differ, and nothing here
is calibrated to a real network.

Run:  python scripts/recall_target_study.py [--levels clean realistic harsh]
Writes results/recall_target_study.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.common import load_params
from scripts.study_common import F2_RULE_LEVELS, LEVELS, apply_level, evaluate_current_dataset, spread

DEV_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107]
EVAL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
ORIGINAL = {"recall": 0.9524, "precision": 0.7692, "f1": 0.8511}   # original README headline (20/21, 20/26)
RESULTS_PATH = Path(__file__).parent.parent / "results" / "recall_target_study.json"
KEYS = ("recall", "precision", "f1", "f2", "event_detection_rate", "false_alerts_per_cell_day",
        "alerts_per_cell_day", "flagged_fraction", "median_delay_minutes")


def summarize(rows: list, floor: float) -> dict:
    out = {}
    for t in TARGETS:
        k = str(t)
        s = {key: spread([r["by_target"][k][key] for r in rows]) for key in KEYS}
        s["seeds_meeting_recall_floor"] = int(sum(r["by_target"][k]["recall"] >= floor for r in rows))
        s["n_seeds"] = len(rows)
        out[k] = s
    return out


def choose_target(dev_summary: dict) -> float:
    """Pre-stated rule: max median F2 on dev seeds; ties -> smaller target."""
    best = max(dev_summary.items(), key=lambda kv: (round(kv[1]["f2"]["median"], 6), -float(kv[0])))
    return float(best[0])


def run_level(level: str, params: dict) -> dict:
    days = params["data"]["periods_days"]
    dev, ev = [], []
    for seeds, bucket, tag in ((DEV_SEEDS, dev, "dev"), (EVAL_SEEDS, ev, "eval")):
        for seed in seeds:
            apply_level(level, days=days, seed=seed)
            r = evaluate_current_dataset(params, TARGETS)
            r["seed"] = seed
            bucket.append(r)
            print(f"  [{level:7s}] {tag:4s} seed {seed:>3}: PR-AUC={r['test_pr_auc']:.3f}", flush=True)
    floor = params["threshold"]["min_recall"]
    dev_summary, eval_summary = summarize(dev, floor), summarize(ev, floor)
    return {
        "level": level, "profile": LEVELS[level]["profile"],
        "chosen_target": choose_target(dev_summary),
        "dev_summary": dev_summary, "eval_summary": eval_summary,
        "eval_pr_auc": spread([r["test_pr_auc"] for r in ev]),
        "eval_test_events": spread([r["test_events"] for r in ev]),
        "eval_test_positive_rows": spread([r["test_positive_rows"] for r in ev]),
        "dev_rows": dev, "eval_rows": ev,
    }


def print_level(res: dict):
    t = str(res["chosen_target"])
    e = res["eval_summary"][t]
    print(f"\n=== {res['level']} — chosen target {t} (max median dev F2) ===")
    p = res["eval_pr_auc"]
    print(f"  {'eval PR-AUC':26s} {p['min']:.3f} / {p['median']:.3f} / {p['max']:.3f}   (min / median / max)")
    for key, label in (("recall", "row recall"), ("precision", "row precision"), ("f1", "row F1"),
                       ("event_detection_rate", "incident detection rate"),
                       ("false_alerts_per_cell_day", "false alerts / cell-day"),
                       ("median_delay_minutes", "median delay (minutes)")):
        s = e[key]
        if s["median"] is None:
            print(f"  {label:26s} n/a")
        else:
            print(f"  {label:26s} {s['min']:.3f} / {s['median']:.3f} / {s['max']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="+", default=list(F2_RULE_LEVELS), choices=list(F2_RULE_LEVELS))
    args = ap.parse_args()

    params = load_params()
    print(f"{params['data']['periods_days']}-day datasets | dev seeds {DEV_SEEDS} | eval seeds {EVAL_SEEDS}")
    print(f"configured validation target in params.yaml: {params['threshold']['validation_recall_target']}\n")

    levels = {}
    if RESULTS_PATH.exists():
        levels = json.loads(RESULTS_PATH.read_text()).get("levels", {})
    for level in args.levels:
        levels[level] = run_level(level, params)
        print_level(levels[level])

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "generated_by": "scripts/recall_target_study.py",
            "periods_days": params["data"]["periods_days"],
            "min_recall_floor": params["threshold"]["min_recall"],
            "realistic_level_params": {k: params["data"]["realism"][k] for k in ("severity_min", "benign_to_fault_ratio")},
            "original_readme_metrics": ORIGINAL,
            "dev_seeds": DEV_SEEDS, "eval_seeds": EVAL_SEEDS, "targets": TARGETS,
            "selection_rule": "max median row-level F2 on dev seeds; ties -> smaller target",
            "levels": levels,
        }, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
