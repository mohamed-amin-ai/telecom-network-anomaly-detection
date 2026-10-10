"""
target_window_search.py
------------------------
Search the benchmark difficulty (onset ramp, timeline length) and the validation recall target so that
per-seed results fall inside a stated window: row recall 95-98% AND precision 75-85%.

Protocol (same discipline as calibrate_to_original.py): the choice is made on DEVELOPMENT seeds only
(score = share of dev seeds whose test result lands inside the window); the chosen setting is then
reported on separate EVALUATION seeds. Per-seed rows are kept, because a median hides exactly the
spread this search is trying to control.

  python scripts/target_window_search.py search          # dev seeds, writes results/target_window_search.json
  python scripts/target_window_search.py evaluate --ramp R --days D --target T

Uses the scikit-learn stand-in (xgboost isn't installable in the build sandbox).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from models.common import load_params
from scripts.study_common import apply_level, evaluate_current_dataset

DEV_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107]
EVAL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RAMPS = [0.0, 0.025, 0.05, 0.075]
DAYS = [365, 730]
TARGETS = [0.9, 0.92, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98]
RECALL_WINDOW, PRECISION_WINDOW = (0.95, 0.98), (0.75, 0.85)
OUT = Path(__file__).parent.parent / "results" / "target_window_search.json"


def in_window(b: dict) -> bool:
    return (RECALL_WINDOW[0] <= b["recall"] <= RECALL_WINDOW[1]) and (PRECISION_WINDOW[0] <= b["precision"] <= PRECISION_WINDOW[1])


def one(task):
    days, ramp, seed = task
    params = load_params()
    apply_level("calibrated", days=days, seed=seed, realism_overrides={"severity_min": 1.0, "ramp_fraction": ramp})
    r = evaluate_current_dataset(params, TARGETS)
    r["seed"], r["days"], r["ramp"] = seed, days, ramp
    print(f"days {days} ramp {ramp} seed {seed}: PR-AUC {r['test_pr_auc']:.3f}", flush=True)
    return r


def summarize(rows):
    out = {}
    for t in TARGETS:
        bt = [r["by_target"][str(t)] for r in rows]
        out[str(t)] = {
            "in_window": round(float(np.mean([in_window(b) for b in bt])), 3),
            "recall_in": round(float(np.mean([RECALL_WINDOW[0] <= b["recall"] <= RECALL_WINDOW[1] for b in bt])), 3),
            "precision_in": round(float(np.mean([PRECISION_WINDOW[0] <= b["precision"] <= PRECISION_WINDOW[1] for b in bt])), 3),
            **{f"{k}_{s}": round(float(fn([b[k] for b in bt])), 3) for k in ("recall", "precision", "f1")
               for s, fn in (("median", np.median), ("min", np.min), ("max", np.max))},
            "event_detection_rate_min": round(float(np.min([b["event_detection_rate"] for b in bt])), 3),
            "false_alerts_per_cell_day_median": round(float(np.median([b["false_alerts_per_cell_day"] for b in bt])), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["search", "evaluate"])
    ap.add_argument("--ramp", type=float)
    ap.add_argument("--days", type=int)
    ap.add_argument("--target", type=float)
    a = ap.parse_args()
    if a.mode == "search":
        tasks = [(d, r, s) for d in DAYS for r in RAMPS for s in DEV_SEEDS]
        with Pool(2) as p:
            rows = p.map(one, tasks, chunksize=1)
        res = {"dev_seeds": DEV_SEEDS, "windows": {"recall": RECALL_WINDOW, "precision": PRECISION_WINDOW}, "targets": TARGETS, "configs": {}}
        for d in DAYS:
            for r in RAMPS:
                sub = [x for x in rows if x["days"] == d and x["ramp"] == r]
                res["configs"][f"days={d},ramp={r}"] = {"summary": summarize(sub), "rows": sub}
        OUT.write_text(json.dumps(res, indent=1))
        best = sorted(((v["summary"][t]["in_window"], -abs(v["summary"][t]["precision_median"] - 0.8), k, t)
                       for k, v in res["configs"].items() for t in map(str, TARGETS)), reverse=True)[:8]
        for iw, _, k, t in best:
            s = res["configs"][k]["summary"][t]
            print(f"{k:22s} target {t:5s} in-window {iw:.2f}  recall {s['recall_median']:.3f} ({s['recall_min']:.3f}-{s['recall_max']:.3f})  "
                  f"precision {s['precision_median']:.3f} ({s['precision_min']:.3f}-{s['precision_max']:.3f})")
    else:
        with Pool(2) as p:
            rows = p.map(one, [(a.days, a.ramp, s) for s in EVAL_SEEDS], chunksize=1)
        s = summarize(rows)[str(a.target)]
        s["per_seed"] = {r["seed"]: {k: r["by_target"][str(a.target)][k] for k in ("recall", "precision", "f1", "event_detection_rate", "false_alerts_per_cell_day", "median_delay_minutes")}
                          | {"pr_auc": r["test_pr_auc"], "test_events": r["test_events"], "test_positive_rows": r["test_positive_rows"]} for r in rows}
        s["pr_auc_median"] = round(float(np.median([r["test_pr_auc"] for r in rows])), 3)
        eval_out = OUT.with_name("target_window_eval.json")
        eval_out.write_text(json.dumps({"ramp": a.ramp, "days": a.days, "target": a.target, "eval_seeds": EVAL_SEEDS, "summary": s}, indent=1))
        print(json.dumps({k: v for k, v in s.items() if k != "per_seed"}, indent=1))
        eval_out.write_text(json.dumps({"ramp": a.ramp, "days": a.days, "target": a.target, "eval_seeds": EVAL_SEEDS, "summary": s}, indent=1, default=float))
        for sd, v in s["per_seed"].items():
            print(sd, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
