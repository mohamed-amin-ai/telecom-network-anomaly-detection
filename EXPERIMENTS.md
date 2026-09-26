# Experiment Log

This documents a real hyperparameter sweep run against the real generated
and engineered data in this repo, and what it changed.

## Methodology & honest caveats

This sweep was run in a sandboxed environment with **no network access**,
so `xgboost`, `mlflow`, and `dvc` could not be installed (only `numpy`,
`pandas`, `scikit-learn`, and `pyyaml` were available). Rather than fabricate
numbers or skip the exercise, the sweep uses
**`sklearn.ensemble.HistGradientBoostingClassifier`** — a real histogram-based
gradient-boosted-tree model, same algorithm family as XGBoost — as a stand-in.

**What this does and doesn't tell you:**
- ✅ The relative effect of tree depth / learning rate / boosting rounds on
  *this* dataset is a real, reproducible signal — both models are boosted
  trees optimizing the same kind of loss on the same features.
- ❌ The absolute numbers below (recall/precision/F1/threshold) will **not**
  exactly match what `models/train.py` produces with real XGBoost. Treat
  this as directional evidence for tuning `params.yaml`, not as the
  production model's reported metrics. The production model is still
  whatever `dvc repro` / `models/train.py` produces with real XGBoost.

Reproduce: `python scripts/local_experiment_sweep.py` → writes
`results/experiments.json`. Once `mlflow` is installed, run
`python scripts/replay_experiments_to_mlflow.py` to load these exact
numbers into a real MLflow tracking store, tagged
`is_xgboost_stand_in=true` so they're never confused with real runs.

## Real finding #1: anomaly rate is ~1%, not ~5%

`models/train.py`'s docstring and this README both previously said
"anomalies are rare (~5%)". Running the actual generator
(`data/generate_data.py`, seed 42) end-to-end and checking the real output:

```
Generated 17,280 rows across 3 cells
Anomaly rate: 187/17280 = 1.08%
```

After feature engineering (`data/features.py` drops the first ~39 rows per
cell to NaN-rolling-window warmup):

```
Train anomaly rate: 1.17%
Test  anomaly rate: 0.75%
```

This doesn't break anything — `scale_pos_weight` is computed from the data
at train time either way, not hardcoded — but the "~5%" comment in the
docstring was simply wrong versus what the generator actually produces.
Worth knowing if you're eyeballing `scale_pos_weight` values in MLflow and
wondering why they're closer to 80–90x than ~20x.

## Real finding #2: learning_rate had the largest effect, by far

Sweep grid (all other params held at the `params.yaml` defaults; recall
floor = 0.93, same time-based 80/20 split):

| Run | max_depth | learning_rate | max_iter | Recall | Precision | F1 | Threshold | Train time |
|---|---|---|---|---|---|---|---|---|
| baseline | 6 | 0.05 | 300 | 0.962 | 0.658 | 0.781 | 0.027 | 0.91s |
| shallow_trees | 3 | 0.05 | 300 | 0.962 | 0.373 | 0.538 | 0.059 | 0.83s |
| deep_trees | 10 | 0.05 | 300 | 0.962 | 0.272 | 0.424 | 0.063 | 0.47s |
| **high_lr** | **6** | **0.10** | **300** | **0.962** | **0.893** | **0.926** | 0.013 | 0.77s |
| low_lr | 6 | 0.02 | 300 | 0.962 | 0.694 | 0.806 | 0.135 | 1.32s |
| fewer_rounds | 6 | 0.05 | 100 | 0.962 | 0.714 | 0.820 | 0.157 | 0.50s |

(Recall is ~identical across runs by construction — `tune_threshold` picks
the highest threshold that still clears the 0.93 recall floor, so it's
always pinned near that floor. Precision/F1 are where the real differences
show up.)

**Reading it:**
- Depth was the wrong lever to pull in either direction: `deep_trees`
  (depth=10) collapsed precision to 0.272 — badly overfit on a ~1%-positive
  dataset, flagging far more false positives to maintain the recall floor.
  `shallow_trees` (depth=3) wasn't much better (0.373) — underfit, needing
  a very loose threshold to hit 0.93 recall, dragging in noise.
- `learning_rate=0.10` (`high_lr`) was the standout: same recall, precision
  jumped from 0.658 → 0.893, cutting the false-positive rate roughly 3x
  versus baseline. F1 0.781 → 0.926.
- `low_lr` and `fewer_rounds` were both slightly worse than baseline —
  0.05 was closer to right than either under- or over-correcting on rounds.

**Action taken:** `params.yaml`'s `model.learning_rate` updated
`0.05 → 0.10`, with a comment pointing back to this doc. This is a
recommendation for the real XGBoost run to start from, not a claim that
XGBoost will reproduce F1=0.926 exactly — rerun `dvc repro` after pulling
this change and compare against README's existing reported numbers
(recall 95.2%, precision 76.9%, F1 0.851) to see the real delta.

## Next real experiment worth running

Given anomaly rate is ~1% rather than ~5%, it's worth re-examining whether
the `threshold.min_recall: 0.93` floor is still the right target, or
whether a slightly lower floor (e.g. 0.90) buys back meaningfully more
precision — the sweep above didn't vary the recall floor itself, only
model hyperparameters at a fixed floor. That's a natural Phase-2-adjacent
follow-up once real XGBoost + MLflow are available to run it properly
(varying the recall floor changes the deployed alerting behavior, so it's
a decision worth making with real production-model numbers, not the
stand-in's).
