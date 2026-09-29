# Experiment Log

Real experiments run against this repo's generated data, and — more
importantly — what they revealed about the repo's own claims. Read the
**Retractions** section first: several things this project previously
stated turned out to be wrong.

## Methodology & caveats

The sandbox these experiments ran in had **no network access**, so `xgboost`,
`mlflow`, and `dvc` could not be installed. The experiments use
`sklearn.ensemble.HistGradientBoostingClassifier` (same algorithm family as
XGBoost) as a stand-in. That is adequate for the questions below — how stable
the *evaluation* is, and whether config differences are real — because those
are properties of the dataset and protocol, not of which boosted-tree library
is used. It is **not** the production model: absolute numbers from real
XGBoost (`models/train.py`, logged to MLflow) will differ.

Protocol (identical in `models/train.py` and the scripts here): chronological
train / validation / test split (`params.yaml` → `split`); scaler fit on train
only; early stopping and the decision threshold use **validation only**; the
test set is touched once, for the reported numbers. Threshold = highest
threshold reaching the recall floor (`threshold.min_recall`) on validation.

Reproduce: `python scripts/local_experiment_sweep.py` and
`python scripts/seed_variance_check.py` (write `results/*.json`).
`scripts/replay_experiments_to_mlflow.py` loads the sweep into a real MLflow
store once mlflow is installed, tagged `is_xgboost_stand_in=true`.

## Retractions

**1. The generator was not deterministic.** `data/generate_data.py` seeded each
cell's RNG from `hash(cell) % 1000`. Python randomizes `str` hashing per
process, so every fresh run produced a *different* dataset despite the pinned
`seed: 42`. Rerunning the script five times gave anomaly counts of 187, 200,
254, 195 and 287. It was noticed only by rerunning the script during final
validation. Fixed (`enumerate` index instead of `hash`, and the module-level
injector RNG is reseeded per build); `tests/test_generate_data.py` runs the
generator in separate processes under different `PYTHONHASHSEED` values and
requires identical output. The test was confirmed to *fail* when the bug is
reintroduced. Consequences:
  - Every number in the previous version of this file came from one
    unreproducible draw.
  - "DVC reproducibility" and the Airflow DAG's "fixed seed" claim were false
    until this fix.
  - The README's original headline (recall 95.2% / precision 76.9% /
    F1 0.851) cannot be reproduced from anything in this repo. It is
    consistent with the *range* below, but is one draw from it.

**2. "`learning_rate` 0.10 lifts F1 from 0.781 to 0.926" — retracted.** That came
from the unreproducible draw. Re-run on the deterministic dataset with a proper
protocol, config differences are within noise (see the sweep table). `params.yaml`
is back at `learning_rate: 0.05`.

**3. `train.py` evaluated on the data it tuned on.** The original code used the
test set for early stopping (`eval_set`), for choosing the threshold, and for
reporting. That makes "recall >= 0.93 on the test set" true *by construction*,
not a measured result, and contradicts the README's statements that the
threshold was tuned "on validation set" and that the test set was "never seen
during training". Fixed with a real validation slice
(`models/common.py::three_way_time_split`, unit-tested — including that the
test window is unchanged, so ONNX parity / Triton / drift code see the same
test rows).

**4. "~1% anomaly rate" was itself a one-draw number.** On the deterministic
dataset the rate is 2.16% (train), 0.47% (validation),
2.20% (test). It is not ~5% as the original comments claimed, but it
also isn't a stable ~1%: it varies with the draw.

## The core problem: very few independent anomaly events

The generator injects roughly 6-9 anomaly *events* per cell regardless of
timeline length. On the default seed the windows contain:

| Window | Rows | Anomalous rows | Independent events |
|---|---|---|---|
| Train | 11,037 | 238 | 17 |
| Validation | 2,760 | 13 | 2 |
| Test | 3,450 | 76 | 6 |

A recall floor tuned on 2 validation events, then measured on 6 test events, is
dominated by whether one or two hard events (e.g. a mild throughput drop that
sits inside normal variation) fall in each window.

## Sweep on the default seed (42)

Baseline model class: HistGradientBoostingClassifier stand-in. Configs are
ranked by **validation** PR-AUC (threshold-free); test columns are reported
alongside, never used to choose. Recall floor = 0.93.

| Run | max_depth | learning_rate | max_iter | val PR-AUC | test PR-AUC | test recall | test precision | test F1 |
|---|---|---|---|---|---|---|---|---|
| baseline | 6 | 0.05 | 300 | 0.377 | 0.678 | 0.829 | 0.176 | 0.291 |
| shallow_trees | 3 | 0.05 | 300 | 0.509 | 0.594 | 0.632 | 0.075 | 0.134 |
| deep_trees | 10 | 0.05 | 300 | 0.444 | 0.706 | 0.895 | 0.134 | 0.233 |
| high_lr | 6 | 0.1 | 300 | 0.424 | 0.694 | 0.816 | 0.205 | 0.327 |
| low_lr | 6 | 0.02 | 300 | 0.465 | 0.594 | 0.658 | 0.263 | 0.376 |
| fewer_rounds | 6 | 0.05 | 100 | 0.446 | 0.547 | 0.645 | 0.297 | 0.407 |

**Reading it:** validation ranks `shallow_trees` first, but on test it is
near-last; `deep_trees` is best on test. With 2 validation events and 6 test
events these orderings are noise. **No hyperparameter conclusion is supported.**
Note also that no config reaches the 0.93 recall floor on the test window —
the floor met on validation did not generalize.

## Spread across generator seeds

Same baseline config, same protocol, 8 differently-seeded datasets
(`scripts/seed_variance_check.py`):

| seed | val events | test events | test +rows | test PR-AUC | test recall | test precision | test F1 |
|---|---|---|---|---|---|---|---|
| 42 (default) | 2 | 6 | 76 | 0.678 | 0.829 | 0.176 | 0.291 |
| 1 | 4 | 3 | 73 | 0.872 | 0.767 | 0.727 | 0.747 |
| 2 | 5 | 3 | 56 | 0.982 | 0.821 | 1.000 | 0.902 |
| 3 | 5 | 3 | 42 | 0.972 | 1.000 | 0.066 | 0.123 |
| 4 | 5 | 4 | 35 | 0.795 | 0.829 | 0.580 | 0.682 |
| 5 | 4 | 4 | 44 | 0.933 | 0.977 | 0.391 | 0.558 |
| 6 | 4 | 4 | 42 | 0.975 | 0.929 | 0.886 | 0.907 |
| 7 | 4 | 3 | 28 | 0.977 | 0.893 | 1.000 | 0.943 |

| Metric | min | median | max |
|---|---|---|---|
| test PR-AUC | 0.678 | 0.953 | 0.982 |
| test recall | 0.767 | 0.861 | 1.0 |
| test precision | 0.066 | 0.654 | 1.0 |
| test F1 | 0.123 | 0.715 | 0.943 |

**Reading it:**
  - The model has real signal: median test PR-AUC is 0.953 against a base
    rate of about 2%.
  - But threshold-dependent metrics are unstable: precision ranges from
    0.066 to 1.0, F1 from 0.123 to 0.943, driven by
    threshold tuning on a handful of validation events.
  - The default seed (42) happens to be the *worst* draw for PR-AUC. It was
    deliberately **not** changed to get nicer numbers — that would be
    cherry-picking.

## What this suggests (not done here)

  - Scale the number of injected events with timeline length (or generate many
    more days), so validation/test each contain dozens of events.
  - Tune the threshold with rolling-origin / blocked cross-validation across
    events rather than one small validation slice.
  - Report PR-AUC (stable) alongside recall/precision at a threshold (unstable),
    and quote ranges, not single numbers.
  - Treat the 0.93 recall floor as a validation-time target and the promotion
    gate (`models/promote.py`, which checks *test* recall) as the honest check:
    on many draws it will correctly decline to promote.

## Finding: a flat KPI silently drops its row (found writing tests)

While writing `tests/test_features.py`, an early version of the synthetic
fixture derived `packet_loss_pct` from a per-cell `base` value
(`base/100 + noise`, clipped to `[0, 5]`). For a cell with `base=1000`,
that clip saturates every single row to the ceiling of `5.0` — zero
variance for the whole cell.

`engineer_features` computes z-score as
`(series - roll_mean_4h) / roll_std_4h.replace(0, np.nan)` — the
`replace(0, np.nan)` guards against divide-by-zero, but it means a KPI
that's genuinely flat for a full 4h window gets `zscore = NaN` for every
row in that window, and the final `dropna()` then drops those rows
entirely. In the test, this deleted **100% of one cell's rows** — not a
few warmup rows, all of them, for as long as the flat streak lasted.

This was a bug in the test fixture, not in `features.py` — real telecom
KPIs are essentially never perfectly flat for 4 hours. But it's a real
latent behavior worth knowing: if a KPI's value did get stuck (e.g. a
stuck sensor reporting the same reading repeatedly, a genuinely
alert-worthy fault condition), the current code would silently drop that
row from training data, and at serving time `live_features.py`'s
`add_and_compute` would return `None` (empty `feat_df` after dropna) →
the API would report `425 Too Early` / "not enough history" for a cell
that actually has plenty of history — just degenerate variance. The
error message would be misleading about the actual cause.

**Not fixed here** (out of scope for a test-writing pass — it's a
product decision, not a bug fix): options would be (a) treat `std==0` as
`zscore=0` instead of `NaN` (a flat series is, definitionally, exactly at
its own mean), or (b) keep it as `NaN` but have `live_features.py` surface
a distinct error ("degenerate variance" vs "insufficient history") rather
than conflating the two under the same 425. Flagging as a known
limitation for Phase 2+ rather than silently patching it.
