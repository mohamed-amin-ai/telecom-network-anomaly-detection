# Experiment Log

Experiments run against this repo's generated data — and, more importantly,
what they revealed about the repo's own claims. Read **Retractions**, **What the
original headline was resting on**, and **Which number to quote** first.

## Methodology & caveats

The sandbox these experiments ran in had **no network access**, so `xgboost`,
`mlflow`, and `dvc` could not be installed. The experiments use
`sklearn.ensemble.HistGradientBoostingClassifier` (same algorithm family as
XGBoost) as a stand-in, fitted the way `models/train.py` does: positives
upweighted by neg/pos, boosting stopped by **validation** PR-AUC with patience 30
(`scripts/study_common.py::fit_stand_in`). It is **not** the production model; XGBoost
numbers (logged to MLflow, uploaded by CI) will differ.

Protocol (identical in `models/train.py` and the scripts): chronological
train / validation / test split; scaler fit on train only; early stopping and the
decision threshold use **validation only**; the test set is touched once.

Reproduce (each writes `results/*.json`): `scripts/calibrate_to_original.py`,
`scripts/recall_target_study.py` (levels, operating point), `scripts/realism_study.py`
(ablation), `scripts/local_experiment_sweep.py`, `scripts/seed_variance_check.py`. The studies
read the dataset you generated, so run `python data/generate_data.py && python data/features.py`
first (this bit me once: a sweep silently ran on a stale file from an older generator).

## Retractions

**1. The generator was not deterministic.** It seeded each cell's RNG from
`hash(cell) % 1000`; Python randomizes `str` hashing per process, so every run produced a
different dataset despite `seed: 42` (five reruns: 187, 200, 254, 195, 287 anomalies).
Fixed; `tests/test_generate_data.py` runs the generator under different `PYTHONHASHSEED`
values and was confirmed to fail when the bug is reintroduced.

**2. "`learning_rate` 0.10 lifts F1 from 0.781 to 0.926" — retracted** (one unreproducible
draw). `params.yaml` stays at 0.05.

**3. `train.py` evaluated on the data it tuned on** (early stopping, threshold, and report
all on the test set). Fixed with a real validation slice.

**4. Which benchmark is the headline has changed three times**, and you should know why. One
revision called the original metrics "met or beaten" on the **clean** benchmark; the next made the
**realistic** benchmark the headline because clean is too easy to say anything about production; the
current default is the **calibrated** benchmark, requested because the clean one overshoots the
original's precision and F1, and tuned last to a realistic window (row recall 95–98%, precision 75–85%). None of these is wrong, but they answer different questions, and
all three are reported side by side. See **Which number to quote**.

**5. A hypothesis I got wrong.** When the realistic profile first produced very low PR-AUC
(0.39 on one seed), I suspected my stand-in was overfitting (300 rounds, no early stopping)
and made it follow `train.py`'s validation early stopping. PR-AUC barely moved (0.392 → 0.388):
the difficulty is in the data, not the stand-in. The fix is kept because it is more faithful.

## What the original headline was resting on

The original README reported recall 95.2%, precision 76.9%, F1 0.851. The original
`model.json`/`metadata.json` pin down what those came from: `scale_pos_weight = 89.1764679`
on 13,797 rows means **153** anomalous training rows, and recall 0.9524 = **20/21**, precision
0.7692 = **20/26**, F1 0.8511 reproduce to four decimals from TP=20, FP=6, FN=1. So the
headline was **one test window with 21 anomalous rows**; its exact 95% intervals are recall
[0.76, 1.00] and precision [0.56, 0.91]. The exact three numbers can't be
reproduced (one draw); the level can — see **Calibration** below.

## Which number to quote

| If you are asked… | Quote | Why |
|---|---|---|
| "How does it compare to the original project?" | the **calibrated** row: recall 0.96 / precision 0.78 / F1 0.87 (median of 10 seeds; range in the table), **and say the benchmark was tuned to a target window** | the benchmark's difficulty was chosen to reproduce the original's level and the final setting was picked by the owner for recall > precision, so landing near the original is by construction |
| "Is the model good?" | **don't answer with the calibrated numbers.** Use the realistic row as the more honest stress test | calibration can't show quality: the target was built in |
| "How will it perform in production?" | **none of these.** Nothing here is calibrated to a real network | all three are synthetic |

## Calibration to the original metrics

**What and why.** On the clean benchmark recall already matches the original (~0.95) but precision
is ~0.96, far above 0.77 — and the threshold can't fix it: pushing precision down to 0.77 pushes recall
up to ~0.975. The original point lies *below* the clean precision–recall curve, so the data has to be a
little harder than clean. `scripts/calibrate_to_original.py` searches two knobs — how gradually faults
begin (`ramp_fraction`; every fault at full strength) and the operating point — to put the median
(recall, precision) over **development** seeds closest to the original. F1 is *not* a target; it follows
from the other two, so its closeness is a genuine consistency check.

**First attempt, kept on record.** The dial was fault faintness (`severity_min`) with the default ramp.
That overshot: `severity_min=1.0` already gave recall 0.860 / precision 0.682 (distance 0.127), and 0.8 was
worse, because the ramp alone is too hard. The dial moved to the ramp itself, and two refinement values
(0.075, 0.125) were added after the coarse pass showed where the best region was.

Development seeds [100, 101, 102, 103, 104, 105, 106, 107] — best operating point per ramp fraction:

| ramp_fraction | best target | recall | precision | F1 | distance to original | median PR-AUC |
|---|---|---|---|---|---|---|
| 0.0 | 0.975 | 0.977 | 0.804 | 0.885 | 0.043 | 0.982 |
| 0.05 | 0.97 | 0.972 | 0.839 | 0.904 | 0.073 | 0.979 |
| 0.075 | 0.96 | 0.964 | 0.726 | 0.826 | 0.045 | 0.965 |
| 0.1 | 0.95 | 0.953 | 0.729 | 0.824 | 0.040 | 0.956 |
| 0.125 | 0.94 | 0.940 | 0.697 | 0.799 | 0.073 | 0.950 |
| 0.15 | 0.92 | 0.917 | 0.873 | 0.904 | 0.110 | 0.940 |
| 0.2 | 0.9 | 0.905 | 0.759 | 0.829 | 0.048 | 0.923 |
| 0.3 | 0.85 | 0.860 | 0.682 | 0.758 | 0.127 | 0.874 |

### Second stage: a realistic window (current setting)

The owner then asked for realistic ranges instead of a point match: **row recall 95–98%, precision 75–85%**.
`scripts/target_window_search.py` searched the onset ramp (0, 0.025, 0.05, 0.075), the timeline length (365 or 730 days) and
the validation recall target on **development** seeds, scoring each by the share of seeds whose test result lands inside the
window (per-seed rows in `results/target_window_search.json`).

**What the search showed.** No setting puts most seeds inside both windows: the best reached 38% of dev seeds
(`days=730,ramp=0.075`, target 0.96). Recall is controllable — at 730 days / ramp 0.075 and targets 0.96–0.97 *every* dev seed
has recall inside 95–98% — but precision at a fixed recall depends on how many false-alert bursts a seed happens to contain,
and swings widely from seed to seed. Dev results for the chosen timeline/ramp (8 seeds):

| target | recall median (min–max) | precision median (min–max) | recall in window | precision in window | both |
|---|---|---|---|---|---|
| 0.95 | 0.955 (0.926–0.973) | 0.930 (0.798–0.989) | 5/8 | 1/8 | 1/8 |
| 0.955 | 0.959 (0.940–0.974) | 0.882 (0.702–0.985) | 7/8 | 1/8 | 1/8 |
| 0.96 | 0.965 (0.951–0.974) | 0.838 (0.595–0.944) | 8/8 | 3/8 | 3/8 |
| 0.965 | 0.965 (0.956–0.976) | 0.790 (0.541–0.937) | 8/8 | 2/8 | 2/8 |
| 0.97 | 0.971 (0.956–0.978) | 0.653 (0.492–0.820) | 8/8 | 1/8 | 1/8 |

**Chosen on dev seeds:** 730 days, `ramp_fraction` 0.075, target **0.965** — recall fully inside the window and median
precision closest to 80% (0.790). Longer timelines were needed: they give each seed more events and tighter recall.

**Held-out result** on the 10 evaluation seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9] (not used in the search), median (min–max):

| metric | original | this setting |
|---|---|---|
| recall | 0.952 | **0.964 (0.948–0.972)** |
| precision | 0.769 | **0.784 (0.598–0.927)** |
| F1 | 0.851 | **0.866 (0.739–0.938)** |

Recall is inside 95–98% on 9/10 seeds, precision inside 75–85% on 5/10, and both on **5/10**. Medians sit inside both
windows; individual seeds do not always. Per seed:

| seed | recall | precision | F1 | false alerts / cell-day |
|---|---|---|---|---|
| 42 | 0.970 | 0.795 | 0.874 | 0.33 |
| 1 | 0.962 | 0.823 | 0.887 | 0.26 |
| 2 | 0.960 | 0.669 | 0.788 | 0.66 |
| 3 | 0.960 | 0.683 | 0.798 | 0.70 |
| 4 | 0.948 | 0.927 | 0.938 | 0.11 |
| 5 | 0.969 | 0.827 | 0.892 | 0.30 |
| 6 | 0.968 | 0.598 | 0.739 | 0.73 |
| 7 | 0.965 | 0.773 | 0.859 | 0.40 |
| 8 | 0.972 | 0.747 | 0.845 | 0.45 |
| 9 | 0.963 | 0.839 | 0.896 | 0.26 |

The default dataset seed is 42 (recall 0.970, precision 0.795), which happens to be one of the seeds inside both windows; it is
not chosen for that reason, but it is a favourable draw, so quote the median and range, not that seed alone.

**What this does and does not mean.**
  - The window is a *target* set by the owner, so landing in it is by construction. It is **not** evidence the model is good,
    and the numbers come from a scikit-learn stand-in, not XGBoost.
  - Incident detection is 100% at ~0.37 false alerts per cell-day (median; 0.11–0.73), detection delay 0 minutes.
  - Precision's per-seed spread (0.60–0.93) is the honest uncertainty: a single run can land well outside 75–85%.
  - The first-stage result (ramp 0.0 / target 0.975 at 365 days: recall 0.971, precision 0.909) is kept in
    `results/calibration_to_original.json` under `previous_selection`.

## Why evaluation windows were thin — and the fix

The generator injected a fixed 6-9 anomaly events per cell *regardless of `periods_days`*,
so a longer timeline only added normal rows (old default: 2 validation events, 6 test events).
Events now **scale with timeline length** (the 60-day clean dataset is byte-identical to
before; `tests/test_generate_data.py` guards the scaling) and `periods_days` is now 730 for the
configured benchmark: roughly 54 test events and 669 anomalous test rows per evaluation seed (the clean/realistic
study rows below were run at 365 days).

## The realistic profile

`params.yaml` → `data.profile: realistic` adds four independently switchable factors:

| Factor | What it models |
|---|---|
| `nonstationarity` | per-cell baselines (x0.8–1.25), weekly seasonality, slow drift (latency +10%, throughput -8% over the year) |
| `severity_and_ramp` | fault severity log-uniform on [0.4, 1] of the clean magnitude (many faults are faint), with gradual onset over the first 30% of each event |
| `label_noise` | ticket-style labels: start/end jittered by up to ±2 intervals, 10% of faults never logged |
| `benign_surges` | real but non-fault load surges (latency up, throughput down, users up) at 0.5 per fault, labeled normal |

(`calibrated` is the same code path with only `severity_and_ramp` on, every fault at full strength,
and the ramp set to 0.075.) Fault positions, types and durations are identical in every profile (a test
enforces it), and with all switches off `realistic` equals `legacy` exactly (also tested). Event metadata
goes in a sidecar (`data/events.csv`), never in `raw_kpis.csv`: any extra column there would silently
become a model feature.

**None of these values is calibrated to a real network** — none was available. One pair has a history:
`severity_min 0.15` / `benign_to_fault_ratio 1.0` were fixed before any model was trained and gave a
near-unusable detector (PR-AUC ≈ 0.2) — kept as the **harsh** level — then were revised once, post hoc, to
0.4 / 0.5, the **realistic** level.

**To switch benchmark:** set `data.profile` *and* `threshold.validation_recall_target` together
(`legacy` → 0.96, `calibrated` → 0.965, `realistic` → 0.5); a test fails if only one changes.

## Results by difficulty level

For clean / realistic / harsh the operating point was chosen by a rule stated in advance — **maximise
median row-level F2 on development seeds [100, 101, 102, 103, 104, 105, 106, 107]** (ties → smaller target) — then reported on disjoint
evaluation seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]; the calibrated row's setting comes from the window search above (730-day timeline) instead.
Median (min–max) over 10 seeds; the test window is held out. A **false alert** is a merged run of
flagged intervals matching no real incident (±2 intervals); delay is time from onset to first flag.

| Level | target | PR-AUC | row recall | row precision | row F1 | incident detection | false alerts / cell-day | delay (min) |
|---|---|---|---|---|---|---|---|---|
| **calibrated (window search, ramp 0.075, 730 d) ← configured** | 0.965 | **0.97 (0.97–0.98)** | **0.96 (0.95–0.97)** | **0.78 (0.60–0.93)** | **0.87 (0.74–0.94)** | **100%** (100%–100%) | **0.37** (0.11–0.73) | **0** (0–0) |
| clean (legacy generator) | 0.96 | 0.98 (0.96–0.99) | 0.95 (0.88–0.98) | 0.96 (0.81–0.99) | 0.95 (0.87–0.97) | 100% (100%–100%) | 0.05 (0.01–0.30) | 0 (0–15) |
| realistic | 0.5 | 0.56 (0.39–0.66) | 0.59 (0.37–0.70) | 0.50 (0.32–0.79) | 0.53 (0.42–0.66) | 96% (67%–100%) | 0.36 (0.08–0.80) | 38 (30–75) |
| harsh (original realism setting) | 0.2 | 0.27 (0.11–0.31) | 0.26 (0.08–0.36) | 0.35 (0.20–0.97) | 0.28 (0.15–0.36) | 52% (18%–76%) | 0.33 (0.01–0.86) | 64 (30–165) |

Original README, for comparison: recall 0.952, precision 0.769, F1 0.851 (one 21-row window).

**Reading it:** calibrated sits in the requested window by construction; clean overshoots precision/F1; realistic is a deployed-detector-shaped
stress test (per cell it would miss ~41% of anomalous rows, yet catch ~96% of incidents ~38 minutes after
onset at ~0.36 false alerts per cell-day); harsh is the original, too-severe realism setting.

## Operating points

Evaluation seeds, medians. "Flagged" is the share of all test rows flagged.

**Clean benchmark**

| validation target | row recall | row precision | flagged | incident detection | false alerts / cell-day | delay (min) | seeds with recall ≥ 0.93 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.14 | 1.00 | 0% | 33% | 0.00 | 75 | 0/10 |
| 0.2 | 0.23 | 1.00 | 0% | 45% | 0.00 | 60 | 0/10 |
| 0.3 | 0.32 | 1.00 | 0% | 62% | 0.00 | 52 | 0/10 |
| 0.4 | 0.39 | 1.00 | 1% | 71% | 0.00 | 52 | 0/10 |
| 0.5 | 0.45 | 1.00 | 1% | 82% | 0.00 | 45 | 0/10 |
| 0.6 | 0.54 | 1.00 | 1% | 93% | 0.00 | 33 | 0/10 |
| 0.7 | 0.64 | 1.00 | 1% | 95% | 0.00 | 30 | 0/10 |
| 0.8 | 0.78 | 1.00 | 1% | 98% | 0.00 | 15 | 0/10 |
| 0.9 | 0.89 | 1.00 | 1% | 100% | 0.00 | 15 | 0/10 |
| 0.93 | 0.92 | 0.99 | 2% | 100% | 0.01 | 15 | 3/10 |
| 0.96 ← chosen | 0.95 | 0.96 | 2% | 100% | 0.05 | 0 | 8/10 |
| 0.99 | 0.99 | 0.47 | 3% | 100% | 1.42 | 0 | 10/10 |

**Realistic benchmark**

| validation target | row recall | row precision | flagged | incident detection | false alerts / cell-day | delay (min) | seeds with recall ≥ 0.93 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.13 | 0.89 | 0% | 32% | 0.01 | 116 | 0/10 |
| 0.2 | 0.25 | 0.89 | 0% | 51% | 0.01 | 60 | 0/10 |
| 0.3 | 0.34 | 0.76 | 1% | 72% | 0.04 | 60 | 0/10 |
| 0.4 | 0.47 | 0.67 | 1% | 83% | 0.15 | 45 | 0/10 |
| 0.5 ← chosen | 0.59 | 0.50 | 2% | 96% | 0.36 | 37 | 0/10 |
| 0.6 | 0.69 | 0.32 | 3% | 96% | 1.09 | 30 | 0/10 |
| 0.7 | 0.73 | 0.14 | 8% | 100% | 2.65 | 18 | 0/10 |
| 0.8 | 0.80 | 0.06 | 18% | 100% | 4.93 | 0 | 0/10 |
| 0.9 | 0.91 | 0.02 | 53% | 100% | 4.85 | 0 | 3/10 |
| 0.93 | 0.95 | 0.02 | 71% | 100% | 3.28 | 0 | 7/10 |
| 0.96 | 0.98 | 0.02 | 85% | 100% | 1.58 | 0 | 10/10 |
| 0.99 | 0.99 | 0.01 | 95% | 100% | 0.42 | 0 | 10/10 |

## What this means for promotion

`params.yaml`'s `min_recall: 0.93` is the test-recall floor `models/promote.py` requires.

  - **Calibrated (configured):** 10/10 evaluation seeds meet it at target 0.965 (the clean level meets it on 8/10).
  - **Clean:** 8/10 at its chosen target (0.96).
  - **Realistic:** 0/10 at its chosen target (0.5); forcing 0.96 gives median recall 0.98 at precision 0.02 while flagging
    85% of all rows. The gate would decline essentially every model there.

I have **not** lowered the floor: it reads as a business requirement, and tuning a gate to whatever the model
achieves would defeat it. On realistic data your options are: accept that the gate never opens, set a floor the data
can support, or gate on **incident-level** detection plus a false-alert budget (`models/alerting.py` computes both).

## Which realism factors cost what (cumulative ablation)

Factors added one at a time, in this order, at the realistic level's own operating point (0.5); 6 seeds,
median (min–max). Order matters (factors interact) — one path, not a unique attribution. PR-AUC is threshold-free and
the cleanest column; the others are at a fixed target, not each profile's own optimum.

| Profile | PR-AUC | row recall | row precision | incident detection | false alerts / cell-day |
|---|---|---|---|---|---|
| clean (legacy) | 0.98 (0.97–0.99) | 0.46 (0.32–0.56) | 1.00 (1.00–1.00) | 0.82 (0.50–0.89) | 0.00 (0.00–0.00) |
| + nonstationarity | 0.98 (0.97–0.99) | 0.46 (0.38–0.61) | 1.00 (1.00–1.00) | 0.80 (0.64–0.89) | 0.00 (0.00–0.00) |
| + severity & ramp | 0.72 (0.64–0.75) | 0.62 (0.51–0.67) | 0.88 (0.47–0.98) | 0.86 (0.78–0.96) | 0.09 (0.01–0.65) |
| + label noise | 0.54 (0.49–0.61) | 0.59 (0.50–0.61) | 0.55 (0.45–0.79) | 0.88 (0.83–0.96) | 0.32 (0.12–0.37) |
| + benign surges (full) | 0.56 (0.45–0.66) | 0.61 (0.49–0.70) | 0.47 (0.32–0.73) | 0.96 (0.83–0.96) | 0.46 (0.13–0.80) |

Faint, gradually-starting faults and noisy labels account for nearly all of the drop in PR-AUC; non-stationarity is nearly
free here, and benign surges mainly add false alerts (that step's PR-AUC change is inside seed-to-seed noise).

## Sweep on the default seed (42) — calibrated benchmark (ramp 0.075, 730 days)

Test columns reported alongside validation PR-AUC, never used to choose. Target 0.965.

| Run | max_depth | learning_rate | max_iter | val PR-AUC | test PR-AUC | test recall | test precision | test F1 |
|---|---|---|---|---|---|---|---|---|
| baseline | 6 | 0.05 | 300 | 0.974 | 0.977 | 0.970 | 0.795 | 0.874 |
| shallow_trees | 3 | 0.05 | 300 | 0.976 | 0.976 | 0.965 | 0.838 | 0.897 |
| deep_trees | 10 | 0.05 | 300 | 0.976 | 0.978 | 0.966 | 0.880 | 0.921 |
| high_lr | 6 | 0.1 | 300 | 0.975 | 0.977 | 0.968 | 0.757 | 0.850 |
| low_lr | 6 | 0.02 | 300 | 0.971 | 0.976 | 0.970 | 0.745 | 0.843 |
| fewer_rounds | 6 | 0.05 | 100 | 0.972 | 0.977 | 0.973 | 0.670 | 0.794 |

All configs are within a point or two of each other on PR-AUC, so no hyperparameter conclusion is supported; `params.yaml` is unchanged.

## Spread across generator seeds — calibrated benchmark, target 0.965

| seed | val events | test events | test +rows | test PR-AUC | test recall | test precision | test F1 |
|---|---|---|---|---|---|---|---|
| 42 (default) | 42 | 53 | 625 | 0.977 | 0.970 | 0.795 | 0.874 |
| 1 | 48 | 60 | 660 | 0.973 | 0.962 | 0.823 | 0.887 |
| 2 | 49 | 60 | 652 | 0.969 | 0.960 | 0.669 | 0.788 |
| 3 | 50 | 63 | 740 | 0.968 | 0.960 | 0.683 | 0.798 |
| 4 | 45 | 55 | 675 | 0.976 | 0.948 | 0.927 | 0.938 |
| 5 | 42 | 53 | 699 | 0.979 | 0.969 | 0.827 | 0.892 |
| 6 | 37 | 45 | 565 | 0.967 | 0.968 | 0.598 | 0.739 |
| 7 | 43 | 52 | 664 | 0.972 | 0.965 | 0.773 | 0.859 |

| Metric | min | median | max |
|---|---|---|---|
| test PR-AUC | 0.967 | 0.973 | 0.979 |
| test recall | 0.948 | 0.964 | 0.97 |
| test precision | 0.598 | 0.784 | 0.927 |
| test F1 | 0.739 | 0.866 | 0.938 |

## Caveats that remain

  - **The calibrated numbers are tuned to a window chosen by the owner (recall 95–98%, precision 75–85%), by construction.** It is a mid-difficulty benchmark between clean and realistic; the realistic one is not calibrated to a real network (parameters are my judgment, one pair revised after
    seeing results). None is a production forecast.
  - **The search used 8 dev seeds and a coarse grid**; no setting put most seeds inside both windows, so precision stays seed-dependent.
  - **Not modeled:** missing readings and stuck sensors, correlated multi-KPI outages, regime changes beyond smooth drift,
    delayed or partial label arrival, and real operator labeling behaviour.
  - **Stand-in model**, not XGBoost. XGBoost numbers come from CI (`model-artifacts`, `mlflow-tracking-store`), and will
    differ from every table here.
  - **The operating-point rules are choices.** An alert budget — the threshold that keeps false alerts per cell-day under a
    limit — is how production usually sets it; the operating-point tables let you read that off.

## Finding: undefined features silently dropped rows — and served stale features (fixed)

**Corrects an earlier version of this section**, which predicted that a dropped
row would make the API return a misleading `425`. That prediction was wrong; the
real behavior was worse.

`engineer_features` drops any row containing a `NaN` feature. Two ordinary
situations produce one: (a) the *previous* reading's value for a KPI is exactly
`0` (rate of change divides by zero — in practice `packet_loss_pct == 0.0`), and
(b) a KPI is perfectly flat over the 4h window (zero std, so z-score is undefined).

In training this just removes a few rows: 9 non-warmup rows of 17,280 on the 60-day clean
dataset, 56 of 105,120 on the 365-day clean dataset, 55 of 105,120 on the 365-day realistic
dataset — in every case each one immediately after a `packet_loss_pct == 0.0` reading (one
dropped row per exact-zero reading).

At serving time it was a correctness bug. `CellHistoryStore.add_and_compute`
returned `feat_df.iloc[-1]` — but when the *newest* reading is the one dropped,
`iloc[-1]` is the **previous** reading. The API scored stale features as current,
raised nothing, and reported `feature_source: "live"`. Replaying real data
through the API's feature path reproduced it on all 9 of 9 such rows. Exact-zero
packet loss is the norm, not the exception, in real telemetry, so this would
have fired routinely in production.

**Fixed:** `add_and_compute` now selects the row for the reading it just added
(by timestamp) and raises `UndefinedFeatureError` if that row was dropped; the
API returns `422` with an explanatory message and increments
`undefined_feature_responses_total`. Guarded by `tests/test_live_features.py`
(confirmed to fail against the old `iloc[-1]` code) and an API-level test that
runs in CI.

**Not fixed — a real limitation you should know about:** rejecting the reading
is honest, but if a cell reports zero packet loss often, every reading after a
zero is unscoreable. The proper fix is a feature redesign (e.g. a rate-of-change
definition that's defined at zero, or NaN-native handling since XGBoost treats
`NaN` as missing) plus retraining and re-validating ONNX/Triton with NaN inputs.
That changes the model's feature semantics, so it wasn't done blind, without the
ability to run XGBoost or ONNX conversion here.

**Train/serve parity itself checks out:** replaying 450 randomly sampled real rows
through the API's feature path reproduced `features.csv` exactly (max absolute
difference 1.9e-11), i.e. the "no train/serve skew" claim in `live_features.py`
holds for every row that is scoreable.


## XGBoost measurements — supersede the stand-in headline numbers

Everything above this section was produced with a scikit-learn gradient-boosting stand-in because XGBoost could not
be installed where those studies ran. The CI evaluation (`.github/workflows/evaluation.yml`, results on the
`evaluation-results` branch) re-measured the project with the XGBoost model. The stand-in was **not** a faithful proxy
for the precision/recall trade-off:

| Recall target (validation) | Stand-in (previous default 0.965) | XGBoost, test: recall / precision |
|---|---|---|
| 0.93 | — | 0.930 / 0.934 |
| 0.945 | — | **0.950 / 0.804** (new default) |
| 0.95 | — | 0.962 / 0.622 |
| 0.965 | 0.970 / 0.795 | 0.973 / 0.428 |
| 0.98 | — | 0.979 / 0.335 |

The XGBoost model's precision-recall curve has a sharp knee near recall 0.94-0.95 that the stand-in's curve lacks, so a
target calibrated on the stand-in (0.965) landed far past the knee. The default is now **0.945**, read off the
curve with thresholds tuned on validation only. Across 10 independent datasets at that target (XGBoost):
recall median 94.2% (92.3-95.2%), precision median 87.4% (80.1-94.9%), every incident detected, median 0.19 false
alerts per cell per day. Seed 42 (the default dataset) is one of the harder draws (F1 0.871; the 10-seed median is 0.907).
Two of the ten draws land below the 0.93 promotion floor on test recall (seeds 4 and 7), so the gate would decline those.

## Serving optimisation

Measured by `.github/workflows/evaluation.yml`; the same load (4 clients, 300 requests each, Redis-backed) run against the
original code path and the optimised one on the same runner. Run-to-run spread on shared runners is about 2x.

| API path | req/s | CPU per request | p50 | p95 | peak RSS |
|---|---|---|---|---|---|
| original (pandas features, sklearn scaler) | 24 | 51.5 ms | 166 ms | 229 ms | 240 MiB |
| optimised (NumPy features and scaler) | 292 | 4.3 ms | 13 ms | 19 ms | 237 MiB |
| optimised + ONNX scorer (`SCORER=onnx`) | 330 | 3.5 ms | 12 ms | 17 ms | 251 MiB |

Cost of one request by stage: features 23.8 ms (pandas) vs 0.64 ms (NumPy); scaler 0.102 ms (sklearn) vs 0.003 ms (NumPy);
model 0.43 ms (XGBoost) vs 0.012 ms (ONNX Runtime). The NumPy feature path equals `engineer_features()` to 1e-9 on thousands of
random histories (`tests/test_live_features_fast.py`); flat windows, a zero previous value, duplicate timestamps and any other
undefined case fall back to the original code, so error behaviour is unchanged.

Smaller models, same data, threshold tuned on validation to the same 0.945 recall target:

| model | trees used | test recall | test precision | test F1 | model.json | predict one row |
|---|---|---|---|---|---|---|
| depth 6, up to 300 trees (shipped) | 189 | 0.950 | 0.804 | 0.871 | 1,127 KiB | 0.41 ms |
| depth 6, up to 150 trees | 143 | 0.955 | 0.742 | 0.835 | 796 KiB | 0.40 ms |
| depth 5, up to 300 trees | 189 | 0.950 | 0.773 | 0.853 | 734 KiB | 0.41 ms |
| depth 4, up to 300 trees | 189 | 0.954 | 0.787 | 0.863 | 447 KiB | 0.40 ms |
| depth 4, up to 150 trees | 143 | 0.952 | 0.776 | 0.855 | 308 KiB | 0.40 ms |
| depth 3, up to 300 trees | 285 | 0.960 | 0.722 | 0.824 | 365 KiB | 0.41 ms |

No smaller model beats the shipped one on F1, and single-row latency does not change (the cost is call overhead, not tree count),
so the model is unchanged. Quantisation was not applied: it targets neural-network weight matrices, and a tree ensemble has none.

Docker: the API image fell from 3,006 MiB (the full `requirements.txt`, rebuilt and measured in the same run) to 886 MiB (runtime dependencies only), then to 481 MiB by moving Evidently (plotly, statsmodels, pyarrow, ...) into a separate
`drift` image target used only by the drift CronJob. The drift job itself peaked at 999m CPU and 284 MiB and finished in 22 s.
