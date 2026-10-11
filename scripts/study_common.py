"""
study_common.py
-----------------
Shared by scripts/realism_study.py and scripts/recall_target_study.py.

One call builds a dataset under a given generator profile, trains the
scikit-learn stand-in model once (xgboost isn't installable in the build
sandbox), and scores it at several validation recall targets — at the
ROW level (recall/precision/F1) and at the INCIDENT level
(models/alerting.py: detection rate, false alerts per cell-day, delay).

Protocol, identical to models/train.py: chronological train/val/test,
scaler fit on train only, threshold tuned on validation only, test touched
once. Absolute numbers from real XGBoost will differ.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
import data.generate_data as gen
from data.features import engineer_features, get_feature_columns
from models.alerting import grouped_event_metrics, runs
from models.common import load_params, three_way_time_split, tune_threshold

FACTORS = ("nonstationarity", "severity_and_ramp", "label_noise", "benign_surges")

# The "realistic" level is params.yaml's data.realism block resolved AS the
# realistic profile — independent of which profile is currently configured
# (gen.REALISM may have been rewritten by the "calibrated" alias).
_DATA_PARAMS = load_params()["data"]
DEFAULT_REALISM = gen.resolve_profile({**_DATA_PARAMS, "profile": "realistic"})[1]

# Named difficulty levels. "clean" is the legacy generator (an upper bound);
# "realistic" uses params.yaml's realism block; "harsh" is the ORIGINAL
# pre-registered realism setting, kept so the sensitivity of every result to
# those judgment calls stays visible. Which of these is the CONFIGURED
# benchmark is decided by params.yaml's data.profile (see configured_level).
LEVELS = {
    "clean":     {"profile": "legacy",    "realism": {}},
    "realistic": {"profile": "realistic", "realism": {}},
    "harsh":     {"profile": "realistic", "realism": {"severity_min": 0.15, "benign_to_fault_ratio": 1.0}},
    # Only faint/gradual faults, one visibility dial (severity_min). The dial's
    # value is NOT fixed here: scripts/calibrate_to_original.py searches it, and
    # params.yaml -> data.calibrated stores the result.
    "calibrated": {"profile": "realistic",
                   "realism": {"nonstationarity": False, "label_noise": False, "benign_surges": False}},
}
# Levels whose operating point comes from the F2 rule in recall_target_study.py.
# "calibrated" gets its operating point from calibrate_to_original.py instead.
F2_RULE_LEVELS = ("clean", "realistic", "harsh")

_PROFILE_TO_LEVEL = {"legacy": "clean", "realistic": "realistic", "calibrated": "calibrated"}


def configured_level() -> str:
    """The level params.yaml's data.profile currently selects."""
    return _PROFILE_TO_LEVEL[gen.CONFIGURED_PROFILE]


def apply_level(level: str, days=None, seed=None, realism_overrides=None):
    """Configure the generator for a named level (module-level state).
    `realism_overrides` lets the calibration script sweep the visibility dial."""
    spec = LEVELS[level]
    gen.PROFILE = spec["profile"]
    gen.REALISM.clear()
    gen.REALISM.update(DEFAULT_REALISM)
    for f in FACTORS:                      # realistic levels start with all four factors on
        gen.REALISM[f] = True
    gen.REALISM.update(spec["realism"])
    if level == "calibrated" and "calibrated" in _DATA_PARAMS:
        gen.REALISM.update(_DATA_PARAMS["calibrated"])
    if realism_overrides:
        gen.REALISM.update(realism_overrides)
    if days is not None:
        gen.PERIODS = days * gen.INTERVALS_PER_DAY
    if seed is not None:
        gen.SEED = seed


def set_profile(profile: str, factors_on=(), days=None, seed=None):
    """Point the generator at a profile/factor set (module-level state, same
    mechanism the tests use)."""
    gen.PROFILE = profile
    for f in FACTORS:
        gen.REALISM[f] = f in factors_on
    if days is not None:
        gen.PERIODS = days * gen.INTERVALS_PER_DAY
    if seed is not None:
        gen.SEED = seed


def fit_stand_in(Xtr, ytr, Xva, yva, model_params: dict):
    """scikit-learn stand-in that follows models/train.py's recipe: positives
    upweighted by neg/pos (scale_pos_weight), boosting stopped by VALIDATION
    PR-AUC with `early_stopping_rounds` patience (train.py: eval_metric='aucpr',
    early_stopping_rounds=30, eval_set=validation).

    The first version of these studies fixed 300 rounds with no early stopping.
    That was harmless on the clean benchmark but badly overfits once label
    noise and faint faults are present — the model memorizes positives that
    are indistinguishable from normal rows — which made the realistic
    profile look far harder than it is for the real training recipe.

    Returns (model, best_round). Score with predict_best().
    """
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    model = HistGradientBoostingClassifier(
        max_depth=model_params["max_depth"], learning_rate=model_params["learning_rate"],
        max_iter=model_params["n_estimators"], early_stopping=False,
        random_state=model_params["random_state"],
    )
    model.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, spw, 1.0))
    best_score, best_round = -1.0, 0
    for i, proba in enumerate(model.staged_predict_proba(Xva)):
        score = average_precision_score(yva, proba[:, 1])
        if score > best_score:
            best_score, best_round = score, i
        elif i - best_round >= model_params["early_stopping_rounds"]:
            break
    return model, best_round


def predict_best(model, X, best_round) -> np.ndarray:
    for i, proba in enumerate(model.staged_predict_proba(X)):
        if i == best_round:
            return proba[:, 1]
    return model.predict_proba(X)[:, 1]


def count_label_events(labels, cell_ids) -> int:
    return sum(len(runs(labels[cell_ids == c])) for c in np.unique(cell_ids))


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    denom = beta**2 * precision + recall
    return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_current_dataset(params: dict, targets, backend: str = "stand_in") -> dict:
    """Build the dataset the generator is CURRENTLY configured for, train the
    baseline once, and score every recall target on the held-out test set."""
    df = engineer_features(gen.build_dataset(), verbose=False)
    cols = get_feature_columns(df)
    tr, va, te = three_way_time_split(df, params["split"]["train_ratio"], params["split"]["val_fraction"])

    sc = StandardScaler()
    Xtr, Xva, Xte = sc.fit_transform(tr[cols].values), sc.transform(va[cols].values), sc.transform(te[cols].values)
    ytr, yva, yte = (d["anomaly"].values for d in (tr, va, te))

    if backend == "xgboost":
        # Exactly models/train.py's recipe (needs the real library; used by scripts/eval_tools.py spread in CI).
        import xgboost as xgb
        mp = params["model"]
        model = xgb.XGBClassifier(
            n_estimators=mp["n_estimators"], max_depth=mp["max_depth"], learning_rate=mp["learning_rate"],
            subsample=mp["subsample"], colsample_bytree=mp["colsample_bytree"],
            scale_pos_weight=(ytr == 0).sum() / max(1, (ytr == 1).sum()), eval_metric="aucpr",
            early_stopping_rounds=mp["early_stopping_rounds"], random_state=mp["random_state"], n_jobs=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best_round = int(model.best_iteration)
        pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    else:
        model, best_round = fit_stand_in(Xtr, ytr, Xva, yva, params["model"])
        pv, pt = predict_best(model, Xva, best_round), predict_best(model, Xte, best_round)

    cells_te = te["cell_id"].values
    minutes_per_interval = 24 * 60 / gen.INTERVALS_PER_DAY
    out = {
        "val_events": count_label_events(yva, va["cell_id"].values),
        "test_events": count_label_events(yte, cells_te),
        "test_positive_rows": int(yte.sum()),
        "test_anomaly_rate": round(float(yte.mean()), 4),
        "test_pr_auc": round(float(average_precision_score(yte, pt)), 4),
        "best_round": int(best_round),
        "by_target": {},
    }
    for t in targets:
        thr, *_ = tune_threshold(yva, pv, t)
        pred = pt >= thr
        tn, fp, fn, tp = confusion_matrix(yte, pred.astype(int), labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        ev = grouped_event_metrics(yte, pred, cells_te, intervals_per_day=gen.INTERVALS_PER_DAY)
        out["by_target"][str(t)] = {
            "threshold": round(float(thr), 4),
            "recall": round(rec, 4), "precision": round(prec, 4),
            "f1": round(2 * rec * prec / (rec + prec), 4) if rec + prec else 0.0,
            "f2": round(f_beta(prec, rec, 2.0), 4),
            "event_detection_rate": round(float(ev["detection_rate"]), 4),
            "alert_precision": round(float(ev["alert_precision"]), 4) if ev["n_alerts"] else None,
            "false_alerts_per_cell_day": round(float(ev["false_alerts_per_cell_day"]), 4),
            "flagged_fraction": round(float(pred.mean()), 4),
            "alerts_per_cell_day": round(ev["n_alerts"] / (ev["n_intervals"] / gen.INTERVALS_PER_DAY), 4),
            "median_delay_minutes": None if np.isnan(ev["median_delay_intervals"])
                                    else round(float(ev["median_delay_intervals"]) * minutes_per_interval, 1),
        }
    return out


def spread(values) -> dict:
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": round(float(v.min()), 3), "median": round(float(np.median(v)), 3), "max": round(float(v.max()), 3)}
