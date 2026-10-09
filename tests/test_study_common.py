"""
test_study_common.py
---------------------
Tests the helpers behind scripts/recall_target_study.py and
scripts/realism_study.py: the F-beta used to pick the operating point, the
pre-stated selection rule, the early-stopping stand-in, and the difficulty
levels. (The studies themselves run for minutes and are not unit tests.)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
import data.generate_data as gen
from scripts.recall_target_study import choose_target
from scripts.study_common import LEVELS, apply_level, f_beta, fit_stand_in, predict_best, spread


def test_f_beta_basics():
    assert f_beta(1.0, 1.0, 2.0) == 1.0
    assert f_beta(0.0, 0.0, 2.0) == 0.0
    assert math.isclose(f_beta(0.5, 0.5, 1.0), 0.5)


def test_f2_weights_recall_more_than_precision():
    """The stated preference: a missed fault costs more than a false alarm."""
    assert f_beta(precision=0.5, recall=1.0, beta=2.0) > f_beta(precision=1.0, recall=0.5, beta=2.0)


def test_spread_ignores_none_and_nan():
    s = spread([1.0, None, float("nan"), 3.0, 2.0])
    assert s == {"min": 1.0, "median": 2.0, "max": 3.0}
    assert spread([None, float("nan")]) == {"min": None, "median": None, "max": None}


def _summary(f2_by_target: dict) -> dict:
    return {str(t): {"f2": {"median": v}} for t, v in f2_by_target.items()}


def test_choose_target_picks_max_median_f2():
    assert choose_target(_summary({0.3: 0.2, 0.5: 0.6, 0.9: 0.4})) == 0.5


def test_choose_target_breaks_ties_toward_the_smaller_target():
    assert choose_target(_summary({0.3: 0.5, 0.5: 0.5, 0.9: 0.1})) == 0.3


def test_early_stopping_standin_stops_before_the_round_cap_on_noise():
    """Pure-noise labels: validation PR-AUC cannot keep improving, so the
    chosen round must be well short of the cap — i.e. the stand-in really
    follows validation-based early stopping instead of fitting all rounds."""
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(size=(4000, 8)), rng.normal(size=(1500, 8))
    ytr, yva = (rng.random(4000) < 0.05).astype(int), (rng.random(1500) < 0.05).astype(int)
    params = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200, "early_stopping_rounds": 10, "random_state": 0}
    model, best = fit_stand_in(Xtr, ytr, Xva, yva, params)
    assert 0 <= best < 150, f"best round {best} suggests no early stopping happened"
    p = predict_best(model, Xva, best)
    assert p.shape == (1500,) and ((p >= 0) & (p <= 1)).all()


def test_predict_best_uses_the_requested_round_not_the_final_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5)); y = (X[:, 0] + 0.3 * rng.normal(size=2000) > 1).astype(int)
    params = {"max_depth": 3, "learning_rate": 0.2, "n_estimators": 60, "early_stopping_rounds": 5, "random_state": 0}
    model, _ = fit_stand_in(X[:1500], y[:1500], X[1500:], y[1500:], params)
    early, final = predict_best(model, X[1500:], 3), model.predict_proba(X[1500:])[:, 1]
    assert not np.allclose(early, final), "round 3 and the final round should score differently"


def test_levels_configure_the_generator_and_are_restorable():
    saved = (gen.PROFILE, dict(gen.REALISM), gen.PERIODS, gen.SEED)
    try:
        apply_level("clean")
        assert gen.PROFILE == "legacy"

        apply_level("harsh", days=30, seed=7)
        assert gen.PROFILE == "realistic" and gen.SEED == 7 and gen.PERIODS == 30 * gen.INTERVALS_PER_DAY
        assert gen.REALISM["severity_min"] == 0.15 and gen.REALISM["benign_to_fault_ratio"] == 1.0
        assert all(gen.REALISM[f] for f in ("nonstationarity", "severity_and_ramp", "label_noise", "benign_surges"))

        apply_level("realistic")
        assert gen.REALISM["severity_min"] == 0.4          # params.yaml's realism block
        assert set(LEVELS) == {"clean", "realistic", "harsh", "calibrated"}
        apply_level("calibrated", realism_overrides={"severity_min": 0.55})
        assert gen.REALISM["severity_min"] == 0.55
        assert gen.REALISM["severity_and_ramp"] is True
        assert not any(gen.REALISM[f] for f in ("nonstationarity", "label_noise", "benign_surges"))
    finally:
        gen.PROFILE, gen.PERIODS, gen.SEED = saved[0], saved[2], saved[3]
        gen.REALISM.clear(); gen.REALISM.update(saved[1])
