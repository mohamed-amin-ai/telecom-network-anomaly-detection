"""api/scorer.py: NumPy scaling must equal StandardScaler.transform, and the scorer must equal the model."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from api.scorer import Scorer, make_scaler_fn


class _Model:
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.column_stack([1 - p, p])


def test_numpy_scaling_equals_sklearn():
    rng = np.random.default_rng(0)
    data = rng.normal(5, 3, size=(200, 68))
    data[:, 7] = 4.0                                   # constant column: sklearn sets scale_ to 1
    sc = StandardScaler().fit(data)
    fn = make_scaler_fn(sc)
    for _ in range(50):
        x = rng.normal(5, 3, size=(1, 68))
        np.testing.assert_allclose(fn(x), sc.transform(x), rtol=1e-12, atol=1e-12)


def test_non_standard_scaler_falls_back_to_transform():
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler().fit(np.random.default_rng(1).normal(size=(20, 3)))
    assert make_scaler_fn(sc) == sc.transform


def test_scorer_matches_model_probability():
    rng = np.random.default_rng(2)
    sc = StandardScaler().fit(rng.normal(size=(100, 68)))
    scorer = Scorer(_Model(), sc, Path("."), backend="xgboost")
    x = rng.normal(size=(1, 68))
    assert scorer.probability(x) == pytest.approx(float(_Model().predict_proba(sc.transform(x))[0, 1]), abs=1e-12)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        Scorer(_Model(), StandardScaler().fit(np.zeros((3, 2)) + [[0, 1], [1, 0], [2, 2]]), Path("."), backend="tensorrt")
