"""
scorer.py
---------
The scoring step of a request: scale the 68-feature vector, return the anomaly probability.

Two things are optimised here, both without changing any output:

  * Scaling with plain NumPy. sklearn's StandardScaler.transform() spends most of its time on input
    validation for a single row; (x - mean_) / scale_ is the same arithmetic.
  * An optional ONNX Runtime backend (SCORER=onnx). It loads models/model.onnx (written by
    `make onnx-convert`) in a single-threaded session, which is the right setting when the server
    already runs one request per worker. The default stays the native XGBoost model, because the ONNX
    file and onnxruntime are not part of the slim serving image.
"""

import os
from pathlib import Path

import numpy as np


def make_scaler_fn(scaler):
    """Return a function equal to scaler.transform for a StandardScaler, else scaler.transform itself."""
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if type(scaler).__name__ != "StandardScaler" or mean is None or scale is None \
            or not getattr(scaler, "with_mean", False) or not getattr(scaler, "with_std", False):
        return scaler.transform
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return lambda X: (np.asarray(X, dtype=float) - mean) / scale


class Scorer:
    def __init__(self, model, scaler, model_dir: Path, backend: str | None = None):
        self.backend = (backend or os.environ.get("SCORER", "xgboost")).lower()
        # SCORER_NUMPY_SCALER=0 restores sklearn's transform (used to measure the original path)
        self._scale = make_scaler_fn(scaler) if os.environ.get("SCORER_NUMPY_SCALER", "1") != "0" else scaler.transform
        self._model = model
        self._session = None
        if self.backend == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), opts,
                                                 providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0].name
        elif self.backend != "xgboost":
            raise ValueError(f"SCORER must be 'xgboost' or 'onnx', got {self.backend!r}")

    def probability(self, X: np.ndarray) -> float:
        """Anomaly probability for one raw (unscaled) feature row of shape (1, n_features)."""
        Xs = self._scale(X)
        if self._session is not None:
            out = self._session.run(None, {self._input: Xs.astype(np.float32)})
            probs = out[-1] if np.ndim(out[-1]) == 2 else out[0]
            return float(probs[0, 1])
        return float(self._model.predict_proba(Xs)[0, 1])
