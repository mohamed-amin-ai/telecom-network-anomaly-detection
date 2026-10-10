"""
convert_to_onnx.py
--------------------
Converts models/model.json (native XGBoost) to models/model.onnx, then
validates it actually agrees with the native model on real held-out data
before accepting it as a usable artifact.

═══════════════════════════════════════════════════════════════════════
CAVEAT — same pattern as Evidently (Phase 4) and Airflow (Phase 5)
═══════════════════════════════════════════════════════════════════════
`onnxmltools`/`onnxruntime` aren't installable in the sandbox this was
built in (no network), so the exact conversion API below is unverified
against the real libraries. Confidence notes:
  - `onnxmltools.convert.convert_xgboost` with a `FloatTensorType` input
    and `zipmap=False` in `options` is long-standing, well-documented
    onnxmltools usage for XGBoost specifically — moderate-high confidence.
  - The exact output tensor name(s) onnxmltools produces (referenced
    below via `session.get_outputs()` rather than a hardcoded string,
    specifically to avoid needing to guess right) is the part I'm least
    sure of.
This script does NOT trust the conversion just because it ran without
raising — it re-scores real held-out test rows through both the native
model and the ONNX model and requires the probabilities to agree within
a tight tolerance AND every thresholded decision to match exactly. If
they don't, it raises rather than writing models/model.onnx at all — see
`.github/workflows/ci.yml`'s "Convert to ONNX + validate parity" step,
the first real execution of this code against the actual libraries.
═══════════════════════════════════════════════════════════════════════

Usage:
  python models/convert_to_onnx.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import load_params, load_data, time_split

MODEL_DIR = Path(__file__).parent
PROB_ATOL = 1e-4  # max allowed |native_prob - onnx_prob| per row


def load_native_artifacts():
    # Imported here, not at module level, so _select_probability_output
    # below stays importable/testable without xgboost/joblib installed
    # at all (verified in this sandbox, neither is).
    import joblib
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler


def _strip_zipmap(onnx_model) -> None:
    from onnx import TensorProto, helper

    graph = onnx_model.graph
    for node in [n for n in graph.node if n.op_type == "ZipMap"]:
        prob, zipped = node.input[0], node.output[0]
        graph.node.remove(node)
        for out in [o for o in graph.output if o.name == zipped]:
            graph.output.remove(out)
        graph.output.append(helper.make_tensor_value_info(prob, TensorProto.FLOAT, [None, 2]))


def convert(model, n_features: int):
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_types)
    # onnxmltools' convert_xgboost() has no ZipMap option (unlike skl2onnx), and depending on
    # the version a classifier may end in a ZipMap node, i.e. a sequence-of-maps output that
    # Triton and non-Python clients cannot consume. Strip it so the output is a plain float
    # tensor [N, 2], which is what triton_model_repository/.../config.pbtxt assumes.
    _strip_zipmap(onnx_model)
    return onnx_model


def get_real_holdout_rows(n_rows: int = 200):
    """Real held-out data (the same time-based test split train.py itself
    uses), not synthetic — parity should be checked against data the
    model wasn't trained on."""
    params = load_params()
    df = load_data()
    feature_cols = get_feature_columns(df)
    _, test_df = time_split(df, params["split"]["train_ratio"])
    sample = test_df[feature_cols].iloc[:n_rows]
    return sample.values, feature_cols


def _select_probability_output(outputs, n_rows: int) -> np.ndarray:
    """Find the output that looks like [N, 2] class probabilities among
    ONNX Runtime's session.run() results — don't assume it's outputs[0]
    or any particular name, since that's exactly the part of
    onnxmltools' behavior this script is least sure of (see module
    docstring). Pulled out of validate_parity so it's testable against
    fake arrays without needing onnxruntime installed at all."""
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 2 and arr.shape[0] == n_rows and arr.shape[1] == 2:
            return arr
    shapes = [np.asarray(o).shape for o in outputs]
    raise RuntimeError(
        f"Could not find a [N, 2] probability output among ONNX outputs "
        f"(got shapes: {shapes}). onnxmltools' output structure may have "
        f"changed — inspect `outputs` directly and fix this function's "
        f"output-selection logic, don't guess an index."
    )


def validate_parity(model, scaler, onnx_model, X_raw: np.ndarray) -> dict:
    import onnxruntime as ort

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    native_probs = model.predict_proba(X_scaled)[:, 1]

    session = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name
    output_meta = [(o.name, o.shape) for o in session.get_outputs()]
    outputs = session.run(None, {input_name: X_scaled})

    prob_output = _select_probability_output(outputs, n_rows=len(X_scaled))
    onnx_probs = prob_output[:, 1]

    abs_diff = np.abs(native_probs - onnx_probs)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    native_decision = native_probs >= 0.5
    onnx_decision = onnx_probs >= 0.5
    decisions_match = bool(np.array_equal(native_decision, onnx_decision))

    return {
        "n_rows": len(X_scaled),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "decisions_match": decisions_match,
        "within_tolerance": max_diff <= PROB_ATOL,
        "input_name": input_name,
        "output_meta": output_meta,
    }


def main():
    model, scaler = load_native_artifacts()
    X_raw, feature_cols = get_real_holdout_rows()
    print(f"Loaded {len(X_raw)} real held-out rows across {len(feature_cols)} features "
          f"for parity checking.")

    print("Converting to ONNX...")
    onnx_model = convert(model, n_features=len(feature_cols))

    print("Validating parity against the native model on real data...")
    result = validate_parity(model, scaler, onnx_model, X_raw)
    print(f"  rows checked      : {result['n_rows']}")
    print(f"  max |prob diff|   : {result['max_abs_diff']:.6f}  (tolerance: {PROB_ATOL})")
    print(f"  mean |prob diff|  : {result['mean_abs_diff']:.6f}")
    print(f"  decisions match   : {result['decisions_match']}")
    print(f"  ONNX input name   : {result['input_name']}")
    print(f"  ONNX output(s)    : {result['output_meta']}")
    print("  → if triton_model_repository/telecom_anomaly_xgb/config.pbtxt's "
          "input/output names don't match the above, update it to match "
          "(see that file's own comments).")

    if not result["within_tolerance"] or not result["decisions_match"]:
        raise RuntimeError(
            f"ONNX conversion FAILED parity validation "
            f"(max_abs_diff={result['max_abs_diff']:.6f} vs tolerance={PROB_ATOL}, "
            f"decisions_match={result['decisions_match']}). NOT writing model.onnx — "
            f"shipping a numerically-diverged model would be worse than shipping "
            f"nothing. See this file's module docstring for what to check."
        )

    out_path = MODEL_DIR / "model.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nParity check passed. Saved → {out_path}")


if __name__ == "__main__":
    main()
