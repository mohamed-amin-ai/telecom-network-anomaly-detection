"""
test_query_triton.py
----------------------
Tests scripts/query_triton.py's pure KServe v2 request/response
functions — no network, no Triton needed.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.query_triton import build_infer_request, parse_infer_response


def test_build_infer_request_shape():
    result = build_infer_request([1.0, 2.0, 3.0], input_name="input")
    assert result == {
        "inputs": [
            {"name": "input", "shape": [1, 3], "datatype": "FP32", "data": [1.0, 2.0, 3.0]}
        ]
    }


def test_build_infer_request_coerces_to_float():
    """Feature values arriving as numpy scalars or ints should come out
    as plain Python floats, so json.dumps (via requests) doesn't choke
    on a non-JSON-serializable numpy type."""
    result = build_infer_request([1, 2, 3])
    assert all(isinstance(x, float) for x in result["inputs"][0]["data"])


def test_build_infer_request_custom_input_name():
    result = build_infer_request([1.0], input_name="dense_input")
    assert result["inputs"][0]["name"] == "dense_input"


def test_parse_infer_response_extracts_named_outputs():
    response = {
        "model_name": "telecom_anomaly_xgb",
        "outputs": [
            {"name": "probabilities", "shape": [1, 2], "datatype": "FP32", "data": [0.9, 0.1]},
            {"name": "label", "shape": [1], "datatype": "INT64", "data": [0]},
        ],
    }
    parsed = parse_infer_response(response)
    assert parsed == {"probabilities": [0.9, 0.1], "label": [0]}


def test_parse_infer_response_handles_missing_outputs_key():
    assert parse_infer_response({}) == {}


def test_parse_infer_response_handles_empty_outputs_list():
    assert parse_infer_response({"outputs": []}) == {}
