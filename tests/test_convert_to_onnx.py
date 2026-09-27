"""
test_convert_to_onnx.py
-------------------------
Tests models/convert_to_onnx.py's _select_probability_output() — the
part of the ONNX conversion pipeline that doesn't need onnxruntime,
onnxmltools, or xgboost installed at all, since it operates on plain
numpy arrays standing in for what session.run() would return.

The conversion + parity-check flow itself (convert(), validate_parity())
needs the real libraries and a real trained model — exercised only in
`.github/workflows/ci.yml`'s "Convert to ONNX + validate parity" step,
not in this offline suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from models.convert_to_onnx import _select_probability_output


def test_finds_probability_output_among_multiple_outputs():
    label_output = np.array([0, 1, 0])
    prob_output = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])

    result = _select_probability_output([label_output, prob_output], n_rows=3)
    assert np.array_equal(result, prob_output)


def test_order_independent():
    label_output = np.array([0, 1, 0])
    prob_output = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])

    result = _select_probability_output([prob_output, label_output], n_rows=3)
    assert np.array_equal(result, prob_output)


def test_single_probability_output_with_no_label_output():
    prob_output = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = _select_probability_output([prob_output], n_rows=2)
    assert np.array_equal(result, prob_output)


def test_raises_clear_error_when_no_matching_output_shape():
    label_output = np.array([0, 1, 0])
    with pytest.raises(RuntimeError, match=r"Could not find a \[N, 2\] probability output"):
        _select_probability_output([label_output], n_rows=3)


def test_does_not_match_wrong_row_count():
    """A [5, 2] array shouldn't be mistaken for the probability output
    when n_rows=3 — row count must match, not just ndim/column count."""
    wrong_row_count = np.zeros((5, 2))
    with pytest.raises(RuntimeError):
        _select_probability_output([wrong_row_count], n_rows=3)


def test_does_not_match_wrong_column_count():
    """A [3, 3] array (e.g. a multi-class problem, or some other
    per-row vector) shouldn't be mistaken for binary-classifier
    probabilities."""
    three_classes = np.zeros((3, 3))
    with pytest.raises(RuntimeError):
        _select_probability_output([three_classes], n_rows=3)
