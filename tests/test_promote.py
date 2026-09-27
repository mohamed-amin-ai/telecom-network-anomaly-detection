"""
test_promote.py
----------------
Tests models/promote.py's should_promote() — the pure decision logic,
with no MLflow I/O. promote_latest_model() itself (the MLflow-calling
half) isn't tested here — it needs a live tracking server/registry, so
it's exercised only in the Airflow smoke path documented in README's
Phase 5 section, not in this offline suite.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.promote import should_promote


def test_promotes_when_recall_above_floor():
    assert should_promote(0.95, min_recall=0.93) is True


def test_does_not_promote_when_recall_below_floor():
    assert should_promote(0.90, min_recall=0.93) is False


def test_promotes_at_exact_floor_boundary():
    """>= , not > — a model that exactly clears the floor should promote."""
    assert should_promote(0.93, min_recall=0.93) is True


def test_does_not_promote_when_recall_is_none():
    """A run with no logged 'recall' metric (e.g. training failed partway,
    or logged under a different key) must never promote blind."""
    assert should_promote(None, min_recall=0.93) is False


def test_does_not_promote_when_recall_is_zero():
    """0.0 is falsy in Python — make sure the check is `is None`, not a
    truthiness check, or a genuinely terrible model (recall=0) that
    happens to be compared against min_recall=0 could wrongly promote,
    and worse, a recall of exactly 0 must not be silently treated the
    same as 'missing'."""
    assert should_promote(0.0, min_recall=0.0) is True  # 0 >= 0, legitimately
    assert should_promote(0.0, min_recall=0.5) is False
