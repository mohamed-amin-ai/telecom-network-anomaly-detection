"""
test_inference_log.py
----------------------
Unit tests for api/inference_log.py. Uses a temp directory (monkeypatched
LOG_DIR/LOG_PATH) so tests never touch the real logs/ directory.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
import api.inference_log as inflog


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    """Point the module's LOG_DIR/LOG_PATH at a temp dir for this test only."""
    log_dir = tmp_path / "logs"
    log_path = log_dir / "inference_log.csv"
    monkeypatch.setattr(inflog, "LOG_DIR", log_dir)
    monkeypatch.setattr(inflog, "LOG_PATH", log_path)
    return log_path


FEATURE_COLS = ["latency_ms", "throughput_mbps", "packet_loss_pct"]


def test_read_log_returns_none_when_no_file_yet(isolated_log):
    assert inflog.read_log() is None


def test_log_inference_creates_file_with_header(isolated_log):
    inflog.log_inference(
        cell_id="CELL_001",
        feature_source="live",
        anomaly_score=0.12,
        features={"latency_ms": 45.0, "throughput_mbps": 90.0, "packet_loss_pct": 0.5},
        feature_cols=FEATURE_COLS,
    )
    assert isolated_log.exists()
    df = inflog.read_log(feature_cols=FEATURE_COLS)
    assert len(df) == 1
    assert df.iloc[0]["cell_id"] == "CELL_001"
    assert df.iloc[0]["latency_ms"] == 45.0


def test_log_inference_appends_multiple_rows(isolated_log):
    for i in range(5):
        inflog.log_inference(
            cell_id=f"CELL_{i}",
            feature_source="precomputed",
            anomaly_score=0.1 * i,
            features={"latency_ms": float(i), "throughput_mbps": float(i), "packet_loss_pct": float(i)},
            feature_cols=FEATURE_COLS,
        )
    df = inflog.read_log(feature_cols=FEATURE_COLS)
    assert len(df) == 5
    assert list(df["cell_id"]) == [f"CELL_{i}" for i in range(5)]


def test_log_inference_missing_feature_value_becomes_null_not_crash(isolated_log):
    """features dict missing a column the caller asked to log should not
    raise — just log null for that column (defensive, since a future
    caller might log with a slightly different feature set mid-rollout)."""
    inflog.log_inference(
        cell_id="CELL_PARTIAL",
        feature_source="live",
        anomaly_score=0.5,
        features={"latency_ms": 10.0},  # missing throughput_mbps, packet_loss_pct
        feature_cols=FEATURE_COLS,
    )
    df = inflog.read_log(feature_cols=FEATURE_COLS)
    assert len(df) == 1
    assert df.iloc[0]["latency_ms"] == 10.0
    import pandas as pd
    assert pd.isna(df.iloc[0]["throughput_mbps"])


def test_read_log_raises_clear_error_on_schema_mismatch(isolated_log):
    inflog.log_inference(
        cell_id="CELL_001",
        feature_source="live",
        anomaly_score=0.1,
        features={"latency_ms": 1.0, "throughput_mbps": 2.0, "packet_loss_pct": 3.0},
        feature_cols=FEATURE_COLS,
    )
    with pytest.raises(ValueError, match="missing expected feature columns"):
        inflog.read_log(feature_cols=[*FEATURE_COLS, "sinr_db_that_never_existed"])


def test_log_inference_never_raises_even_if_log_dir_is_unwritable(isolated_log, monkeypatch):
    """Logging failures must be swallowed — observability must never
    break the thing it's observing."""
    def boom(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(inflog.Path, "mkdir", boom)
    # Should not raise, despite mkdir being broken.
    inflog.log_inference(
        cell_id="CELL_001",
        feature_source="live",
        anomaly_score=0.1,
        features={"latency_ms": 1.0},
        feature_cols=FEATURE_COLS,
    )
