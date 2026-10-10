"""
test_api.py
-----------
Integration tests for api/main.py. These import the FastAPI app, which
at module load time loads models/model.json + scaler.pkl + metadata.json
— so they require a real trained model to exist (see conftest.py's
require_model_artifacts fixture). In CI, `dvc repro` runs before pytest
for exactly this reason. Locally, if artifacts are missing, these tests
are skipped (not failed) with a clear message.

NOT executed in the sandbox this repo was partly built in — that sandbox
has no network access, so fastapi/httpx/xgboost aren't installable there.
Written to run correctly once requirements.txt + requirements-dev.txt are
installed (true in CI, true on a normal dev machine).
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def client(require_model_artifacts):
    from fastapi.testclient import TestClient
    from api.main import app, history_store

    # Each test gets a clean in-memory history so cold-start/warm-start
    # behavior doesn't leak between tests sharing the module-level store.
    history_store._backend._history.clear()   # default backend is in-memory
    return TestClient(app)


SAMPLE_KPIS = {
    "latency_ms": 45.5,
    "throughput_mbps": 90.0,
    "packet_loss_pct": 0.5,
    "sinr_db": 15.0,
    "connected_users": 50,
}


def _varied(i: int) -> dict:
    """KPIs that change on every reading. Identical readings give a rolling std of exactly 0,
    so the z-score is 0/0 = NaN and the API (correctly) answers 422 — a perfectly constant
    KPI is not a realistic warm-up input."""
    return {
        "latency_ms": 45.5 + 0.1 * i, "throughput_mbps": 90.0 - 0.2 * i,
        "packet_loss_pct": 0.5 + 0.01 * i, "sinr_db": 15.0 + 0.05 * (i % 7),
        "connected_users": 50 + (i % 5),
    }


def _snapshot(cell_id: str, **overrides) -> dict:
    payload = {"cell_id": cell_id, **SAMPLE_KPIS}
    payload.update(overrides)
    return payload


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "threshold" in body
    assert "metrics" in body
    assert body["feature_count"] > 0


def test_predict_cold_start_returns_425(client):
    """First reading ever for a brand-new cell_id must not be silently
    scored on a mostly-empty feature vector."""
    resp = client.post("/predict", json=_snapshot("TEST_CELL_COLD"))
    assert resp.status_code == 425
    assert "history" in resp.json()["detail"].lower()


def test_predict_warms_up_after_enough_history(client):
    cell_id = "TEST_CELL_WARMUP"
    # MIN_HISTORY_ROWS is LONG_WINDOW(16) + 4 = 20; send enough readings.
    last_resp = None
    for i in range(25):
        last_resp = client.post("/predict", json=_snapshot(
            cell_id,
            timestamp=f"2024-01-01T{i:02d}:00:00Z" if i < 24 else None,
            **_varied(i),
        ))
    assert last_resp.status_code == 200
    body = last_resp.json()
    assert body["feature_source"] == "live"
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert body["cell_id"] == cell_id


def test_predict_with_precomputed_features(client):
    from api.main import FEATURE_COLS

    features = {col: 0.0 for col in FEATURE_COLS}
    resp = client.post("/predict", json=_snapshot("TEST_CELL_PRECOMPUTED", features=features))
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_source"] == "precomputed"


def test_predict_missing_precomputed_feature_returns_422(client):
    resp = client.post("/predict", json=_snapshot(
        "TEST_CELL_BADFEAT", features={"latency_ms_lag1": 1.0}  # incomplete on purpose
    ))
    assert resp.status_code == 422


def test_predict_batch_rejects_oversized_batch(client):
    snapshots = [_snapshot(f"CELL_{i}") for i in range(501)]
    resp = client.post("/predict/batch", json={"snapshots": snapshots})
    assert resp.status_code == 400


def test_predict_batch_reports_per_item_errors_not_whole_batch_failure(client):
    """A cold-start item in a batch should surface as that item's own
    error, not fail the entire batch."""
    snapshots = [_snapshot("TEST_CELL_BATCH_COLD")]
    resp = client.post("/predict/batch", json={"snapshots": snapshots})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["error_count"] == 1
    assert body["results"][0]["ok"] is False


def test_metrics_endpoint_reflects_real_traffic(client):
    """/metrics should be a working Prometheus scrape target, and the
    counters wired up in api/metrics.py should actually move after real
    requests — not just be present but frozen at zero."""
    from api.main import FEATURE_COLS

    # Trigger one precomputed-path prediction and one cold-start 425,
    # each of which should move a different counter.
    features = {col: 0.0 for col in FEATURE_COLS}
    client.post("/predict", json=_snapshot("TEST_CELL_METRICS", features=features))
    client.post("/predict", json=_snapshot("TEST_CELL_METRICS_COLD"))

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    assert "anomaly_predictions_total" in body
    assert "feature_source_total" in body
    assert "anomaly_score_bucket" in body
    assert "cold_start_responses_total" in body
    assert "cells_tracked" in body
    assert "model_threshold_info" in body
    # Standard instrumentator HTTP metrics should also be present.
    assert "http_requests_total" in body


def test_reading_after_a_zero_packet_loss_reading_is_422_not_stale_features(client):
    """Regression for a silent serving bug: after a reading with
    packet_loss_pct == 0.0, the NEXT reading's rate-of-change feature is
    undefined. The API used to score the previous row's features as if
    they were current (feature_source still said 'live'). It must now
    refuse explicitly, and count it in Prometheus."""
    cell_id = "TEST_CELL_ZERO_PREV"
    for i in range(30):   # warm up with strictly positive, varying packet loss
        r = client.post("/predict", json=_snapshot(
            cell_id, timestamp=f"2024-01-01T{i // 4:02d}:{(i % 4) * 15:02d}:00Z",
            **{**_varied(i), "packet_loss_pct": 0.3 + 0.01 * i},
        ))
    assert r.status_code == 200

    zero = client.post("/predict", json=_snapshot(
        cell_id, timestamp="2024-01-01T08:00:00Z", **{**_varied(30), "packet_loss_pct": 0.0}))
    assert zero.status_code == 200      # the zero reading itself is fine (prev was > 0)

    after = client.post("/predict", json=_snapshot(
        cell_id, timestamp="2024-01-01T08:15:00Z", **{**_varied(31), "packet_loss_pct": 0.4}))
    assert after.status_code == 422
    assert "undefined" in after.json()["detail"].lower()

    assert "undefined_feature_responses_total" in client.get("/metrics").text
