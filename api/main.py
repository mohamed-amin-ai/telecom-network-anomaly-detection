"""
main.py  —  FastAPI Anomaly Detection Service
----------------------------------------------
Provides real-time inference for telecom network KPI anomaly detection.

Endpoints:
  GET  /health          — liveness check
  GET  /model/info      — model metadata (threshold, metrics, features)
  POST /predict         — single snapshot prediction
  POST /predict/batch   — batch of snapshots (e.g. last hour of readings)

Usage:
  uvicorn api.main:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.live_features import (
    CellHistoryStore,
    HistoryStoreUnavailable,
    UndefinedFeatureError,
    make_raw_row,
    now_iso,
    MIN_HISTORY_ROWS,
)
from api.auth import require_api_key
from api.scorer import Scorer
from api.shared_store import build_history_backend
from api.metrics import (
    record_history_store_error,
    instrument_app,
    set_model_info,
    record_prediction,
    record_cold_start,
    record_undefined_features,
    set_cells_tracked,
)
from api.inference_log import log_inference, log_inferences

# ── Load model artifacts ───────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "models"

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    scorer = Scorer(model, scaler, MODEL_DIR)

    FEATURE_COLS = metadata["feature_cols"]
    THRESHOLD    = metadata["threshold"]

    set_model_info(THRESHOLD)

except FileNotFoundError:
    raise RuntimeError(
        "Model artifacts not found. Train the model first:\n"
        "  python models/train.py"
    )

# Per-cell rolling history used to compute real-time features when the
# caller sends raw KPIs instead of pre-computed features. See
# api/live_features.py for why this replaced the old zero-fill fallback.
history_store = CellHistoryStore(build_history_backend())

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom Network Anomaly Detection API",
    description=(
        "Real-time ML inference for detecting anomalies in telecom network KPIs. "
        "Powered by XGBoost trained on latency, throughput, packet loss, SINR, "
        "and connected users time-series data."
    ),
    version="1.0.0",
)

instrument_app(app)


# ── Schemas ────────────────────────────────────────────────────────────────

class KPISnapshot(BaseModel):
    cell_id: str = Field(..., example="CELL_001")

    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of this reading. Defaults to now().",
        example="2026-07-18T14:00:00Z",
    )

    latency_ms:       float = Field(..., ge=0,   example=145.5)
    throughput_mbps:  float = Field(..., ge=0,   example=12.3)
    packet_loss_pct:  float = Field(..., ge=0, le=100, example=8.7)
    sinr_db:          float = Field(...,         example=4.2)
    connected_users:  int   = Field(..., ge=0,   example=87)

    hour:        Optional[int] = Field(default=None, ge=0, le=23, example=14)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6,  example=2)
    is_weekend:  Optional[int] = Field(default=None, ge=0, le=1,  example=0)

    features: dict[str, float] = Field(
        default={},
        description="Pre-computed feature dict from features.py. If empty, "
                    "raw KPIs are used for the known base features only."
    )


class PredictionResult(BaseModel):
    cell_id:        str
    anomaly:        bool
    anomaly_score:  float = Field(..., description="Probability of anomaly (0–1)")
    threshold:      float
    severity:       str   = Field(..., description="normal / warning / critical")
    feature_source: str   = Field(
        ..., description="'precomputed' if caller supplied features, "
                          "'live' if computed from this cell's buffered history"
    )
    raw_kpis: dict[str, Any]


class BatchRequest(BaseModel):
    snapshots: list[KPISnapshot]


class BatchItemResult(BaseModel):
    cell_id: str
    ok:      bool
    result:  Optional[PredictionResult] = None
    error:   Optional[str] = None


class BatchResult(BaseModel):
    results:         list[BatchItemResult]
    anomaly_count:   int
    scored_count:    int
    error_count:     int
    total:           int


# ── Helpers ────────────────────────────────────────────────────────────────

def _raw_row(snapshot: KPISnapshot) -> dict:
    return make_raw_row(
        timestamp=snapshot.timestamp or now_iso(),
        cell_id=snapshot.cell_id,
        kpis={
            "latency_ms":      snapshot.latency_ms,
            "throughput_mbps": snapshot.throughput_mbps,
            "packet_loss_pct": snapshot.packet_loss_pct,
            "sinr_db":         snapshot.sinr_db,
            "connected_users": snapshot.connected_users,
        },
    )


def build_feature_vector(snapshot: KPISnapshot, prepared=None) -> tuple[np.ndarray, str]:
    """`prepared` is set by /predict/batch, which appends all readings to the history store in one
    round trip first: either (raw_row, buffer) or the HistoryStoreUnavailable that append raised."""
    if snapshot.features:
        try:
            vec = [snapshot.features[col] for col in FEATURE_COLS]
            return np.array(vec, dtype=float).reshape(1, -1), "precomputed"
        except KeyError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature: {e}. Provide all {len(FEATURE_COLS)} features."
            )

    try:
        if isinstance(prepared, HistoryStoreUnavailable):
            raise prepared
        if prepared is not None:
            raw_row, buf = prepared
        else:
            raw_row, buf = _raw_row(snapshot), None
        computed = history_store.add_and_compute(snapshot.cell_id, raw_row, buf=buf)
    except UndefinedFeatureError as e:
        record_undefined_features()
        raise HTTPException(status_code=422, detail=str(e))
    except HistoryStoreUnavailable as e:
        record_history_store_error()
        raise HTTPException(
            status_code=503,
            detail=f"History store unavailable ({e}). Retry shortly, or pass pre-computed `features`, "
                   "which do not use the store.",
            headers={"Retry-After": "5"},
        )

    if computed is None:
        try:
            rows_so_far = history_store.rows_available(snapshot.cell_id)
        except HistoryStoreUnavailable:
            rows_so_far = "?"
        record_cold_start()
        raise HTTPException(
            status_code=425,  # Too Early
            detail=(
                f"Not enough history for cell '{snapshot.cell_id}' yet to compute "
                f"time-series features ({rows_so_far}/{MIN_HISTORY_ROWS} readings "
                "buffered). Keep sending readings for this cell, in chronological "
                "order, or pass pre-computed `features` directly."
            ),
        )

    try:
        vec = [computed[col] for col in FEATURE_COLS]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Live feature computation is missing expected column: {e}"
        )
    return np.array(vec, dtype=float).reshape(1, -1), "live"


def score_to_severity(score: float, threshold: float) -> str:
    if score < threshold:
        return "normal"
    elif score < threshold + 0.15:
        return "warning"
    else:
        return "critical"


def run_inference(snapshot: KPISnapshot, prepared=None, log_sink: Optional[list] = None) -> PredictionResult:
    """`prepared` and `log_sink` are used by /predict/batch to share round trips: the sink collects
    inference-log entries and the caller writes them, and updates the cells gauge, once per batch."""
    X, feature_source = build_feature_vector(snapshot, prepared)
    prob = scorer.probability(X)
    is_anomaly = prob >= THRESHOLD
    severity = score_to_severity(prob, THRESHOLD)

    record_prediction(severity=severity, feature_source=feature_source, anomaly_score=prob,
                      cell_id=snapshot.cell_id)
    if log_sink is None:
        try:
            set_cells_tracked(history_store.tracked_cell_count())
        except HistoryStoreUnavailable:
            pass   # gauge keeps its last value; a metrics read must not fail a scored request

    # Best-effort log of the exact (unscaled) feature vector the model saw
    # — this is what monitoring/drift_check.py compares against training
    # data. Reconstructed from X rather than threaded through
    # build_feature_vector's return signature, to keep that function's
    # contract unchanged for existing callers/tests.
    features = dict(zip(FEATURE_COLS, X.flatten().tolist()))
    if log_sink is not None:
        log_sink.append((snapshot.cell_id, feature_source, prob, features))
    else:
        log_inference(
            cell_id=snapshot.cell_id,
            feature_source=feature_source,
            anomaly_score=prob,
            features=features,
            feature_cols=FEATURE_COLS,
        )

    return PredictionResult(
        cell_id=snapshot.cell_id,
        anomaly=is_anomaly,
        anomaly_score=round(prob, 4),
        threshold=THRESHOLD,
        severity=severity,
        feature_source=feature_source,
        raw_kpis={
            "latency_ms":      snapshot.latency_ms,
            "throughput_mbps": snapshot.throughput_mbps,
            "packet_loss_pct": snapshot.packet_loss_pct,
            "sinr_db":         snapshot.sinr_db,
            "connected_users": snapshot.connected_users,
        },
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    # No store round-trip here: this is the liveness/readiness probe, and a Redis outage should
    # degrade raw-reading requests (503), not take every pod out of rotation.
    return {"status": "ok", "model": "xgboost", "version": "1.0.0", "history_backend": history_store.backend_name}


@app.get("/model/info", dependencies=[Depends(require_api_key)])
def model_info():
    return {
        "threshold":    THRESHOLD,
        "metrics":      metadata["metrics"],
        "feature_count": len(FEATURE_COLS),
        "train_rows":   metadata["train_rows"],
        "test_rows":    metadata["test_rows"],
    }


@app.post("/predict", response_model=PredictionResult, dependencies=[Depends(require_api_key)])
def predict(snapshot: KPISnapshot):
    return run_inference(snapshot)


@app.post("/predict/batch", response_model=BatchResult, dependencies=[Depends(require_api_key)])
def predict_batch(request: BatchRequest):
    if len(request.snapshots) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500.")

    # Raw readings go to the history store in one round trip, scored readings are logged in one,
    # and the cells gauge is read once: ~3 Redis calls per reading become 3 per batch.
    live = [i for i, snap in enumerate(request.snapshots) if not snap.features]
    prepared: dict[int, Any] = {}
    if live:
        rows = [_raw_row(request.snapshots[i]) for i in live]
        try:
            bufs = history_store.append_many([(request.snapshots[i].cell_id, r) for i, r in zip(live, rows)])
            prepared = {i: (r, b) for i, r, b in zip(live, rows, bufs)}
        except HistoryStoreUnavailable as e:
            prepared = {i: e for i in live}

    items: list[BatchItemResult] = []
    log_sink: list = []
    for i, snapshot in enumerate(request.snapshots):
        try:
            result = run_inference(snapshot, prepared.get(i), log_sink)
            items.append(BatchItemResult(cell_id=snapshot.cell_id, ok=True, result=result))
        except HTTPException as e:
            items.append(BatchItemResult(cell_id=snapshot.cell_id, ok=False, error=e.detail))

    log_inferences(log_sink, FEATURE_COLS)
    try:
        set_cells_tracked(history_store.tracked_cell_count())
    except HistoryStoreUnavailable:
        pass

    anomaly_count = sum(1 for it in items if it.ok and it.result.anomaly)
    error_count   = sum(1 for it in items if not it.ok)

    return BatchResult(
        results=items,
        anomaly_count=anomaly_count,
        scored_count=len(items) - error_count,
        error_count=error_count,
        total=len(items),
    )
