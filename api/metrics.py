"""
metrics.py
----------
Prometheus instrumentation for the API.

Two layers:
  1. Generic HTTP metrics (request count, latency, in-progress requests)
     via prometheus-fastapi-instrumentator — labeled by route template and
     status code, which is what that library does by default and is safe
     cardinality-wise (fixed set of routes).
  2. Hand-written business metrics specific to anomaly detection: how many
     predictions land in each severity band, whether the request path used
     precomputed or live-computed features, how often cold-starts (425)
     happen, and the raw anomaly-score distribution.

Deliberately NOT labeled by cell_id.
-------------------------------------
The README describes "hundreds of cell towers" in a real deployment. A
Prometheus label creates one time series per unique label value, per
metric — labeling by cell_id would mean hundreds of time series per
metric, growing unboundedly as cells are added/decommissioned (a classic
"cardinality explosion" that degrades Prometheus's own performance and
storage). Severity and feature_source are small, fixed label sets
(3 and 2 values respectively) — safe. Per-cell drill-down, if needed,
belongs in logs/traces, not Prometheus labels.

All setter functions are no-ops-safe to call even if instrumentation
hasn't been wired up (e.g. during unit tests that import api.main without
a real ASGI server) — they just update in-process counters, never raise.
"""

import re

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# ── Metric definitions (module-level singletons, like the FastAPI app) ────

ANOMALY_PREDICTIONS_TOTAL = Counter(
    "anomaly_predictions_total",
    "Total predictions made, by severity band.",
    labelnames=["severity"],
)
# Create every severity series at 0 so Prometheus' increase() also sees the very first anomaly.
for _severity in ("normal", "warning", "critical"):
    ANOMALY_PREDICTIONS_TOTAL.labels(severity=_severity)

# Flagged readings only (warning or critical), per cell, so an alert and its email can name the cell.
# A cell_id label is unbounded cardinality in general, so it is limited twice: only flagged readings
# are counted, and cell_id values are checked for shape and capped (see _cell_label).
ANOMALY_DETECTIONS_TOTAL = Counter(
    "anomaly_detections_total",
    "Readings flagged as an anomaly (warning or critical), by cell and severity band.",
    labelnames=["cell_id", "severity"],
)
MAX_CELL_LABELS = 1000
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_labelled_cells: set = set()


def _cell_label(cell_id: str) -> str:
    """The cell_id label value: the id itself while it looks like one and the cap is not reached,
    else "other", so a client sending random ids cannot grow Prometheus' memory without limit."""
    if cell_id in _labelled_cells:
        return cell_id
    if _CELL_ID_RE.match(cell_id) and len(_labelled_cells) < MAX_CELL_LABELS:
        _labelled_cells.add(cell_id)
        return cell_id
    return "other"


FEATURE_SOURCE_TOTAL = Counter(
    "feature_source_total",
    "Total predictions, by which code path built the feature vector.",
    labelnames=["source"],  # "precomputed" | "live"
)

ANOMALY_SCORE = Histogram(
    "anomaly_score",
    "Distribution of raw anomaly probability scores returned by the model.",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

COLD_START_RESPONSES_TOTAL = Counter(
    "cold_start_responses_total",
    "Total 425 responses returned because a cell had insufficient history.",
)

UNDEFINED_FEATURE_RESPONSES_TOTAL = Counter(
    "undefined_feature_responses_total",
    "Total 422 responses because the newest reading's engineered features were "
    "undefined (e.g. previous packet_loss_pct was exactly 0). Distinct from cold starts.",
)

HISTORY_STORE_ERRORS_TOTAL = Counter(
    "history_store_errors_total",
    "Total 503 responses because the shared history store (Redis) was unreachable or errored.",
)

CELLS_TRACKED = Gauge(
    "cells_tracked",
    "Number of distinct cell_ids currently held in the in-process history "
    "store. NOT a substitute for a real fleet inventory — resets to 0 on "
    "every API restart since history is in-memory (see live_features.py).",
)

MODEL_THRESHOLD_INFO = Gauge(
    "model_threshold_info",
    "The anomaly-probability threshold currently loaded from metadata.json. "
    "A step change in this value on a dashboard marks a model redeploy.",
)


def instrument_app(app) -> None:
    """Wire up default HTTP metrics and expose GET /metrics. Call once,
    right after creating the FastAPI app instance."""
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )


def set_model_info(threshold: float) -> None:
    """Call once at startup after metadata.json is loaded."""
    MODEL_THRESHOLD_INFO.set(threshold)


def record_prediction(severity: str, feature_source: str, anomaly_score: float, cell_id: str = "") -> None:
    ANOMALY_PREDICTIONS_TOTAL.labels(severity=severity).inc()
    if severity != "normal" and cell_id:
        ANOMALY_DETECTIONS_TOTAL.labels(cell_id=_cell_label(cell_id), severity=severity).inc()
    FEATURE_SOURCE_TOTAL.labels(source=feature_source).inc()
    ANOMALY_SCORE.observe(anomaly_score)


def record_cold_start() -> None:
    COLD_START_RESPONSES_TOTAL.inc()


def record_undefined_features() -> None:
    UNDEFINED_FEATURE_RESPONSES_TOTAL.inc()


def record_history_store_error() -> None:
    HISTORY_STORE_ERRORS_TOTAL.inc()


def set_cells_tracked(count: int) -> None:
    CELLS_TRACKED.set(count)
