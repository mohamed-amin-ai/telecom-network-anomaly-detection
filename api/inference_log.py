"""
inference_log.py
-----------------
Appends each scored request's feature vector to a local CSV log, so
drift detection (monitoring/drift_check.py) has real production data to
compare against the training distribution.

Why this didn't already exist: api/live_features.py's CellHistoryStore
is in-memory and per-cell — it's built to answer "does this cell have
enough recent history to compute features right now", not "what did we
serve over the last N days". Those are different questions with
different lifetimes; conflating them would mean either losing drift
history on every restart, or holding unbounded history in memory
forever. This is a separate, append-only, disk-backed log.

Design choices:
  - Local CSV by default (docker-compose, single instance). When REDIS_URL
    is set (Kubernetes, several replicas) rows go to a capped Redis list
    instead, so every replica's traffic lands in one place the drift-check
    job can read. Redis here is a bounded buffer of the newest rows
    (INFERENCE_LOG_MAX_ROWS), not an archive — a real long-term record
    would ship rows to cloud logging / a feature store. Documented, not hidden.
  - Logging failures NEVER raise — a full disk or a permissions issue
    must not take down inference. Every write is best-effort.
  - Logs the ENGINEERED feature vector (all FEATURE_COLS), not just the
    raw KPIs — drift detection needs to see what the model actually
    saw, not just the inputs that were used to derive it.
"""

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from api import shared_store

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_PATH = LOG_DIR / "inference_log.csv"

_lock = threading.Lock()  # csv.writer isn't thread-safe across processes' fd

REDIS_LOG_KEY = "anomaly:inference_log"
_redis_client = None


def _redis():
    """Lazily-built shared client, or None when REDIS_URL is unset."""
    global _redis_client
    if _redis_client is None:
        _redis_client = shared_store.get_redis_client()
    return _redis_client


def _fieldnames(feature_cols: Sequence[str]) -> list[str]:
    return ["timestamp", "cell_id", "feature_source", "anomaly_score", *feature_cols]


def log_inference(
    cell_id: str,
    feature_source: str,
    anomaly_score: float,
    features: dict,
    feature_cols: Sequence[str],
) -> None:
    """Best-effort append. Never raises — a logging failure must not
    break the actual prediction response."""
    try:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell_id": cell_id,
            "feature_source": feature_source,
            "anomaly_score": anomaly_score,
            **{col: features.get(col) for col in feature_cols},
        }

        client = _redis()
        if client is not None:
            pipe = client.pipeline(transaction=True)
            pipe.lpush(REDIS_LOG_KEY, json.dumps(row))
            pipe.ltrim(REDIS_LOG_KEY, 0, shared_store.inference_log_max_rows() - 1)
            pipe.execute()
            return

        LOG_DIR.mkdir(exist_ok=True)
        fieldnames = _fieldnames(feature_cols)
        is_new_file = not LOG_PATH.exists()

        with _lock:
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if is_new_file:
                    writer.writeheader()
                writer.writerow(row)
    except Exception:
        # Deliberately swallowed. Observability must never take down
        # the thing it's observing. If you need to debug why logging
        # isn't working, temporarily remove this try/except locally —
        # don't remove it in the committed version.
        pass


def read_log(feature_cols: Sequence[str] | None = None):
    """Read the inference log back as a pandas DataFrame. Returns None
    (not an empty DataFrame) if the log doesn't exist yet, so callers
    can distinguish "no production traffic yet" from "empty dataset"."""
    import pandas as pd

    client = _redis()
    if client is not None:
        # Raises on a Redis error, on purpose: a drift job that can't read its input
        # must fail loudly, not report "no traffic yet".
        raw = client.lrange(REDIS_LOG_KEY, 0, -1)
        if not raw:
            return None
        df = pd.DataFrame([json.loads(r) for r in reversed(raw)])   # oldest first, like the CSV
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        return _check_columns(df, feature_cols, "Redis key " + REDIS_LOG_KEY)

    if not LOG_PATH.exists():
        return None

    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return _check_columns(df, feature_cols, str(LOG_PATH))


def _check_columns(df, feature_cols, source: str):
    if feature_cols is not None:
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Inference log at {source} is missing expected feature "
                f"columns: {sorted(missing)}. Was it written by an older "
                f"version of the model (different FEATURE_COLS)? If so, "
                f"archive/delete the old log and let it rebuild."
            )
    return df
