"""
live_features.py
-----------------
Maintains a rolling per-cell history of raw KPI readings and computes the
SAME engineered features used at training time (lags, rolling stats,
z-scores, rate-of-change) for each new reading as it arrives.

Why this exists:
  The API previously accepted raw KPIs and, if the caller didn't supply
  pre-computed features, silently substituted 0.0 for the ~60 missing
  time-series features (lags/rolling stats/z-scores). The model would
  still return a confident-looking probability — just a wrong one, since
  it was scored on a mostly-zeroed feature vector. That's a silent
  correctness bug, not a graceful fallback.

  The fix: reuse data/features.py's engineer_features() directly against
  each cell's buffered history, so the API computes features exactly the
  way training did. No duplicated formulas, no train/serve skew.

Storage:
  By default history is kept in-process memory, keyed by cell_id: fine for
  one instance, lost on restart, and wrong across replicas (each would see
  only a fraction of a cell's readings and keep answering 425). Set
  REDIS_URL and the store uses RedisHistoryBackend instead, shared by all
  replicas. See api/shared_store.py.
"""

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from data.features import engineer_features, KPI_COLS, LAG_STEPS, LONG_WINDOW, SHORT_WINDOW

# Enough history so the most recent row survives engineer_features' dropna:
# needs LONG_WINDOW (16) prior points for the rolling window's min_periods,
# plus the max lag (8) already falls inside that. Keep a safety margin.
MIN_HISTORY_ROWS = LONG_WINDOW + 4          # 20 rows before we can score
MAX_HISTORY_ROWS = LONG_WINDOW + max(LAG_STEPS) + 10   # buffer cap per cell


def fast_latest_features(buf: list[dict], target_timestamp: str) -> dict | None:
    """Features of ONE reading (the one stamped `target_timestamp`) from a cell's buffered history,
    computed with plain NumPy instead of running data/features.py's full DataFrame pipeline.

    It follows engineer_features() exactly (same lags, same shift(1)-based rolling windows with the
    same min_periods, ddof=1 std, zero-std and zero-previous -> undefined), and returns None whenever
    it cannot be sure it would match: duplicate or non-string timestamps, the target missing, or any
    feature undefined. The caller then falls back to engineer_features(), which also raises the
    detailed UndefinedFeatureError, so behaviour is unchanged in every edge case. Equality with the
    original is checked by tests/test_live_features_fast.py.
    """
    if not all(isinstance(r.get("timestamp"), str) for r in buf):
        return None
    rows = sorted(buf, key=lambda r: r["timestamp"])
    stamps = [r["timestamp"] for r in rows]
    if len(set(stamps)) != len(stamps):
        return None
    try:
        i = stamps.index(target_timestamp)
    except ValueError:
        return None

    def window(x, w, min_periods):
        vals = x[max(0, i - w):i]                 # == series.shift(1).rolling(w) evaluated at row i
        vals = vals[~np.isnan(vals)]
        return vals if len(vals) >= min_periods else None

    out: dict = {}
    for kpi in KPI_COLS:
        x = np.array([float(r[kpi]) for r in rows])
        cur = x[i]
        out[kpi] = cur
        for lag in LAG_STEPS:
            if i - lag < 0:
                return None
            out[f"{kpi}_lag{lag}"] = x[i - lag]
        stats_4h = None
        for w, label in ((SHORT_WINDOW, "1h"), (LONG_WINDOW, "4h")):
            vals = window(x, w, w // 2)
            if vals is None or len(vals) < 2:
                return None
            if vals.max() == vals.min():
                return None       # flat window: pandas' rolling std carries ~1e-7 noise, so use the original path
            mean, std = vals.mean(), vals.std(ddof=1)
            out[f"{kpi}_roll_mean_{label}"] = mean
            out[f"{kpi}_roll_std_{label}"] = std
            if label == "4h":
                stats_4h = (mean, std)
        vals = window(x, SHORT_WINDOW, 2)
        if vals is None:
            return None
        out[f"{kpi}_roll_min_1h"] = vals.min()
        out[f"{kpi}_roll_max_1h"] = vals.max()
        mean4, std4 = stats_4h
        if std4 == 0:
            return None
        out[f"{kpi}_zscore"] = (cur - mean4) / std4
        prev = x[i - 1]
        if prev == 0:
            return None
        out[f"{kpi}_roc"] = (cur - prev) / prev

    if any(np.isnan(v) for v in out.values()):
        return None
    ts = pd.to_datetime(target_timestamp, format="ISO8601", utc=True)
    out["hour"] = ts.hour
    out["day_of_week"] = ts.dayofweek
    out["is_weekend"] = int(ts.dayofweek >= 5)
    return out


# Set LIVE_FEATURES_FAST=0 to force the original engineer_features() path for every request.
USE_FAST_FEATURES = os.environ.get("LIVE_FEATURES_FAST", "1") != "0"


class UndefinedFeatureError(Exception):
    """The newest reading is present in the buffer, but one or more of its
    engineered features is undefined (NaN), so engineer_features() dropped
    the row. Distinct from a cold start (None): history is sufficient, the
    reading itself just can't be featurized.

    Typical cause: the PREVIOUS reading's value for some KPI was exactly 0
    (e.g. packet_loss_pct == 0.0), so that KPI's rate of change divides by
    zero. Also: a KPI perfectly flat across the 4h window (zero std, so
    z-score is undefined)."""


class HistoryStoreUnavailable(Exception):
    """The shared history backend (Redis) could not be reached or errored.
    The API maps this to HTTP 503 for raw-reading requests; requests that
    supply pre-computed features never touch the store and are unaffected."""


class MemoryHistoryBackend:
    """Per-process ring buffer, one deque per cell_id. The default, and what
    docker-compose uses. NOT shared between processes — see RedisHistoryBackend."""

    name = "memory"

    def __init__(self):
        self._history: dict[str, deque] = {}

    def append(self, cell_id: str, row: dict) -> list[dict]:
        buf = self._history.setdefault(cell_id, deque(maxlen=MAX_HISTORY_ROWS))
        buf.append(row)
        return list(buf)

    def length(self, cell_id: str) -> int:
        return len(self._history.get(cell_id, ()))

    def cell_count(self) -> int:
        return len(self._history)


class RedisHistoryBackend:
    """Per-cell history shared by every API replica, so a cell's readings can
    arrive on any pod and still accumulate into one history.

    Layout: a Redis LIST per cell (`<prefix>:cell:<cell_id>`, JSON rows,
    trimmed to MAX_HISTORY_ROWS) and a ZSET of cell_id -> last-seen time
    (`<prefix>:cells`) for the cells_tracked gauge. Keys expire after
    `ttl_seconds` without a reading, so retired cells don't accumulate.

    append() is one MULTI/EXEC transaction: push, trim and read-back happen
    atomically, so two pods writing the same cell see a consistent buffer.
    Rows are scored in timestamp order by engineer_features(), so arrival
    order across pods doesn't change the features.
    """

    name = "redis"

    def __init__(self, client, ttl_seconds: int = 86400, key_prefix: str = "anomaly:history", errors=None):
        self._client = client
        self._ttl = int(ttl_seconds)
        self._prefix = key_prefix
        if errors is None:
            import redis
            errors = (redis.RedisError, OSError)
        self._errors = errors

    def _key(self, cell_id: str) -> str:
        return f"{self._prefix}:cell:{cell_id}"

    @property
    def _cells_key(self) -> str:
        return f"{self._prefix}:cells"

    def append(self, cell_id: str, row: dict) -> list[dict]:
        import json
        import time
        now = time.time()
        key = self._key(cell_id)
        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.rpush(key, json.dumps(row))
            pipe.ltrim(key, -MAX_HISTORY_ROWS, -1)
            pipe.lrange(key, 0, -1)
            pipe.expire(key, self._ttl)
            pipe.zadd(self._cells_key, {cell_id: now})
            pipe.zremrangebyscore(self._cells_key, "-inf", now - self._ttl)
            results = pipe.execute()
        except self._errors as e:
            raise HistoryStoreUnavailable(f"history store unavailable: {e}") from e
        return [json.loads(x) for x in results[2]]

    def length(self, cell_id: str) -> int:
        try:
            return int(self._client.llen(self._key(cell_id)))
        except self._errors as e:
            raise HistoryStoreUnavailable(f"history store unavailable: {e}") from e

    def cell_count(self) -> int:
        import time
        try:
            return int(self._client.zcount(self._cells_key, time.time() - self._ttl, "+inf"))
        except self._errors as e:
            raise HistoryStoreUnavailable(f"history store unavailable: {e}") from e


class CellHistoryStore:
    """Rolling raw-reading history per cell_id, over a pluggable backend
    (in-memory by default; Redis for multi-replica deployments)."""

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else MemoryHistoryBackend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def rows_available(self, cell_id: str) -> int:
        return self._backend.length(cell_id)

    def tracked_cell_count(self) -> int:
        """Number of distinct cell_ids with any buffered history at all
        (regardless of whether they've cleared MIN_HISTORY_ROWS yet).
        Used for the cells_tracked Prometheus gauge — see api/metrics.py."""
        return self._backend.cell_count()

    def add_and_compute(self, cell_id: str, raw_row: dict) -> dict | None:
        """
        Append a new raw reading for this cell and compute its feature
        vector using the exact training-time logic.

        Returns the feature dict for the LATEST row, or None if there
        isn't yet enough history for the rolling windows to be valid
        (cold start — caller should backfill more readings first).
        Raises HistoryStoreUnavailable if a shared backend is down.
        """
        buf = self._backend.append(cell_id, raw_row)

        if len(buf) < MIN_HISTORY_ROWS:
            return None

        if USE_FAST_FEATURES:
            fast = fast_latest_features(buf, raw_row["timestamp"])
            if fast is not None:
                return fast

        df = pd.DataFrame(buf)
        df["cell_id"] = cell_id
        df["anomaly"] = 0  # placeholder column; engineer_features doesn't use it for X

        feat_df = engineer_features(df, verbose=False)

        # Select the row for the reading we JUST added, by timestamp — do not
        # take feat_df.iloc[-1]. engineer_features() drops any row with an
        # undefined feature, so when the newest reading is dropped, iloc[-1]
        # is the PREVIOUS reading, and the API would silently score stale
        # features as if they were current (found by replaying real data:
        # 9 of 17,247 rows, every one following a packet_loss_pct == 0.0).
        matches = feat_df[feat_df["timestamp"] == raw_row["timestamp"]]
        if matches.empty:
            raise UndefinedFeatureError(
                f"Cell '{cell_id}': features are undefined for the reading at "
                f"{raw_row['timestamp']} (history is sufficient, but a feature is NaN — "
                f"typically the previous reading's packet_loss_pct/other KPI was exactly 0, "
                f"making its rate of change undefined, or a KPI was perfectly flat over the "
                f"window). Not scoring stale features."
            )
        latest = matches.iloc[-1]
        return latest.drop(labels=["timestamp", "cell_id", "anomaly"]).to_dict()


def make_raw_row(timestamp: str, cell_id: str, kpis: dict) -> dict:
    """Build the raw-row dict shape engineer_features() expects."""
    row = {"timestamp": timestamp, "cell_id": cell_id}
    for col in KPI_COLS:
        row[col] = kpis[col]
    return row


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
