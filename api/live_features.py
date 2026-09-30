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

Limitation (documented, not hidden):
  History is kept in-process memory, keyed by cell_id. This is fine for a
  single-instance demo/portfolio service, but does not survive a restart
  and won't work across multiple API replicas. A real deployment would
  back this with a shared store (Redis / feature store) keyed by cell_id.
"""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from data.features import engineer_features, KPI_COLS, LAG_STEPS, LONG_WINDOW

# Enough history so the most recent row survives engineer_features' dropna:
# needs LONG_WINDOW (16) prior points for the rolling window's min_periods,
# plus the max lag (8) already falls inside that. Keep a safety margin.
MIN_HISTORY_ROWS = LONG_WINDOW + 4          # 20 rows before we can score
MAX_HISTORY_ROWS = LONG_WINDOW + max(LAG_STEPS) + 10   # buffer cap per cell


class UndefinedFeatureError(Exception):
    """The newest reading is present in the buffer, but one or more of its
    engineered features is undefined (NaN), so engineer_features() dropped
    the row. Distinct from a cold start (None): history is sufficient, the
    reading itself just can't be featurized.

    Typical cause: the PREVIOUS reading's value for some KPI was exactly 0
    (e.g. packet_loss_pct == 0.0), so that KPI's rate of change divides by
    zero. Also: a KPI perfectly flat across the 4h window (zero std, so
    z-score is undefined)."""


class CellHistoryStore:
    """In-memory ring buffer of raw KPI readings, one deque per cell_id."""

    def __init__(self):
        self._history: dict[str, deque] = {}

    def _buffer_for(self, cell_id: str) -> deque:
        if cell_id not in self._history:
            self._history[cell_id] = deque(maxlen=MAX_HISTORY_ROWS)
        return self._history[cell_id]

    def rows_available(self, cell_id: str) -> int:
        return len(self._history.get(cell_id, ()))

    def tracked_cell_count(self) -> int:
        """Number of distinct cell_ids with any buffered history at all
        (regardless of whether they've cleared MIN_HISTORY_ROWS yet).
        Used for the cells_tracked Prometheus gauge — see api/metrics.py."""
        return len(self._history)

    def add_and_compute(self, cell_id: str, raw_row: dict) -> dict | None:
        """
        Append a new raw reading for this cell and compute its feature
        vector using the exact training-time logic.

        Returns the feature dict for the LATEST row, or None if there
        isn't yet enough history for the rolling windows to be valid
        (cold start — caller should backfill more readings first).
        """
        buf = self._buffer_for(cell_id)
        buf.append(raw_row)

        if len(buf) < MIN_HISTORY_ROWS:
            return None

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
