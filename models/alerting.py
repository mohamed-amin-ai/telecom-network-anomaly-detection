"""
alerting.py
-----------
Incident-level evaluation of a row-level detector — the way an operations
team experiences it, as opposed to row-level recall/precision.

Row-level metrics treat every 15-minute interval as independent. A NOC does
not: a 20-interval fault that is flagged on 3 intervals is ONE detected
incident (and the missed 17 intervals cost nothing once someone is paged),
while 40 scattered flagged intervals on a healthy cell are up to 40 pages.
So production numbers are usually quoted as:

  - incident detection rate  : fraction of real incidents with >= 1 alert
  - false alerts per cell-day: pages that match no real incident
  - time to detect           : intervals from incident onset to first flag

Consecutive flagged intervals are merged into a single alert (a flapping
detector pages once, not repeatedly), allowing small gaps (`max_gap`).
An alert "matches" an incident if they overlap within `tolerance` intervals
either side — ground-truth timing is itself imprecise in practice.

All functions are pure (numpy only) and assume rows are in time order
within each group.
"""

import numpy as np


def runs(mask) -> list:
    """Contiguous True runs as (start, end_exclusive) index pairs."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    padded = np.concatenate(([0], m.astype(np.int8), [0]))
    diff = np.diff(padded)
    return list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))


def merge_runs(run_list: list, max_gap: int) -> list:
    """Merge runs separated by at most `max_gap` unflagged intervals."""
    merged = []
    for start, end in run_list:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _overlaps(a_start, a_end, b_start, b_end, tolerance) -> bool:
    return a_start < b_end + tolerance and a_end > b_start - tolerance


def event_metrics(labels, pred, max_gap: int = 2, tolerance: int = 2) -> dict:
    """Incident-level counts for ONE contiguous time series."""
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    events = runs(labels)
    alerts = merge_runs(runs(pred), max_gap)

    detected, delays = 0, []
    for e_start, e_end in events:
        lo, hi = max(0, e_start - tolerance), min(len(pred), e_end + tolerance)
        hits = np.flatnonzero(pred[lo:hi])
        if hits.size:
            detected += 1
            delays.append(max(0, int(lo + hits[0]) - e_start))

    false_alerts = sum(
        1 for a_start, a_end in alerts
        if not any(_overlaps(a_start, a_end, e_start, e_end, tolerance) for e_start, e_end in events)
    )
    return {
        "n_events": len(events),
        "n_detected": detected,
        "n_alerts": len(alerts),
        "n_false_alerts": false_alerts,
        "delays": delays,
        "n_intervals": int(len(labels)),
    }


def grouped_event_metrics(labels, pred, groups, intervals_per_day: int,
                          max_gap: int = 2, tolerance: int = 2) -> dict:
    """Aggregate incident-level metrics over groups (e.g. cells). Rows must
    be time-ordered within each group."""
    labels, pred, groups = np.asarray(labels), np.asarray(pred), np.asarray(groups)
    totals = {"n_events": 0, "n_detected": 0, "n_alerts": 0, "n_false_alerts": 0, "n_intervals": 0}
    delays = []
    for g in np.unique(groups):
        sel = groups == g
        m = event_metrics(labels[sel], pred[sel], max_gap, tolerance)
        for k in totals:
            totals[k] += m[k]
        delays.extend(m["delays"])

    cell_days = totals["n_intervals"] / intervals_per_day
    return {
        **totals,
        "detection_rate": totals["n_detected"] / totals["n_events"] if totals["n_events"] else float("nan"),
        "alert_precision": (totals["n_alerts"] - totals["n_false_alerts"]) / totals["n_alerts"] if totals["n_alerts"] else float("nan"),
        "false_alerts_per_cell_day": totals["n_false_alerts"] / cell_days if cell_days else float("nan"),
        "median_delay_intervals": float(np.median(delays)) if delays else float("nan"),
    }
