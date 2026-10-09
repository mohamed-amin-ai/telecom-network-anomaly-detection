"""
test_alerting.py
-----------------
Tests models/alerting.py — incident-level (alert) metrics. Pure numpy.
Each case is small enough to verify by hand.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from models.alerting import event_metrics, grouped_event_metrics, merge_runs, runs


def _m(s: str):
    """'..XX..X' -> boolean mask."""
    return np.array([c == "X" for c in s])


def test_runs_finds_contiguous_blocks():
    assert runs(_m("..XX..X.")) == [(2, 4), (6, 7)]
    assert runs(_m("XXXX")) == [(0, 4)]
    assert runs(_m("....")) == []
    assert runs([]) == []


def test_merge_runs_respects_max_gap():
    r = [(2, 4), (5, 7), (12, 13)]
    assert merge_runs(r, max_gap=1) == [(2, 7), (12, 13)]     # gap of 1 merges
    assert merge_runs(r, max_gap=0) == r                       # gap of 1 > 0: no merge
    assert merge_runs(r, max_gap=10) == [(2, 13)]


def test_one_event_flagged_on_few_rows_is_one_detected_incident():
    labels = _m("...XXXXXXXXXX...")          # 10-row fault
    pred = _m("...X..X..........")           # only 2 of 10 rows flagged
    m = event_metrics(labels, pred, max_gap=2, tolerance=0)
    assert m["n_events"] == 1 and m["n_detected"] == 1
    assert m["n_alerts"] == 1 and m["n_false_alerts"] == 0   # flapping merged into one alert


def test_missed_event_and_false_alert_are_counted():
    labels = _m("..XXX.........XXX..")
    pred = _m("..................X..")        # flags the second event? no: index 18 is after it
    m = event_metrics(labels, pred, max_gap=2, tolerance=0)
    assert m["n_events"] == 2 and m["n_detected"] == 0
    assert m["n_alerts"] == 1 and m["n_false_alerts"] == 1


def test_tolerance_lets_slightly_early_or_late_alerts_match():
    labels = _m("......XXX......")
    late = _m(".........X.....")             # flag 1 interval after the event ends
    assert event_metrics(labels, late, 2, tolerance=0)["n_detected"] == 0
    assert event_metrics(labels, late, 2, tolerance=2)["n_detected"] == 1
    assert event_metrics(labels, late, 2, tolerance=2)["n_false_alerts"] == 0


def test_delay_is_intervals_from_onset_to_first_flag():
    labels = _m("..XXXXXX..")
    pred = _m("....XX....")                   # first flag at index 4, onset at 2
    assert event_metrics(labels, pred, 2, 0)["delays"] == [2]


def test_early_flag_within_tolerance_has_zero_delay_not_negative():
    labels = _m("....XXX...")
    pred = _m("..X.......")                   # flagged 2 intervals BEFORE onset
    m = event_metrics(labels, pred, 2, tolerance=2)
    assert m["delays"] == [0]


def test_gap_merging_prevents_paging_storms():
    labels = np.zeros(30, dtype=bool)
    pred = _m("..X.X.X.X.X.X.X.X.X.X.........")   # flapping every other interval, no real fault
    assert event_metrics(labels, pred, max_gap=2, tolerance=0)["n_false_alerts"] == 1
    assert event_metrics(labels, pred, max_gap=0, tolerance=0)["n_false_alerts"] == 10


def test_grouped_metrics_aggregate_and_convert_to_cell_days():
    # two "cells" of 96 intervals (one day each): cell A has 1 detected event; cell B has 1 false alert
    a_lab = np.zeros(96, dtype=bool); a_lab[10:15] = True
    a_pred = np.zeros(96, dtype=bool); a_pred[11] = True
    b_lab = np.zeros(96, dtype=bool)
    b_pred = np.zeros(96, dtype=bool); b_pred[50] = True
    labels = np.concatenate([a_lab, b_lab]); pred = np.concatenate([a_pred, b_pred])
    groups = np.array(["A"] * 96 + ["B"] * 96)

    m = grouped_event_metrics(labels, pred, groups, intervals_per_day=96, max_gap=2, tolerance=0)
    assert m["n_events"] == 1 and m["n_detected"] == 1 and m["detection_rate"] == 1.0
    assert m["n_alerts"] == 2 and m["n_false_alerts"] == 1
    assert m["alert_precision"] == 0.5
    assert m["false_alerts_per_cell_day"] == 0.5          # 1 false alert over 2 cell-days
    assert m["median_delay_intervals"] == 1.0


def test_no_events_gives_nan_detection_rate_not_a_crash():
    m = grouped_event_metrics(np.zeros(10, dtype=bool), np.zeros(10, dtype=bool),
                              np.array(["A"] * 10), intervals_per_day=96)
    assert math.isnan(m["detection_rate"]) and m["n_false_alerts"] == 0
