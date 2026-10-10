"""
test_generate_data.py
-----------------------
Regression tests for data/generate_data.py's determinism.

Background: the generator originally seeded each cell's RNG from
`hash(cell) % 1000`. Python randomizes str hash() per process
(PYTHONHASHSEED), so every fresh run silently produced a DIFFERENT
dataset despite a pinned seed — which undermined every "reproducible"
claim downstream (DVC, the experiment log, the Airflow DAG docstring).
It was only caught by rerunning the script and noticing the anomaly
count changed. These tests make that class of bug fail loudly.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

_SNIPPET = (
    "import sys; sys.path.insert(0, %r);"
    "import pandas as pd;"
    "from data.generate_data import build_dataset;"
    "df = build_dataset();"
    "print(int(pd.util.hash_pandas_object(df, index=True).sum()) %% (2**63), len(df), int(df['anomaly'].sum()))"
) % str(REPO_ROOT)


def _run_with_hash_seed(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=True,
    )
    return result.stdout.strip()


def test_dataset_identical_across_processes_with_different_hash_seeds():
    """The core regression: two separate processes, deliberately given
    different string-hash seeds, must produce byte-identical datasets."""
    outputs = {seed: _run_with_hash_seed(seed) for seed in ("1", "2", "12345")}
    assert len(set(outputs.values())) == 1, (
        f"Dataset differs across PYTHONHASHSEED values — something is "
        f"seeding from hash() or other per-process state: {outputs}"
    )


def test_build_dataset_is_idempotent_within_one_process():
    """The anomaly injectors use a module-level rng; without reseeding in
    build_dataset(), a second call would continue the first call's random
    stream and return different data."""
    sys.path.insert(0, str(REPO_ROOT))
    import pandas as pd
    from data.generate_data import build_dataset

    first = build_dataset()
    second = build_dataset()
    pd.testing.assert_frame_equal(first, second)


def test_anomaly_events_scale_with_timeline_length():
    """Regression: the event count used to be a fixed 6-9 per cell no matter
    how long the timeline was, so 'more data' only added normal rows and the
    validation/test windows kept holding a handful of events.

    Each cell draws a base of 6-9 events at the 60-day reference length and
    scales it by (days / 60), so 3 cells inject between 18*k and 27*k faults
    for k = days/60. (The base draw is itself random, so asserting a RATIO
    between two runs would be flaky; asserting each total falls inside its
    own scaled range is exact.) The old flat behaviour (<= 27 total) fails
    this at k = 3.

    Counts the FAULTS INJECTED (the events sidecar), not runs in the label
    column: with label noise on, labels legitimately drop some faults and
    merge neighbours."""
    sys.path.insert(0, str(REPO_ROOT))
    import data.generate_data as gen

    original = gen.PERIODS
    try:
        for k in (1, 3):
            gen.PERIODS = 60 * k * 96
            n_faults = int((gen.build_dataset_and_events()[1].kind == "fault").sum())
            assert 18 * k <= n_faults <= 27 * k, f"{60 * k} days injected {n_faults} faults, expected {18 * k}-{27 * k}"
    finally:
        gen.PERIODS = original


# ── profile / realism regression tests ─────────────────────────────────────

EXPECTED_COLUMNS = ["timestamp", "cell_id", "latency_ms", "throughput_mbps", "packet_loss_pct",
                    "sinr_db", "connected_users", "anomaly"]


def _with_profile(profile, realism_overrides=None, days=60):
    """Context-free helper: returns a (df, events) built under the given
    profile, restoring module state afterwards."""
    sys.path.insert(0, str(REPO_ROOT))
    import data.generate_data as gen

    saved = (gen.PROFILE, dict(gen.REALISM), gen.PERIODS, gen.SEED)
    try:
        gen.SEED = 42    # pinned: these invariants are for seed 42, independent of params.yaml's default seed
        gen.PROFILE, gen.PERIODS = profile, days * 96
        if realism_overrides is not None:
            gen.REALISM.update(realism_overrides)
        return gen.build_dataset_and_events()
    finally:
        gen.PROFILE, gen.PERIODS, gen.SEED = saved[0], saved[2], saved[3]
        gen.REALISM.clear()
        gen.REALISM.update(saved[1])


ALL_ON = {"nonstationarity": True, "severity_and_ramp": True, "label_noise": True, "benign_surges": True}
ALL_OFF = {k: False for k in ALL_ON}


def test_legacy_profile_is_unchanged_from_the_original_generator():
    """Pinned invariants of the original clean-room dataset (60 days, seed 42):
    18,280-row shape and 328 anomalous rows. Byte-identity was verified
    directly against the pre-profile generator when profiles were added."""
    df, events = _with_profile("legacy", days=60)
    assert len(df) == 17280
    assert int(df["anomaly"].sum()) == 328
    assert set(events["kind"]) == {"fault"}


def test_every_profile_emits_exactly_the_original_columns():
    """Event metadata lives in a sidecar. An extra column in raw_kpis would
    be silently picked up as a MODEL FEATURE by get_feature_columns() — a
    label-leak waiting to happen."""
    for profile, overrides in (("legacy", None), ("realistic", ALL_ON), ("realistic", ALL_OFF)):
        df, _ = _with_profile(profile, overrides)
        assert list(df.columns) == EXPECTED_COLUMNS, (profile, list(df.columns))


def test_realistic_profile_with_all_factors_off_equals_legacy():
    """The factor switches must be genuinely independent: with every switch
    off, 'realistic' has to reduce to the legacy dataset exactly, or the
    ablation study would be measuring something else."""
    legacy, _ = _with_profile("legacy")
    off, _ = _with_profile("realistic", ALL_OFF)
    import pandas as pd
    pd.testing.assert_frame_equal(legacy, off)


def test_fault_positions_are_identical_across_profiles():
    """Ablations must change HOW the data looks, never WHICH events exist."""
    _, ev_legacy = _with_profile("legacy")
    _, ev_real = _with_profile("realistic", ALL_ON)
    key = ["cell_id", "type", "start_ts", "end_ts"]
    a = ev_legacy[ev_legacy.kind == "fault"][key].reset_index(drop=True)
    b = ev_real[ev_real.kind == "fault"][key].reset_index(drop=True)
    import pandas as pd
    pd.testing.assert_frame_equal(a, b)


def test_benign_surges_are_never_labeled_anomalies_and_match_requested_ratio():
    df, ev = _with_profile("realistic", {**ALL_OFF, "benign_surges": True, "benign_to_fault_ratio": 1.0})
    faults, benign = (ev.kind == "fault").sum(), (ev.kind == "benign").sum()
    assert benign == faults
    assert not ev[ev.kind == "benign"]["labeled"].any()
    # with label noise off, labeled rows are exactly the fault rows
    assert int(df["anomaly"].sum()) == int(
        sum(((e.end_ts - e.start_ts) / (df.timestamp.iloc[1] - df.timestamp.iloc[0])) for e in ev[ev.kind == "fault"].itertuples())
    )


def test_label_noise_leaves_some_faults_unlabeled_and_labels_only_where_logged():
    df, ev = _with_profile("realistic", {**ALL_OFF, "label_noise": True, "unlabeled_fault_prob": 0.3}, days=120)
    faults = ev[ev.kind == "fault"]
    assert (~faults["labeled"]).sum() > 0, "expected some unlabeled faults at p=0.3"
    # every positive label lies inside some logged label span for its cell
    for cell, g in df.groupby("cell_id"):
        spans = faults[(faults.cell_id == cell) & faults.labeled]
        pos = g.loc[g.anomaly == 1, "timestamp"]
        covered = pos.apply(lambda t: bool(((spans.label_start_ts <= t) & (t <= spans.label_end_ts)).any()))
        assert covered.all(), f"{cell}: labeled rows outside any logged event span"


def test_realistic_profile_is_deterministic_across_hash_seeds():
    """The realistic profile draws from its own RNG stream; it must be just
    as reproducible as legacy."""
    snippet = (
        "import sys; sys.path.insert(0, %r);"
        "import pandas as pd; import data.generate_data as g;"
        "g.PROFILE='realistic'; g.PERIODS=60*96;"
        "[g.REALISM.__setitem__(k, True) for k in ('nonstationarity','severity_and_ramp','label_noise','benign_surges')];"
        "df, ev = g.build_dataset_and_events();"
        "print(int(pd.util.hash_pandas_object(df, index=True).sum()) %% (2**63), len(ev))"
    ) % str(REPO_ROOT)
    outs = set()
    for seed in ("1", "2", "999"):
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}, cwd=str(REPO_ROOT), check=True)
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"realistic profile not deterministic: {outs}"


# ── resolve_profile (pure) ─────────────────────────────────────────────────

def test_resolve_profile_legacy_and_realistic():
    sys.path.insert(0, str(REPO_ROOT))
    import data.generate_data as gen

    profile, realism = gen.resolve_profile({"profile": "legacy"})
    assert profile == "legacy" and not any(realism[f] for f in gen._FACTOR_SWITCHES)

    profile, realism = gen.resolve_profile({"profile": "realistic", "realism": {"nonstationarity": True, "severity_min": 0.4}})
    assert profile == "realistic" and realism["nonstationarity"] is True and realism["severity_min"] == 0.4


def test_calibrated_profile_is_only_the_onset_ramp_and_ignores_other_realism_switches():
    """'calibrated' must mean exactly one thing — faint-free gradual onset —
    even if the realism block (used by the 'realistic' level) turns other
    factors on. Otherwise switching benchmark would silently inherit them."""
    sys.path.insert(0, str(REPO_ROOT))
    import data.generate_data as gen

    data_params = {
        "profile": "calibrated",
        "calibrated": {"severity_min": 1.0, "ramp_fraction": 0.1},
        "realism": {"nonstationarity": True, "label_noise": True, "benign_surges": True,
                    "severity_min": 0.4, "ramp_fraction": 0.3},
    }
    profile, realism = gen.resolve_profile(data_params)
    assert profile == "realistic"                      # alias: runs the realistic code path
    assert realism["severity_and_ramp"] is True
    assert not any(realism[f] for f in ("nonstationarity", "label_noise", "benign_surges"))
    assert realism["severity_min"] == 1.0 and realism["ramp_fraction"] == 0.1   # calibrated block wins


def test_ramp_fraction_zero_is_a_step_change():
    """The calibration dial's endpoint: ramp_fraction=0 means full effect from
    the first interval of every fault (what the clean generator does)."""
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import data.generate_data as gen

    assert np.array_equal(gen._ramp_weights(10, 0.0), np.ones(10))
    w = gen._ramp_weights(10, 0.3)
    assert w[0] < 1.0 and w[-1] == 1.0 and (np.diff(w) >= 0).all()
