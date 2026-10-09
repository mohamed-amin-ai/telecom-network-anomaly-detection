"""
generate_data.py
----------------
Generates synthetic telecom network KPI time-series data with realistic
anomaly patterns. Simulates 3 cell towers over 60 days at 15-min intervals.

KPIs simulated:
  - latency_ms       : packet round-trip latency (ms)
  - throughput_mbps  : downlink throughput (Mbps)
  - packet_loss_pct  : % of packets lost
  - sinr_db          : Signal-to-Interference-plus-Noise Ratio (dB)
  - connected_users  : active UEs on cell

Anomaly types injected:
  - Latency spike      : sudden latency jump (congestion / backhaul issue)
  - Throughput drop    : sustained throughput degradation (hardware fault)
  - Packet loss burst  : bursty packet loss (interference / link instability)
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"


def _load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["data"]


_params = _load_params()

SEED = _params["seed"]
rng = np.random.default_rng(SEED)

CELLS = _params["cells"]
START = "2024-01-01"
PERIODS = _params["periods_days"] * 24 * (60 // _params["freq_minutes"])  # intervals total
FREQ = f"{_params['freq_minutes']}min"
INTERVALS_PER_DAY = 24 * (60 // _params["freq_minutes"])

# Profiles:
#   "legacy"     the original clean-room generator (byte-identical to before
#                profiles existed).
#   "realistic"  adds the difficulty factors below, each independently
#                switchable so their individual effect can be measured
#                (scripts/realism_study.py).
#   "calibrated" an ALIAS for the realistic code path with only
#                `severity_and_ramp` on and its visibility dial
#                (`severity_min`) taken from params.yaml -> data.calibrated.
#                That dial was chosen by scripts/calibrate_to_original.py so a
#                baseline model lands near the ORIGINAL project's reported
#                metrics. The match is therefore by construction — see
#                EXPERIMENTS.md before quoting it.
# PROFILE/REALISM are modified at runtime by the study scripts and by tests.
_FACTOR_SWITCHES = ("nonstationarity", "severity_and_ramp", "label_noise", "benign_surges")
_REALISM_DEFAULTS = {
    "nonstationarity": False, "severity_and_ramp": False, "label_noise": False, "benign_surges": False,
    "severity_min": 0.15, "ramp_fraction": 0.3, "label_jitter_intervals": 2,
    "unlabeled_fault_prob": 0.10, "benign_to_fault_ratio": 1.0,
    "cell_scale_range": [0.8, 1.25], "latency_drift_pct": 10, "throughput_drift_pct": -8,
}


def resolve_profile(data_params: dict):
    """Map params.yaml's data block to the (profile, realism) the generator
    actually runs. Pure function (tested in tests/test_generate_data.py)."""
    profile = data_params.get("profile", "legacy")
    realism = {**_REALISM_DEFAULTS, **data_params.get("realism", {})}
    if profile == "calibrated":
        realism.update({f: False for f in _FACTOR_SWITCHES})
        realism["severity_and_ramp"] = True
        realism.update(data_params["calibrated"])
        profile = "realistic"
    return profile, realism


CONFIGURED_PROFILE = _params.get("profile", "legacy")   # as written in params.yaml (may be "calibrated")
PROFILE, REALISM = resolve_profile(_params)


# ── Baseline KPI generators ────────────────────────────────────────────────

def base_latency(n, rng):
    """Normal latency: 20–60 ms with daily cycle + realistic noise bursts."""
    t = np.arange(n)
    daily = 10 * np.sin(2 * np.pi * t / 96)
    # Higher base noise + occasional non-anomaly spikes (natural jitter)
    noise = rng.normal(0, 8, n)
    jitter = rng.exponential(5, n) * (rng.random(n) < 0.05)  # 5% random spikes
    return np.clip(35 + daily + noise + jitter, 5, 200)

def base_throughput(n, rng):
    """Throughput: 50–150 Mbps with higher variance and random dips."""
    t = np.arange(n)
    daily = -20 * np.sin(2 * np.pi * t / 96)
    noise = rng.normal(0, 12, n)
    # Occasional legitimate short dips (handover, scheduling)
    dips = rng.uniform(0.6, 0.9, n) * (rng.random(n) < 0.04)
    base = np.clip(100 + daily + noise, 10, 300)
    return np.where(dips > 0, base * dips, base)

def base_packet_loss(n, rng):
    """Packet loss: noisy baseline with frequent small spikes."""
    base = rng.exponential(0.8, n)       # higher mean noise
    spikes = rng.exponential(1.5, n) * (rng.random(n) < 0.08)  # 8% random spikes
    return np.clip(base + spikes, 0, 5)  # allow up to 5% normal loss

def base_sinr(n, rng):
    """SINR: more variation to simulate real interference environment."""
    t = np.arange(n)
    daily = 3 * np.sin(2 * np.pi * t / 96 + np.pi)
    noise = rng.normal(0, 3.5, n)        # wider spread
    interference = -rng.exponential(2, n) * (rng.random(n) < 0.06)
    return np.clip(15 + daily + noise + interference, -5, 30)

def base_users(n, rng):
    """Connected users: noisy daily pattern with random surges."""
    t = np.arange(n)
    daily = 80 * np.clip(np.sin(2 * np.pi * (t / 96 - 0.2)), 0, 1)
    noise = rng.normal(0, 12, n)
    surges = rng.uniform(20, 60, n) * (rng.random(n) < 0.03)
    return np.clip(20 + daily + noise + surges, 1, 300).astype(int)


# ── Anomaly injectors ──────────────────────────────────────────────────────

def inject_latency_spike(latency, start, duration=8):
    """Inject a latency spike — range reduced to overlap with noise floor."""
    latency = latency.copy()
    # Softer range: some spikes will be borderline vs natural jitter
    latency[start:start+duration] += rng.uniform(30, 120, duration)
    return latency

def inject_throughput_drop(throughput, start, duration=24):
    """Inject throughput degradation — less extreme to create ambiguous cases."""
    throughput = throughput.copy()
    # 0.25–0.6x instead of 0.1–0.3x: some drops look like natural dips
    throughput[start:start+duration] *= rng.uniform(0.25, 0.6)
    return throughput

def inject_packet_loss_burst(packet_loss, start, duration=6):
    """Inject packet loss burst — softer range overlaps with noisy baseline."""
    packet_loss = packet_loss.copy()
    # 3–10% instead of 5–15%: borderline cases force the model to work harder
    packet_loss[start:start+duration] += rng.uniform(3, 10, duration)
    return packet_loss


# ── Realism factors (profile == "realistic") ───────────────────────────────
# All randomness below comes from a SEPARATE stream (`real_rng`), never the
# legacy streams, so (a) the legacy profile is untouched and (b) the same
# fault positions/types/durations are selected in every profile — ablations
# change only HOW the data looks, not WHICH events exist.

def _ramp_weights(dur: int, ramp_fraction: float) -> np.ndarray:
    """Gradual onset: effect ramps 0 -> 1 over the first `ramp_fraction` of
    the event. Real degradations rarely switch on like a step."""
    ramp_len = max(1, int(round(ramp_fraction * dur)))
    return np.minimum(1.0, (np.arange(dur) + 1) / ramp_len)


def _apply_nonstationarity(n, real_rng, latency, throughput, users):
    """Per-cell baselines, weekly seasonality, and slow drift. Real cells
    differ in their normal operating point, weekends carry less traffic, and
    load creeps up over months — a model trained on early data meets a
    shifted world later (the time-based split makes that visible)."""
    lo, hi = REALISM["cell_scale_range"]
    lat_scale, thr_scale = real_rng.uniform(lo, hi), real_rng.uniform(lo, hi)
    t = np.arange(n)
    frac = t / max(1, n - 1)
    weekend = ((t // INTERVALS_PER_DAY) % 7) >= 5          # START is a Monday
    latency = latency * lat_scale * (1 + REALISM["latency_drift_pct"] / 100 * frac) * np.where(weekend, 0.92, 1.0)
    throughput = throughput * thr_scale * (1 + REALISM["throughput_drift_pct"] / 100 * frac) * np.where(weekend, 1.08, 1.0)
    users = users.astype(float) * np.where(weekend, 0.7, 1.0)
    return latency, throughput, users


def _inject_fault_realistic(kind, start, dur, latency, throughput, packet_loss, real_rng) -> float:
    """Severity is log-uniform on [severity_min, 1]: many faults are mild
    (barely distinguishable from normal variation), a few are dramatic.
    Returns the severity drawn. Modifies the arrays in place."""
    severity = float(np.exp(real_rng.uniform(np.log(REALISM["severity_min"]), 0.0)))
    w = _ramp_weights(dur, REALISM["ramp_fraction"])
    sl = slice(start, start + dur)
    if kind == "latency":
        latency[sl] += w * severity * real_rng.uniform(30, 120, dur)
    elif kind == "throughput":
        floor_mult = real_rng.uniform(0.25, 0.6)
        throughput[sl] *= 1 - w * severity * (1 - floor_mult)
    else:
        packet_loss[sl] += w * severity * real_rng.uniform(3, 10, dur)
    return severity


def _inject_benign_surges(n, n_faults, real_rng, latency, throughput, packet_loss, users) -> list:
    """Load surges (events, outages elsewhere, rerouting): latency up,
    throughput down, loss up, users up — and NOT faults. In production the
    bulk of false alarms are real but benign episodes like these, which
    look like mild faults on most KPIs."""
    events = []
    for _ in range(int(round(n_faults * REALISM["benign_to_fault_ratio"]))):
        dur = int(real_rng.integers(4, 25))
        start = int(real_rng.integers(50, n - dur - 50))
        w = _ramp_weights(dur, REALISM["ramp_fraction"])
        sl = slice(start, start + dur)
        users[sl] = users[sl] * (1 + w * real_rng.uniform(0.5, 1.5))
        latency[sl] += w * real_rng.uniform(10, 40)
        throughput[sl] *= 1 - w * (1 - real_rng.uniform(0.7, 0.9))
        packet_loss[sl] += w * real_rng.uniform(0.5, 2.0)
        events.append({"start": start, "dur": dur, "kind": "benign", "type": "surge", "severity": float("nan"),
                       "labeled": False, "label_start": None, "label_end": None})
    return events


def _noisy_labels(n, fault_events, real_rng) -> np.ndarray:
    """Ground truth in production comes from tickets/alarms: onset and end
    are logged late or early, and some real incidents are never logged at
    all. Boundaries jitter by +/- label_jitter_intervals; a fraction of
    faults are left unlabeled (the detector will then be "wrong" for
    correctly flagging them — as it is in real evaluations)."""
    labels = np.zeros(n, dtype=int)
    jitter, p_unlabeled = REALISM["label_jitter_intervals"], REALISM["unlabeled_fault_prob"]
    for ev in fault_events:
        if real_rng.random() < p_unlabeled:
            ev["labeled"] = False
            continue
        d_start, d_end = (int(x) for x in real_rng.integers(-jitter, jitter + 1, size=2))
        l_start = max(0, ev["start"] + d_start)
        l_end = min(n, ev["start"] + ev["dur"] + d_end)
        if l_end <= l_start:
            l_start, l_end = ev["start"], ev["start"] + ev["dur"]
        labels[l_start:l_end] = 1
        ev.update(labeled=True, label_start=l_start, label_end=l_end)
    return labels


# ── Main generation ────────────────────────────────────────────────────────

def generate_cell_data(cell_id: str, n: int, rng, cell_index: int = 0):
    """Returns (DataFrame, list-of-event-dicts). The DataFrame's columns are
    exactly the 8 below in every profile — event metadata never goes in it,
    because any extra column would silently be treated as a model feature."""
    realistic = PROFILE == "realistic"
    real_rng = np.random.default_rng([SEED, cell_index, 7919]) if realistic else None
    timestamps = pd.date_range(START, periods=n, freq=FREQ)

    latency     = base_latency(n, rng)
    throughput  = base_throughput(n, rng)
    packet_loss = base_packet_loss(n, rng)
    sinr        = base_sinr(n, rng)
    users       = base_users(n, rng)

    if realistic and REALISM["nonstationarity"]:
        latency, throughput, users = _apply_nonstationarity(n, real_rng, latency, throughput, users)

    labels = np.zeros(n, dtype=int)

    # Anomaly EVENTS scale with timeline length (see EXPERIMENTS.md). Scaling
    # is relative to the 60-day default so that dataset is byte-identical to
    # before, and the RNG draw order is unchanged.
    reference_periods = 60 * 24 * 4
    num_events = int(round(int(rng.integers(6, 10)) * n / reference_periods))
    # Spread anomalies across full timeline so test set (last 20%) also gets some
    segment_size = (n - 200) // num_events
    chosen = []
    for i in range(num_events):
        seg_start = 100 + i * segment_size
        seg_end   = seg_start + segment_size - 50
        if seg_end > seg_start + 10:
            chosen.append(int(rng.integers(seg_start, seg_end)))

    fault_events = []
    for start in chosen:
        anomaly_type = rng.choice(["latency", "throughput", "packet_loss"])
        if anomaly_type == "latency":
            dur = int(rng.integers(4, 12))
        elif anomaly_type == "throughput":
            dur = int(rng.integers(12, 32))
        else:
            dur = int(rng.integers(3, 8))

        severity = 1.0
        if realistic and REALISM["severity_and_ramp"]:
            severity = _inject_fault_realistic(str(anomaly_type), start, dur, latency, throughput, packet_loss, real_rng)
        elif anomaly_type == "latency":
            latency = inject_latency_spike(latency, start, dur)
        elif anomaly_type == "throughput":
            throughput = inject_throughput_drop(throughput, start, dur)
        else:
            packet_loss = inject_packet_loss_burst(packet_loss, start, dur)

        labels[start:start+dur] = 1
        fault_events.append({"start": start, "dur": dur, "kind": "fault", "type": str(anomaly_type),
                             "severity": severity, "labeled": True, "label_start": start, "label_end": start + dur})

    benign_events = []
    if realistic and REALISM["benign_surges"]:
        users = users.astype(float)
        benign_events = _inject_benign_surges(n, len(fault_events), real_rng, latency, throughput, packet_loss, users)
    if realistic and REALISM["label_noise"]:
        labels = _noisy_labels(n, fault_events, real_rng)
    if realistic:
        users = np.clip(np.round(users), 1, 400).astype(int)

    events = []
    for ev in fault_events + benign_events:
        end = min(ev["start"] + ev["dur"], n - 1)
        events.append({
            "cell_id": cell_id, "kind": ev["kind"], "type": ev["type"], "severity": ev["severity"],
            "start_ts": timestamps[ev["start"]], "end_ts": timestamps[end], "labeled": ev["labeled"],
            "label_start_ts": timestamps[ev["label_start"]] if ev["label_start"] is not None else pd.NaT,
            "label_end_ts": timestamps[min(ev["label_end"], n - 1)] if ev["label_end"] is not None else pd.NaT,
        })

    df = pd.DataFrame({
        "timestamp":        timestamps,
        "cell_id":          cell_id,
        "latency_ms":       np.round(latency, 2),
        "throughput_mbps":  np.round(throughput, 2),
        "packet_loss_pct":  np.round(np.clip(packet_loss, 0, 100), 3),
        "sinr_db":          np.round(sinr, 2),
        "connected_users":  users,
        "anomaly":          labels,
    })
    return df, events


def build_dataset_and_events():
    """Build the full synthetic dataset (and its event sidecar) in memory.
    Fully deterministic given params.yaml's seed — across processes,
    machines, AND repeated calls within one process
    (tests/test_generate_data.py enforces all of this).

    Two sources of hidden nondeterminism this deliberately avoids:
      1. hash(cell): Python randomizes str hash() per process
         (PYTHONHASHSEED), so seeding from it silently made every fresh
         run produce a DIFFERENT dataset despite the pinned seed. Caught
         by rerunning this script and getting different anomaly counts
         each time. enumerate()'s index is a plain int — stable.
      2. The module-level `rng` used by the legacy anomaly injectors is
         shared mutable state; without reseeding here, a second call in
         the same process would continue the first call's random stream.
    """
    global rng
    rng = np.random.default_rng(SEED)

    frames, events = [], []
    for cell_index, cell in enumerate(CELLS):
        cell_rng = np.random.default_rng(SEED + cell_index)
        df_cell, ev_cell = generate_cell_data(cell, PERIODS, cell_rng, cell_index)
        frames.append(df_cell)
        events.extend(ev_cell)

    df = pd.concat(frames, ignore_index=True).sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    return df, pd.DataFrame(events)


def build_dataset() -> pd.DataFrame:
    return build_dataset_and_events()[0]


def main():
    df, events = build_dataset_and_events()

    out = Path(__file__).parent / "raw_kpis.csv"
    df.to_csv(out, index=False)
    events.to_csv(Path(__file__).parent / "events.csv", index=False)

    total     = len(df)
    anomalies = df["anomaly"].sum()
    print(f"Profile: {PROFILE}")
    print(f"Generated {total:,} rows across {len(CELLS)} cells")
    print(f"Anomaly rate: {anomalies}/{total} = {anomalies/total:.2%}")
    print(f"Events: {int((events.kind == 'fault').sum())} faults "
          f"({int(((events.kind == 'fault') & ~events.labeled).sum())} unlabeled), "
          f"{int((events.kind == 'benign').sum())} benign surges")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
