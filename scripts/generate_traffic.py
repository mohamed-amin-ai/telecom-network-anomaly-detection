"""
generate_traffic.py
--------------------
Hits a running instance of the API with a mix of realistic and
deliberately-anomalous KPI snapshots, so Prometheus/Grafana have real
traffic to visualize instead of an empty dashboard.

Not part of the pipeline or the test suite — a manual demo/dev tool.
Requires the API to actually be running (`docker compose up` or
`uvicorn api.main:app`).

Usage:
  python scripts/generate_traffic.py --url http://localhost:8000 --cells 3 --requests 200
"""

import argparse
import random
import time

import requests

NORMAL_RANGES = {
    "latency_ms": (20, 60),
    "throughput_mbps": (60, 140),
    "packet_loss_pct": (0.0, 1.5),
    "sinr_db": (8, 22),
    "connected_users": (10, 150),
}

# Occasionally send values that should trip the anomaly detector, so the
# severity/score panels show more than a flat line of "normal".
ANOMALOUS_RANGES = {
    "latency_ms": (120, 250),
    "throughput_mbps": (5, 25),
    "packet_loss_pct": (5, 15),
    "sinr_db": (-5, 3),
    "connected_users": (10, 150),
}


def make_snapshot(cell_id: str, anomalous: bool) -> dict:
    ranges = ANOMALOUS_RANGES if anomalous else NORMAL_RANGES
    return {
        "cell_id": cell_id,
        "latency_ms": round(random.uniform(*ranges["latency_ms"]), 2),
        "throughput_mbps": round(random.uniform(*ranges["throughput_mbps"]), 2),
        "packet_loss_pct": round(random.uniform(*ranges["packet_loss_pct"]), 3),
        "sinr_db": round(random.uniform(*ranges["sinr_db"]), 2),
        "connected_users": random.randint(*[int(x) for x in ranges["connected_users"]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--cells", type=int, default=3, help="number of distinct cell_ids to simulate")
    parser.add_argument("--requests", type=int, default=200, help="total /predict calls to make")
    parser.add_argument("--anomaly-rate", type=float, default=0.08, help="fraction of requests that are anomalous")
    parser.add_argument("--sleep", type=float, default=0.05, help="seconds between requests")
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        health.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Could not reach {args.url}/health — is the API running? ({e})")
        return

    cell_ids = [f"DEMO_CELL_{i:03d}" for i in range(args.cells)]
    print(f"Sending {args.requests} requests across {len(cell_ids)} cells to {args.url} ...")

    ok = cold_start = error = 0
    for i in range(args.requests):
        cell_id = random.choice(cell_ids)
        anomalous = random.random() < args.anomaly_rate
        snapshot = make_snapshot(cell_id, anomalous)
        try:
            resp = requests.post(f"{args.url}/predict", json=snapshot, timeout=5)
            if resp.status_code == 200:
                ok += 1
            elif resp.status_code == 425:
                cold_start += 1
            else:
                error += 1
        except requests.exceptions.RequestException as e:
            error += 1
            print(f"  request {i} failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.requests} sent (ok={ok}, cold_start={cold_start}, error={error})")

        time.sleep(args.sleep)

    print(f"\nDone. ok={ok}  cold_start={cold_start}  error={error}")
    print(f"Check metrics at {args.url}/metrics, or open Grafana at http://localhost:3000 "
          f"(default login admin/admin).")


if __name__ == "__main__":
    main()
