"""
k8s_traffic.py
---------------
Cluster smoke test for the shared history store. Sends raw KPI readings to the API, each over a NEW
connection, so Kubernetes spreads them across the API replicas. A cell's readings therefore arrive on
different pods; if history were per-pod the cell would never warm up.

Expectation (api/live_features.py: MIN_HISTORY_ROWS = 20): for every cell the first 19 readings
answer 425, the 20th onwards answer 200. Exits non-zero otherwise.

Standard library only, so it can be piped into any image that has python:
  kubectl -n telecom-anomaly run traffic --rm -i --restart=Never --image=<api image> -- \
      python - --url http://anomaly-api:8000 < scripts/k8s_traffic.py
"""

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

MIN_HISTORY_ROWS = 20


def make_reading(cell_id: str, i: int, rng: random.Random, start: datetime) -> dict:
    """A plausible, non-degenerate reading. packet_loss_pct is never exactly 0 and no KPI is flat,
    because either makes an engineered feature undefined (the API then answers 422, by design)."""
    return {
        "cell_id": cell_id,
        "timestamp": (start + timedelta(minutes=15 * i)).isoformat(),
        "latency_ms": rng.uniform(20, 60),
        "throughput_mbps": rng.uniform(60, 140),
        "packet_loss_pct": rng.uniform(0.2, 1.5),
        "sinr_db": rng.uniform(8, 22),
        "connected_users": rng.randint(10, 150),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Connection": "close",
                                      **({"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def expected_status(i: int) -> int:
    """i is the 0-based index of the reading within its cell."""
    return 425 if i < MIN_HISTORY_ROWS - 1 else 200


def run(url: str, cells: int, readings: int, seed: int) -> int:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seen, mismatches = Counter(), []
    for i in range(readings):
        for c in range(cells):
            cell = f"SMOKE_{c:02d}"
            status = post(f"{url.rstrip('/')}/predict", make_reading(cell, i, rng, start))
            seen[status] += 1
            if status != expected_status(i):
                mismatches.append((cell, i, status))
    print(f"statuses: {dict(seen)}")
    if mismatches:
        print(f"UNEXPECTED ({len(mismatches)}), first few (cell, reading index, status): {mismatches[:5]}")
        return 1
    print("OK: every cell warmed up after exactly 20 readings although they were spread across replicas")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--readings", type=int, default=32, help="per cell; must be > 19 to see a 200")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(run(args.url, args.cells, args.readings, args.seed))
