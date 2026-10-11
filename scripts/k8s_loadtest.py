"""
Closed-loop load generator that runs INSIDE the cluster (standard library only):
  kubectl -n telecom-anomaly run loadgen --restart=Never --image=<api image> -- \
      python - --url http://anomaly-api:8000 --threads 24 --seconds 240 < scripts/k8s_loadtest.py
Each thread owns one cell, so after the 20-reading cold start every request is a scored 200.
Prints one summary line:  LOADTEST requests=<n> ok=<n> errors=<n> rps=<x> p50_ms=<x> p95_ms=<x>
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone


def reading(cell, i, rng, start):
    return {"cell_id": cell, "timestamp": (start + timedelta(minutes=15 * i)).isoformat(),
            "latency_ms": rng.uniform(20, 60), "throughput_mbps": rng.uniform(60, 140),
            "packet_loss_pct": rng.uniform(0.2, 1.5), "sinr_db": rng.uniform(8, 22),
            "connected_users": rng.randint(10, 150)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://anomaly-api:8000")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=240)
    a = ap.parse_args()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    stop = time.time() + a.seconds
    lock, lat, counts = threading.Lock(), [], {"ok": 0, "err": 0}

    def worker(n):
        rng, start, i = random.Random(n), datetime(2026, 1, 1, tzinfo=timezone.utc), 0
        while time.time() < stop:
            body = json.dumps(reading(f"LOAD_{n}", i, rng, start)).encode()
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(urllib.request.Request(a.url + "/predict", body, headers), timeout=30) as r:
                    r.read()
                    ok = r.status == 200
            except urllib.error.HTTPError as e:
                ok = e.code == 425 and i < 20      # expected cold start only
            except Exception:
                ok = False
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                counts["ok" if ok else "err"] += 1
                lat.append(dt)
            i += 1

    ts = [threading.Thread(target=worker, args=(n,)) for n in range(a.threads)]
    t_start = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t_start
    lat.sort()
    n = len(lat)
    print(f"LOADTEST requests={n} ok={counts['ok']} errors={counts['err']} rps={n / wall:.1f} "
          f"p50_ms={lat[n // 2]:.0f} p95_ms={lat[int(n * .95)]:.0f}")


if __name__ == "__main__":
    main()
