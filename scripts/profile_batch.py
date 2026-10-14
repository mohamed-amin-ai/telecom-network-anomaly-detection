"""
profile_batch.py
-----------------
Shows where the time goes in POST /predict/batch, so a speed-up can target the real cost instead of a guess.

Warms CELLS cells past the 20-reading cold start, then scores one batch (one new reading per cell) in
process with cProfile and prints the milliseconds per reading and the hottest functions. Run it with the
history in memory (default) and again with Redis (REDIS_URL) to see what the Redis round trips add.

  python scripts/profile_batch.py                      # needs models/ artifacts
  REDIS_URL=redis://127.0.0.1:6379/12 python scripts/profile_batch.py    # flushes that db
  CELLS=500 TOP=25 python scripts/profile_batch.py
"""
import cProfile
import os
import pstats
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from scripts.k8s_traffic import make_reading  # noqa: E402

CELLS = int(os.environ.get("CELLS", "500"))          # the endpoint accepts at most 500 per request
TOP = int(os.environ.get("TOP", "20"))
WARMUP_READINGS = 20                                  # api.live_features.MIN_HISTORY_ROWS


def main() -> int:
    if os.environ.get("REDIS_URL"):
        import redis
        redis.Redis.from_url(os.environ["REDIS_URL"]).flushdb()

    from fastapi.testclient import TestClient
    from api.main import app

    rng = random.Random(7)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cells = [f"PROFILE_CELL_{i:04d}" for i in range(CELLS)]
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

    with TestClient(app) as client:
        def batch(i: int) -> dict:
            body = {"snapshots": [make_reading(c, i, rng, start) for c in cells]}
            r = client.post("/predict/batch", json=body, headers=headers)
            r.raise_for_status()
            return r.json()

        for i in range(WARMUP_READINGS):
            batch(i)                                   # builds each cell's history

        t = time.perf_counter()
        out = batch(WARMUP_READINGS)                   # timed without the profiler
        plain = time.perf_counter() - t
        if out["scored_count"] != CELLS:
            print(f"warning: only {out['scored_count']} of {CELLS} readings were scored: {out['error_count']} errors")

        prof = cProfile.Profile()
        prof.enable()
        batch(WARMUP_READINGS + 1)
        prof.disable()

    backend = "redis" if os.environ.get("REDIS_URL") else "memory"
    print(f"history backend: {backend}")
    print(f"batch of {CELLS}: {plain * 1000:.0f} ms total, {plain * 1000 / CELLS:.2f} ms per reading (no profiler)")
    print(f"=> {CELLS / plain:.0f} readings/s in one process, {7000 / (CELLS / plain):.0f} s for 7,000 cells\n")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(TOP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
