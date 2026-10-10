"""
measure_resources.py
---------------------
Measures what one API process actually costs, so k8s resource requests/limits come from numbers.
Starts uvicorn locally against a Redis (REDIS_URL, default redis://127.0.0.1:6379/12 — flushed!),
warms 4 cells past the 20-reading cold start, then drives 4 concurrent clients through
/predict and reports RSS, CPU per scored request, throughput and latency, plus Redis memory.

  python scripts/measure_resources.py            # needs models/ artifacts and a running Redis

Measure with the REAL image's model: the numbers recorded in k8s/base/deployment.yaml came from the
scikit-learn stand-in used where XGBoost isn't installable, so they should be re-checked.
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from scripts.k8s_traffic import make_reading  # noqa: E402

PORT = 18010
URL = f"http://127.0.0.1:{PORT}/predict"


def rss_mib(pid: int) -> float:
    return int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0]) / 1024


def cpu_seconds(pid: int) -> float:
    f = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def post(cell: str, i: int, rng, start) -> tuple[int, float]:
    req = urllib.request.Request(URL, data=json.dumps(make_reading(cell, i, rng, start)).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        code = urllib.request.urlopen(req, timeout=10).status
    except urllib.error.HTTPError as e:
        code = e.code
    return code, time.time() - t


def main() -> int:
    import redis
    redis_url = os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/12")
    client = redis.Redis.from_url(redis_url)
    client.flushdb()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--log-level", "warning"], cwd=REPO)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        print(f"idle RSS {rss_mib(proc.pid):.0f} MiB")
        rng, start = random.Random(1), datetime(2026, 1, 1, tzinfo=timezone.utc)
        for k in range(4):
            for i in range(20):
                post(f"W{k}", i, rng, start)
        lat, codes = [], {}
        c0, t0 = cpu_seconds(proc.pid), time.time()

        def worker(k):
            for i in range(20, 320):
                code, d = post(f"W{k}", i, rng, start)
                codes[code] = codes.get(code, 0) + 1
                lat.append(d)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall, cpu = time.time() - t0, cpu_seconds(proc.pid) - c0
        lat.sort()
        n = len(lat)
        print(f"status counts {codes}  (all 200 expected; anything else means the numbers below are not scored requests)")
        print(f"{n} requests in {wall:.1f}s = {n / wall:.0f} req/s; CPU {cpu:.1f}s = {1000 * cpu / n:.1f} ms per request")
        print(f"latency p50 {1000 * lat[n // 2]:.0f} ms, p95 {1000 * lat[int(n * .95)]:.0f} ms; peak RSS {rss_mib(proc.pid):.0f} MiB")
        mem = int(client.info("memory")["used_memory"]) / 2**20
        print(f"redis used_memory {mem:.1f} MiB with {client.llen('anomaly:inference_log')} inference-log rows")
        return 0 if set(codes) == {200} else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
