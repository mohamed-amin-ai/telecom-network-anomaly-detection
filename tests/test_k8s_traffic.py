"""test_k8s_traffic.py — the cluster smoke script's logic, against a stub HTTP server (no cluster, no model)."""

import http.server
import json
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from api.live_features import MIN_HISTORY_ROWS as API_MIN_HISTORY_ROWS
from scripts import k8s_traffic


# cells=3 against 2 pods on purpose: with an even cell count, strict alternation would pin each cell to one pod.
def _stub_api(shared_history: bool):
    """Stub of /predict's warm-up rule. shared_history=False models per-pod history: two 'pods' alternate."""
    counts = [Counter(), Counter()]
    turn = [0]

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pod = 0 if shared_history else turn[0] % 2
            turn[0] += 1
            counts[pod][body["cell_id"]] += 1
            ok = counts[pod][body["cell_id"]] >= k8s_traffic.MIN_HISTORY_ROWS
            self.send_response(200 if ok else 425)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_smoke_script_threshold_matches_the_api():
    assert k8s_traffic.MIN_HISTORY_ROWS == API_MIN_HISTORY_ROWS


def test_passes_when_history_is_shared_and_fails_when_it_is_per_pod():
    shared = _stub_api(shared_history=True)
    per_pod = _stub_api(shared_history=False)
    try:
        assert k8s_traffic.run(f"http://127.0.0.1:{shared.server_port}", cells=3, readings=24, seed=1) == 0
        assert k8s_traffic.run(f"http://127.0.0.1:{per_pod.server_port}", cells=3, readings=24, seed=1) == 1
    finally:
        shared.shutdown()
        per_pod.shutdown()


def test_readings_are_never_degenerate():
    import random
    from datetime import datetime, timezone
    rng = random.Random(0)
    rows = [k8s_traffic.make_reading("C", i, rng, datetime(2026, 1, 1, tzinfo=timezone.utc)) for i in range(50)]
    assert all(r["packet_loss_pct"] > 0 for r in rows)
    assert len({r["latency_ms"] for r in rows}) == 50
    assert len({r["timestamp"] for r in rows}) == 50
