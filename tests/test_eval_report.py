import json

from scripts.eval_report import build, load, parse_load_log


def test_report_merges_status_tools_and_logs(tmp_path):
    job = tmp_path / "core"
    (job / "logs").mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"api": {"ok": True, "seconds": 5}, "docker": {"ok": False, "seconds": 9}}))
    (job / "docker.json").write_text(json.dumps({"image_mib": 812.5, "layers": 11}))
    (job / "logs" / "load.log").write_text(
        "idle RSS 175 MiB\n1000 requests in 40.0s = 25 req/s; CPU 41.0s = 41.0 ms per request\n"
        "latency p50 90 ms, p95 190 ms; peak RSS 181 MiB\nredis used_memory 4.9 MiB with 1200 inference-log rows\n")
    md = build(load(tmp_path))
    assert "| core | docker | NO | 9 s |" in md
    assert "812.5" in md and "p95_ms | 190" in md and "peak_rss_mib | 181" in md


def test_parse_load_log_ignores_missing_fields():
    assert parse_load_log("nothing useful") == {}
