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


def test_report_compares_original_and_optimised_api_paths(tmp_path):
    job = tmp_path / "core"
    (job / "logs").mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"load": {"ok": True, "seconds": 1}}))
    log = ("idle RSS 175 MiB\n1000 requests in 40.0s = {rps} req/s; CPU 41.0s = {cpu} ms per request\n"
           "latency p50 90 ms, p95 190 ms; peak RSS 181 MiB\n")
    (job / "logs" / "load_baseline.log").write_text(log.format(rps=25, cpu="41.0"))
    (job / "logs" / "load.log").write_text(log.format(rps=300, cpu="3.1"))
    (job / "stages.json").write_text(json.dumps({"features_pandas_ms": 18.2, "features_numpy_ms": 0.33}))
    (job / "compact.json").write_text(json.dumps({"target": 0.945, "rows": {"depth4_trees150": {
        "trees_used": 150, "recall": 0.95, "precision": 0.8, "f1": 0.87, "model_json_kib": 300, "predict_one_ms_median": 0.2}}}))
    md = build(load(tmp_path))
    assert "Optimisation: whole API under the same load" in md and "| original |" in md and "| optimised |" in md
    assert "features_numpy_ms" in md and "depth4_trees150" in md
