"""Merge the per-job evaluation output (eval/<job>/status.json + <tool>.json + logs) into one markdown report.
usage: python scripts/eval_report.py <root-dir-containing-one-folder-per-job> <out.md>"""
import json
import re
import sys
from pathlib import Path


def load(root: Path) -> dict:
    data = {}
    for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        job = {"status": {}, "tools": {}, "logs": {}}
        if (job_dir / "status.json").exists():
            job["status"] = json.loads((job_dir / "status.json").read_text())
        for f in job_dir.glob("*.json"):
            if f.name != "status.json":
                job["tools"][f.stem] = json.loads(f.read_text())
        for f in (job_dir / "logs").glob("*.log") if (job_dir / "logs").exists() else []:
            job["logs"][f.stem] = f.read_text()
        data[job_dir.name] = job
    return data


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def parse_load_log(text: str) -> dict:
    pats = {"idle_rss_mib": r"idle RSS (\d+)", "peak_rss_mib": r"peak RSS (\d+) MiB",
            "req_per_s": r"= (\d+) req/s", "cpu_ms_per_request": r"([\d.]+) ms per request",
            "p50_ms": r"p50 (\d+) ms", "p95_ms": r"p95 (\d+) ms", "redis_mib": r"used_memory ([\d.]+) MiB"}
    return {k: m.group(1) for k, p in pats.items() if (m := re.search(p, text))}


def build(data: dict) -> str:
    md = ["# Full evaluation of every tool (measured in CI on real libraries)", ""]
    steps = [(job, name, s) for job, d in data.items() for name, s in d["status"].items()]
    md += ["## Step status", "", table([(j, n, s["ok"], f"{s['seconds']} s") for j, n, s in steps],
                                      ["job", "step", "passed", "time"]), ""]
    for job, d in data.items():
        t = d["tools"]
        if "events" in t:
            e = t["events"]
            md += ["## Model on the held-out window (real XGBoost)", "",
                   table([("interval recall", e["interval_level"]["recall"]), ("interval precision", e["interval_level"]["precision"]),
                          ("interval F1", e["interval_level"]["f1"]), ("PR-AUC", e["pr_auc"]), ("ROC-AUC", e["roc_auc"]),
                          ("incident detection rate", e["incident_level"].get("detection_rate")),
                          ("median detection delay (intervals)", e["incident_level"].get("median_delay_intervals")),
                          ("false alerts / cell / day", e["incident_level"].get("false_alerts_per_cell_day")),
                          ("threshold", e["threshold"])], ["metric", "value"]), ""]
        if "curve" in t:
            c = t["curve"]
            md += ["## Precision/recall trade-off (threshold tuned on validation, applied to test)", "",
                   table([(k, v["threshold"], f"{v['val']['recall']} / {v['val']['precision']}", f"{v['test']['recall']} / {v['test']['precision']}",
                           v["test"]["f1"], v["event_detection_rate"], v["false_alerts_per_cell_day"])
                          for k, v in c["by_validation_target"].items()],
                         ["recall target", "threshold", "val recall / precision", "test recall / precision", "test F1",
                          "incidents detected", "false alerts/cell/day"]), "",
                   "Best precision any threshold could reach on test at a given recall: "
                   + ", ".join(f"recall >= {k}: {v}" for k, v in c["test_best_possible_precision_at_recall"].items()), ""]
        if "spread" in t:
            sp = t["spread"]
            md += [f"## Real XGBoost across {len(sp['seeds'])} independent datasets (recall target {sp['target']})", "",
                   table([(k, v["min"], v["median"], v["max"]) for k, v in sp["summary"].items()], ["metric", "min", "median", "max"]), "",
                   table([(sd, r["recall"], r["precision"], r["f1"]) for sd, r in sp["per_seed"].items()], ["seed", "recall", "precision", "F1"]), ""]
        if "api" in t:
            a = t["api"]
            md += ["## FastAPI", "", table([(k, v) for k, v in a["checks"].items()], ["check", "passed"]), ""]
            p = a["parity"]
            md += [table([("held-out rows scored through the API", p["rows_scored_by_api"]), ("max |API score - offline score|", p["max_abs_score_diff"]),
                          ("decision agreement", p["decision_agreement"]), ("API recall / precision / F1",
                           f"{p['api_metrics']['recall']} / {p['api_metrics']['precision']} / {p['api_metrics']['f1']}"),
                          ("batch throughput (rows/s, 500 per call)", p["rows_per_second"]),
                          ("501-snapshot batch status", a["batch_501_status"])], ["measure", "value"]), ""]
        if "load" in d["logs"]:
            md += ["## API under load (1 process, Redis-backed)", "", table(parse_load_log(d["logs"]["load"]).items(), ["measure", "value"]), ""]
        loads = {n: parse_load_log(d["logs"][k]) for n, k in (("original", "load_baseline"), ("optimised", "load"), ("optimised + ONNX scorer", "load_onnx"))
                 if k in d["logs"]}
        if len(loads) > 1:
            cols = ["idle_rss_mib", "peak_rss_mib", "req_per_s", "cpu_ms_per_request", "p50_ms", "p95_ms"]
            md += ["## Optimisation: whole API under the same load", "", table([(n, *(v.get(c, "") for c in cols)) for n, v in loads.items()],
                                                                               ["API path", *cols]), ""]
        if "stages" in t:
            md += ["## Optimisation: cost of each stage of one request (ms)", "", table(sorted(t["stages"].items()), ["stage", "ms"]), ""]
        if "compact" in t:
            c = t["compact"]
            md += ["## Smaller models (same data, threshold tuned on validation to the same recall target)", "", table(
                [(k, v["trees_used"], v["recall"], v["precision"], v["f1"], v["model_json_kib"], v["predict_one_ms_median"]) for k, v in c["rows"].items()],
                ["model", "trees used", "test recall", "test precision", "test F1", "model.json KiB", "predict one row ms"]), ""]
        if "onnx" in t:
            o = t["onnx"]
            md += ["## ONNX Runtime vs XGBoost", "", table([
                ("max |prob diff|", o["max_abs_prob_diff"]), ("decision agreement", o["decision_agreement"]),
                ("model.json / model.onnx (KiB)", f"{o['model_json_kib']} / {o['model_onnx_kib']}"),
                ("single row median ms (xgboost / onnx)", f"{o['single_row_latency']['xgboost']['median_ms']} / {o['single_row_latency']['onnxruntime']['median_ms']}"),
                ("500-row batch median ms (xgboost / onnx)", f"{o['batch_500_latency']['xgboost']['median_ms']} / {o['batch_500_latency']['onnxruntime']['median_ms']}"),
                ("onnx outputs", ", ".join(o["onnx_outputs"]))], ["measure", "value"]), ""]
        if "mlflow" in t:
            m = t["mlflow"]
            md += ["## MLflow", "", table([(k, json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in m.items()], ["item", "value"]), ""]
        if "dvc" in t:
            md += ["## DVC", ""]
            md += [table([(k, v.get("exit"), v.get("seconds", ""), " ".join(v.get("stages", []))) for k, v in t["dvc"].items()],
                         ["run", "exit", "seconds", "stages"]), ""]
        if "docker" in t:
            md += ["## Docker image", "", table(t["docker"].items(), ["measure", "value"]), ""]
        if "drift_demo" in t:
            md += ["## Evidently drift detection", "", table([(k, v["dataset_drift"], f"{v['drifted_columns']}/{v['columns']}", v["drift_share"])
                                                              for k, v in t["drift_demo"]["scenarios"].items()],
                                                             ["scenario", "dataset drift", "features flagged", "share"]), ""]
        if "k8s_hpa" in t:
            k = t["k8s_hpa"]
            lt = k.get("loadtest") or {}
            md += ["## Kubernetes: HPA scale-out / scale-in (kind, metrics-server)", "", table([
                ("HPA (min / max / CPU target)", f"{k['hpa']['min']} / {k['hpa']['max']} / {k['hpa']['target_cpu_pct']}%"),
                ("load generator (24 threads, 240 s)", f"{lt.get('rps')} req/s, p50 {lt.get('p50_ms')} ms, p95 {lt.get('p95_ms')} ms, errors {lt.get('errors')}"),
                ("seconds from load start to first scale-out decision", k.get("seconds_from_load_start_to_first_scale_decision")),
                ("max desired / max ready replicas", f"{k['max_desired_replicas']} / {k['max_ready_replicas']}"),
                ("peak average CPU utilisation (of the 500m request)", f"{k['peak_cpu_util_pct']}%"),
                ("max Pending API pods (insufficient CPU on the runner)", k["max_pending_pods"]),
                ("peak memory of one API pod (MiB)", k["peak_memory_mi_per_api_pod"]),
                ("seconds from load end to first scale-in", k.get("seconds_from_load_end_to_first_scale_in")),
                ("seconds from load end back at min replicas", k.get("seconds_from_load_end_to_back_at_min"))], ["measure", "value"]), ""]
            comps = k.get("components_cpu_m_mem_mi") or {}
            md += ["## Kubernetes: per-component footprint (peak CPU millicores / memory MiB)", "", table(
                [(n, *(f"{v[ph]['cpu_m_max']}m / {v[ph]['mem_mi_max']} Mi" for ph in ("idle", "under_load", "after_load"))) for n, v in sorted(comps.items())],
                ["component", "idle", "under load", "after load"]), ""]
            pr, dj = k.get("prometheus") or {}, k.get("drift_job") or {}
            md += [table([("Prometheus head series", pr.get("head_series")), ("samples / s", pr.get("samples_per_second")),
                          ("TSDB on disk (KiB) after the run", pr.get("disk_kib")), ("estimated disk for 15 d retention (MiB)", pr.get("estimated_disk_mib_for_15d")),
                          ("drift job result", " ".join(dj.get("result") or [])), ("drift job peak CPU / memory", f"{dj.get('cpu_m_max')}m / {dj.get('mem_mi_max')} Mi")],
                         ["measure", "value"]), ""]
        if "triton" in t:
            x = t["triton"]
            md += ["## Triton (CPU) vs XGBoost", "", table([
                ("max |prob diff|", x["parity_vs_xgboost"]["max_abs_prob_diff"]), ("decision agreement", x["parity_vs_xgboost"]["decision_agreement"]),
                ("output names", ", ".join(x["parity_vs_xgboost"]["output_names"])),
                ("HTTP single row median / p95 ms", f"{x['http_latency']['single_row']['median_ms']} / {x['http_latency']['single_row']['p95_ms']}"),
                ("HTTP 64-row batch median ms", x["http_latency"]["batch_64"]["median_ms"])], ["measure", "value"]), ""]
        for name in ("airflow_idle", "airflow_drift"):
            if name in d["logs"]:
                text = d["logs"][name]
                counts = {k: len(re.findall(p, text)) for k, p in
                          {"tasks succeeded": r"Marking task as SUCCESS", "tasks failed": r"Marking task as FAILED",
                           "tasks skipped": r"Marking task as SKIPPED|state=skipped"}.items()}
                ran = sorted(set(re.findall(r"task_id=(\w+)", text)))
                md += [f"## Airflow run: {name}", "", table(list(counts.items()) + [("tasks seen", ", ".join(ran))], ["measure", "value"]), ""]
    return "\n".join(md)


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(build(load(Path(sys.argv[1]))))
