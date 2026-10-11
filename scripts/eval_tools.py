"""
Measured evaluation of each tool in the project, run against the REAL libraries in CI
(.github/workflows/evaluation.yml). Every subcommand writes one JSON file into $EVAL_DIR
(default eval/core) and never asserts on the numbers: the point is to record what is true.

  python scripts/eval_tools.py api      # FastAPI: endpoints, serving-vs-offline parity, auth
  python scripts/eval_tools.py onnx     # ONNX Runtime vs XGBoost: parity, latency, size
  python scripts/eval_tools.py mlflow   # MLflow: runs, registry, promotion gate, load-back
  python scripts/eval_tools.py events   # incident-level alerting metrics on the held-out window
  python scripts/eval_tools.py triton   # Triton (CPU) vs XGBoost: parity + HTTP latency
"""

import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

EVAL_DIR = Path(os.environ.get("EVAL_DIR", "eval/core"))
PORT = 18020


def save(name: str, payload: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str)[:3000])


def load_artifacts():
    import joblib
    import xgboost as xgb

    from data.features import get_feature_columns
    from models.common import MODEL_DIR, load_data, load_params, time_split

    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    df = load_data()
    cols = meta["feature_cols"] if "feature_cols" in meta else get_feature_columns(df)
    _, test_df = time_split(df, load_params()["split"]["train_ratio"])
    return meta, model, scaler, cols, test_df.reset_index(drop=True)


def confusion_scores(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return {"recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# ── HTTP helpers ────────────────────────────────────────────────────────────────────────────

def http(method: str, path: str, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")


class Server:
    def __init__(self, env: dict | None = None):
        self.env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH", "")])), **(env or {})}
        self.env.pop("REDIS_URL", None)
        if env and "API_KEYS" not in env:
            self.env.pop("API_KEYS", None)

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--log-level", "warning"],
            cwd=REPO, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                return self
            except Exception:
                time.sleep(1)
        self.proc.kill()
        raise RuntimeError("API did not become healthy within 60 s")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def snapshot(row, cols) -> dict:
    return {"cell_id": str(row["cell_id"]),
            "latency_ms": float(row["latency_ms"]), "throughput_mbps": float(row["throughput_mbps"]),
            "packet_loss_pct": float(row["packet_loss_pct"]), "sinr_db": float(row["sinr_db"]),
            "connected_users": int(row["connected_users"]),
            "features": {c: float(row[c]) for c in cols}}


# ── api ─────────────────────────────────────────────────────────────────────────────────────

def cmd_api():
    from scripts.k8s_traffic import make_reading

    meta, model, scaler, cols, test_df = load_artifacts()
    checks, out = {}, {}

    with Server():
        checks["GET /health -> 200"] = http("GET", "/health")[0] == 200
        code, info = http("GET", "/model/info")
        checks["/model/info threshold == metadata threshold"] = (
            code == 200 and abs(info.get("threshold", -1) - meta["threshold"]) < 1e-9)

        rng, start = random.Random(3), datetime(2026, 1, 1, tzinfo=timezone.utc)
        first = http("POST", "/predict", make_reading("EVAL_COLD", 0, rng, start))[0]
        checks["first raw reading for a new cell -> 425 (no guessing)"] = first == 425
        last = None
        for i in range(1, 25):
            last = http("POST", "/predict", make_reading("EVAL_COLD", i, rng, start))
        checks["after warm-up raw readings -> 200 with live features"] = (
            last[0] == 200 and last[1].get("feature_source") == "live")
        bad = http("POST", "/predict", {**make_reading("EVAL_BAD", 0, rng, start), "latency_ms": -5})
        checks["invalid reading (negative latency) -> 422"] = bad[0] == 422

        # Serving-vs-offline parity on the ENTIRE held-out window, through /predict/batch.
        n = len(test_df)
        probs, t0 = np.empty(n), time.time()
        statuses = set()
        for lo in range(0, n, 500):
            chunk = test_df.iloc[lo:lo + 500]
            code, res = http("POST", "/predict/batch", {"snapshots": [snapshot(r, cols) for _, r in chunk.iterrows()]},
                             timeout=120)
            statuses.add(code)
            if code != 200:
                break
            for j, item in enumerate(res["results"]):
                probs[lo + j] = item["result"]["anomaly_score"] if item["ok"] else np.nan
        wall = time.time() - t0
        scored = ~np.isnan(probs)
        offline = model.predict_proba(scaler.transform(test_df[cols].to_numpy(dtype=float)))[:, 1]
        thr = meta["threshold"]
        y = test_df["anomaly"].to_numpy()
        api_scores = confusion_scores(y[scored], probs[scored] >= thr)
        off_scores = confusion_scores(y, offline >= thr)
        out["parity"] = {
            "rows": int(n), "rows_scored_by_api": int(scored.sum()), "batch_http_statuses": sorted(statuses),
            "max_abs_score_diff": float(np.nanmax(np.abs(probs - offline))) if scored.any() else None,
            "decision_agreement": float(((probs >= thr) == (offline >= thr))[scored].mean()) if scored.any() else None,
            "api_metrics": api_scores, "offline_metrics": off_scores, "metadata_metrics": meta["metrics"],
            "seconds_for_all_rows": round(wall, 1), "rows_per_second": round(n / wall, 0),
        }
        checks["API scores every held-out row"] = int(scored.sum()) == n
        checks["API scores match offline model (max diff < 1e-3)"] = bool(
            scored.any() and np.nanmax(np.abs(probs - offline)) < 1e-3)

        code, over = http("POST", "/predict/batch", {"snapshots": [snapshot(r, cols) for _, r in test_df.head(501).iterrows()]})
        out["batch_501_status"] = code
        mcode, mtext = http("GET", "/metrics")
        checks["GET /metrics -> 200 (Prometheus text)"] = mcode == 200 and isinstance(mtext, str) and "# HELP" in mtext
        out["metric_families_exposed"] = sorted({line.split()[2] for line in mtext.splitlines()
                                                 if line.startswith("# TYPE") and len(line.split()) > 2})[:60] \
            if isinstance(mtext, str) else []

    row = test_df.iloc[0]
    with Server({"API_KEYS": "key-one,key-two"}):
        body = snapshot(row, cols)
        checks["auth: no key -> 401"] = http("POST", "/predict", body)[0] == 401
        checks["auth: wrong key -> 401"] = http("POST", "/predict", body, {"X-API-Key": "nope"})[0] == 401
        checks["auth: first key -> 200"] = http("POST", "/predict", body, {"X-API-Key": "key-one"})[0] == 200
        checks["auth: second (rotation) key -> 200"] = http("POST", "/predict", body, {"X-API-Key": "key-two"})[0] == 200
        checks["auth: /health stays open"] = http("GET", "/health")[0] == 200
        checks["auth: /metrics stays open"] = http("GET", "/metrics")[0] == 200

    out["checks"] = checks
    save("api", out)


# ── onnx ────────────────────────────────────────────────────────────────────────────────────

def cmd_onnx():
    import onnxruntime as ort

    meta, model, scaler, cols, test_df = load_artifacts()
    onnx_path = REPO / "models" / "model.onnx"
    sample = test_df.sample(n=min(5000, len(test_df)), random_state=1)
    X = scaler.transform(sample[cols].to_numpy(dtype=float)).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: X})
    prob_out = next(o for o in outs if getattr(o, "ndim", 0) == 2 and o.shape == (len(X), 2))
    ref = model.predict_proba(X)[:, 1]
    thr = meta["threshold"]
    res = {
        "rows_compared": int(len(X)),
        "max_abs_prob_diff": float(np.abs(prob_out[:, 1] - ref).max()),
        "decision_agreement": float(((prob_out[:, 1] >= thr) == (ref >= thr)).mean()),
        "model_json_kib": round((REPO / "models" / "model.json").stat().st_size / 1024, 1),
        "model_onnx_kib": round(onnx_path.stat().st_size / 1024, 1),
        "onnx_outputs": [o.name for o in sess.get_outputs()],
    }

    def timed(fn, reps):
        t = []
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            t.append((time.perf_counter() - s) * 1000)
        t.sort()
        return {"median_ms": round(t[len(t) // 2], 3), "p95_ms": round(t[int(len(t) * .95)], 3)}

    one = X[:1]
    res["single_row_latency"] = {
        "xgboost": timed(lambda: model.predict_proba(one), 300),
        "onnxruntime": timed(lambda: sess.run(None, {in_name: one}), 300),
    }
    big = X[:500]
    res["batch_500_latency"] = {
        "xgboost": timed(lambda: model.predict_proba(big), 40),
        "onnxruntime": timed(lambda: sess.run(None, {in_name: big}), 40),
    }
    save("onnx", res)


# ── mlflow ──────────────────────────────────────────────────────────────────────────────────

def cmd_mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient

    from models.common import load_params
    from models.promote import promote_latest_model

    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    out = {"tracking_uri": mlflow.get_tracking_uri()}

    exps = [e for e in client.search_experiments()]
    out["experiments"] = {e.name: len(client.search_runs([e.experiment_id])) for e in exps}

    exp = client.get_experiment_by_name(params["mlflow"]["experiment_name"])
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"]) if exp else []
    out["training_runs"] = len(runs)
    if runs:
        r = runs[0]
        out["latest_run"] = {"id": r.info.run_id, "status": r.info.status,
                             "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                             "params_logged": len(r.data.params),
                             "artifacts": sorted(a.path for a in client.list_artifacts(r.info.run_id))}

    versions = client.search_model_versions(f"name='{name}'")
    out["registered_versions"] = sorted(int(v.version) for v in versions)

    # Registry round-trip: load the newest registered version and compare with models/model.json.
    if versions:
        import xgboost as xgb
        newest = max(versions, key=lambda v: int(v.version))
        loaded = mlflow.xgboost.load_model(f"models:/{name}/{newest.version}")
        local = xgb.XGBClassifier()
        local.load_model(REPO / "models" / "model.json")
        X = np.random.default_rng(0).normal(size=(50, len(local.feature_names_in_ if hasattr(local, "feature_names_in_") else range(68))))
        try:
            diff = float(np.abs(loaded.predict_proba(X)[:, 1] - local.predict_proba(X)[:, 1]).max())
        except Exception as e:  # feature-name mismatch etc. is a finding, not a crash
            diff = f"error: {type(e).__name__}: {e}"
        out["registry_roundtrip_max_prob_diff"] = diff

    out["promotion_gate"] = promote_latest_model(name, params["threshold"]["min_recall"])
    prod = client.get_latest_versions(name, stages=["Production"])
    out["production_versions_after_gate"] = [int(v.version) for v in prod]
    save("mlflow", out)


# ── events ──────────────────────────────────────────────────────────────────────────────────

def cmd_events():
    from models.alerting import grouped_event_metrics

    meta, model, scaler, cols, test_df = load_artifacts()
    prob = model.predict_proba(scaler.transform(test_df[cols].to_numpy(dtype=float)))[:, 1]
    pred = prob >= meta["threshold"]
    y = test_df["anomaly"].to_numpy().astype(bool)
    import data.generate_data as gen

    ev = grouped_event_metrics(y, pred, test_df["cell_id"].to_numpy(), intervals_per_day=gen.INTERVALS_PER_DAY)
    from sklearn.metrics import average_precision_score, roc_auc_score
    imp = sorted(zip(cols, getattr(model, "feature_importances_", [0.0] * len(cols))), key=lambda t: -t[1])[:8]
    save("events", {
        "interval_level": confusion_scores(y, pred),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "incident_level": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()},
        "threshold": meta["threshold"],
        "positive_rate_test": round(float(y.mean()), 4),
        "top_features": [[c, round(float(v), 4)] for c, v in imp],
    })


# ── triton ──────────────────────────────────────────────────────────────────────────────────

def cmd_triton():
    base = os.environ.get("TRITON_URL", "http://127.0.0.1:8003")
    name = "telecom_anomaly_xgb"
    meta, model, scaler, cols, test_df = load_artifacts()

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}

    out = {"model_metadata": call("GET", f"/v2/models/{name}")}
    sample = test_df.sample(n=min(2000, len(test_df)), random_state=2)
    X = scaler.transform(sample[cols].to_numpy(dtype=float)).astype(np.float32)

    def infer(batch):
        res = call("POST", f"/v2/models/{name}/infer", {"inputs": [
            {"name": "input", "shape": list(batch.shape), "datatype": "FP32", "data": batch.flatten().tolist()}]})
        o = next(o for o in res["outputs"] if o["shape"] == [len(batch), 2])
        return np.asarray(o["data"], dtype=float).reshape(len(batch), 2), [x["name"] for x in res["outputs"]]

    probs, names = [], []
    for lo in range(0, len(X), 64):   # config.pbtxt max_batch_size is 64
        p, names = infer(X[lo:lo + 64])
        probs.append(p[:, 1])
    probs = np.concatenate(probs)
    ref = model.predict_proba(X)[:, 1]
    thr = meta["threshold"]
    out["parity_vs_xgboost"] = {"rows": int(len(X)), "output_names": names,
                                "max_abs_prob_diff": float(np.abs(probs - ref).max()),
                                "decision_agreement": float(((probs >= thr) == (ref >= thr)).mean())}

    def timed(fn, reps):
        t = []
        for _ in range(reps):
            s0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - s0) * 1000)
        t.sort()
        return {"median_ms": round(t[len(t) // 2], 2), "p95_ms": round(t[int(len(t) * .95)], 2)}

    out["http_latency"] = {"single_row": timed(lambda: infer(X[:1]), 200), "batch_64": timed(lambda: infer(X[:64]), 30)}
    metrics = urllib.request.urlopen(base.replace(":8003", ":8005") + "/metrics", timeout=10).read().decode()
    out["triton_metrics_lines"] = len(metrics.splitlines())
    save("triton", out)


# ── curve ───────────────────────────────────────────────────────────────────────────────────

def cmd_curve():
    """Precision/recall trade-off of the REAL model. Thresholds are chosen on validation only."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    import data.generate_data as gen
    from models.alerting import grouped_event_metrics
    from models.common import load_data, load_params, three_way_time_split, tune_threshold

    meta, model, scaler, cols, _ = load_artifacts()
    params = load_params()
    df = load_data()
    _, va, te = three_way_time_split(df, params["split"]["train_ratio"], params["split"]["val_fraction"])
    pv = model.predict_proba(scaler.transform(va[cols].to_numpy(dtype=float)))[:, 1]
    pt = model.predict_proba(scaler.transform(te[cols].to_numpy(dtype=float)))[:, 1]
    yv, yt = va["anomaly"].to_numpy(), te["anomaly"].to_numpy()

    rows = {}
    for target in (0.90, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.98):
        thr, *_ = tune_threshold(yv, pv, target)
        tv, tt = confusion_scores(yv, pv >= thr), confusion_scores(yt, pt >= thr)
        ev = grouped_event_metrics(yt, pt >= thr, te["cell_id"].to_numpy(), intervals_per_day=gen.INTERVALS_PER_DAY)
        rows[str(target)] = {"threshold": round(thr, 5), "val": {k: tv[k] for k in ("recall", "precision", "f1")},
                             "test": {k: tt[k] for k in ("recall", "precision", "f1")},
                             "event_detection_rate": round(float(ev["detection_rate"]), 4),
                             "false_alerts_per_cell_day": round(float(ev["false_alerts_per_cell_day"]), 3)}
    pr_p, pr_r, _ = precision_recall_curve(yt, pt)
    oracle = {str(r): round(float(pr_p[:-1][pr_r[:-1] >= r].max()), 4) for r in (0.80, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)
              if (pr_r[:-1] >= r).any()}
    save("curve", {"by_validation_target": rows, "test_best_possible_precision_at_recall": oracle,
                   "val_pr_auc": round(float(average_precision_score(yv, pv)), 4),
                   "test_pr_auc": round(float(average_precision_score(yt, pt)), 4),
                   "positive_rate": {"val": round(float(yv.mean()), 4), "test": round(float(yt.mean()), 4)}})


# ── spread ──────────────────────────────────────────────────────────────────────────────────

def cmd_spread():
    """Real XGBoost on 10 independently generated datasets, at the configured recall target."""
    from scripts.study_common import apply_level, configured_level, evaluate_current_dataset, spread
    from models.common import load_params

    params = load_params()
    target = params["threshold"]["validation_recall_target"]
    rows = {}
    for seed in (42, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        apply_level(configured_level(), days=params["data"]["periods_days"], seed=seed)
        r = evaluate_current_dataset(params, [target], backend="xgboost")
        rows[str(seed)] = {"pr_auc": r["test_pr_auc"], **r["by_target"][str(target)]}
        print(seed, rows[str(seed)]["recall"], rows[str(seed)]["precision"])
    keys = ("recall", "precision", "f1", "event_detection_rate", "false_alerts_per_cell_day", "pr_auc")
    save("spread", {"target": target, "seeds": list(rows), "per_seed": rows,
                    "summary": {k: spread([rows[s][k] for s in rows]) for k in keys}})


if __name__ == "__main__":
    cmds = {"api": cmd_api, "onnx": cmd_onnx, "mlflow": cmd_mlflow, "events": cmd_events, "triton": cmd_triton, "curve": cmd_curve, "spread": cmd_spread}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
