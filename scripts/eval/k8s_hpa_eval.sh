#!/usr/bin/env bash
# Kubernetes: deploy k8s/overlays/full (API + HPA + Redis + Prometheus + Pushgateway + Grafana) on kind with
# metrics-server, drive sustained load from inside the cluster, then let it go idle. Records how the HPA scales
# out AND in, and what every component actually uses (idle / under load / after), plus the drift job's footprint.
# Needs: a kind cluster, image $IMAGE already loaded into it.
set -u
EVAL_DIR=${EVAL_DIR:-eval/k8s}
IMAGE=${IMAGE:-eval-api:ci}
NS=telecom-anomaly
REMOTE_IMAGE=ghcr.io/mohamed-amin-ai/mlops-anomaly:latest
mkdir -p "$EVAL_DIR"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml > /dev/null
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' > /dev/null
kubectl -n kube-system rollout status deploy/metrics-server --timeout=180s || exit 1

kubectl create namespace $NS --dry-run=client -o yaml | kubectl apply -f - > /dev/null
kubectl -n $NS create secret generic grafana-admin --from-literal=admin-password=eval-only-password --dry-run=client -o yaml | kubectl apply -f - > /dev/null
kubectl kustomize k8s/overlays/full | sed "s#${REMOTE_IMAGE}#${IMAGE}#g" | kubectl apply -f - > /dev/null
for d in redis anomaly-api pushgateway prometheus grafana; do
  kubectl -n $NS rollout status deploy/$d --timeout=300s || exit 1
done
echo "idle for 90 s so metrics-server and Prometheus have data"; sleep 90
kubectl -n $NS top pods --no-headers > "$EVAL_DIR/k8s_top_idle.txt" 2>&1
kubectl -n $NS get hpa anomaly-api -o yaml > "$EVAL_DIR/k8s_hpa_manifest_live.yaml" 2>&1

kubectl -n $NS run loadgen --restart=Never --image="$IMAGE" --image-pull-policy=IfNotPresent -i -- \
  python - --url http://anomaly-api:8000 --threads 24 --seconds 240 < scripts/k8s_loadtest.py > "$EVAL_DIR/k8s_loadgen.log" 2>&1 &
LG=$!
t0=$(date +%s)
: > "$EVAL_DIR/k8s_samples.tsv"; : > "$EVAL_DIR/k8s_top_all.tsv"
for i in $(seq 1 54); do          # 54 x 15 s = 13.5 min: 240 s of load, then scale-in
  now=$(( $(date +%s) - t0 ))
  hpa=$(kubectl -n $NS get hpa anomaly-api -o jsonpath='{.status.currentReplicas} {.status.desiredReplicas} {.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null)
  ready=$(kubectl -n $NS get deploy anomaly-api -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  pending=$(kubectl -n $NS get pods -l app.kubernetes.io/name=anomaly-api --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
  top=$(kubectl -n $NS top pods -l app.kubernetes.io/name=anomaly-api --no-headers 2>/dev/null | awk '{c+=$2; m+=$3; if ($3+0>mx) mx=$3+0} END {print c, m, mx}')
  echo -e "${now}\t${hpa:-- -- --}\t${ready:-0}\t${pending}\t${top:-0 0 0}" >> "$EVAL_DIR/k8s_samples.tsv"
  kubectl -n $NS top pods --no-headers 2>/dev/null | awk -v t="$now" '{print t "\t" $1 "\t" $2 "\t" $3}' >> "$EVAL_DIR/k8s_top_all.tsv"
  if [ "$i" = 17 ]; then            # load has just ended: Prometheus footprint while it holds the busiest data
    kubectl -n $NS exec deploy/prometheus -- wget -qO- 'http://localhost:9090/api/v1/query?query=prometheus_tsdb_head_series' > "$EVAL_DIR/prom_head_series.json" 2>&1
    kubectl -n $NS exec deploy/prometheus -- wget -qO- 'http://localhost:9090/api/v1/query?query=rate(prometheus_tsdb_head_samples_appended_total[2m])' > "$EVAL_DIR/prom_samples_rate.json" 2>&1
    kubectl -n $NS exec deploy/prometheus -- du -sk /prometheus > "$EVAL_DIR/prom_disk_kb.txt" 2>&1
  fi
  sleep 15
done
wait $LG

# drift job footprint: run the CronJob once against the (now busy) shared inference log, sampling its pod
kubectl -n $NS create job drift-eval --from=cronjob/drift-check > /dev/null 2>&1
: > "$EVAL_DIR/k8s_drift_top.tsv"
jt0=$(date +%s)
for i in $(seq 1 90); do
  kubectl -n $NS top pods -l job-name=drift-eval --no-headers 2>/dev/null | awk -v t="$(( $(date +%s) - jt0 ))" '{print t "\t" $2 "\t" $3}' >> "$EVAL_DIR/k8s_drift_top.tsv"
  s=$(kubectl -n $NS get job drift-eval -o jsonpath='{.status.succeeded}' 2>/dev/null); f=$(kubectl -n $NS get job drift-eval -o jsonpath='{.status.failed}' 2>/dev/null)
  [ "${s:-0}" = 1 ] || [ "${f:-0}" -ge 1 ] && break
  sleep 2
done
echo "drift job: succeeded=${s:-0} failed=${f:-0} after $(( $(date +%s) - jt0 )) s" > "$EVAL_DIR/k8s_drift_result.txt"
kubectl -n $NS logs job/drift-eval 2>&1 | tail -n 5 >> "$EVAL_DIR/k8s_drift_result.txt"
kubectl -n $NS get events --sort-by=.lastTimestamp 2>/dev/null | grep -i -E "SuccessfulRescale|FailedScheduling|OOMKilled|Unhealthy" | tail -n 20 > "$EVAL_DIR/k8s_events.txt"

python3 - "$EVAL_DIR" <<'PY'
import json, re, sys
d = sys.argv[1]
num = lambda x: int(re.sub(r"\D", "", x)) if x and re.search(r"\d", x) else None
rows = []
for line in open(f"{d}/k8s_samples.tsv"):
    f = line.rstrip("\n").split("\t")
    cur, des, util = (f[1].split() + ["--", "--", "--"])[:3]
    cpu, mem, mx = (f[4].split() + ["0", "0", "0"])[:3]
    rows.append({"t": int(f[0]), "current": num(cur), "desired": num(des), "cpu_util_pct": num(util), "ready": num(f[2]),
                 "pending": num(f[3]), "mem_mi_max_pod": num(mx)})
lg = open(f"{d}/k8s_loadgen.log").read()
m = re.search(r"LOADTEST requests=(\d+) ok=(\d+) errors=(\d+) rps=([\d.]+) p50_ms=(\d+) p95_ms=(\d+)", lg)
first_scale = next((r["t"] for r in rows if r["desired"] and r["desired"] > 2), None)
peak_t = next((r["t"] for r in rows if r["desired"] == max(x["desired"] or 0 for x in rows)), None)
after_peak = [r for r in rows if peak_t is not None and r["t"] > peak_t]
back_to_min = next((r["t"] for r in after_peak if r["desired"] == 2 and r["current"] == 2), None)
load_end = 240
first_scale_in = next((r["t"] for r in rows if r["t"] > load_end and r["current"] is not None and peak_t is not None
                       and r["current"] < max(x["current"] or 0 for x in rows)), None)

def comp(name):
    return re.sub(r"-[a-z0-9]{5,10}(-[a-z0-9]{5})?$", "", name)
by = {}
for line in open(f"{d}/k8s_top_all.tsv"):
    t, pod, cpu, mem = line.rstrip("\n").split("\t")
    by.setdefault(comp(pod), []).append((int(t), num(cpu), num(mem)))
def stats(series, lo, hi):
    pts = [(c, mm) for t, c, mm in series if lo <= t < hi and c is not None]
    return {"cpu_m_max": max((c for c, _ in pts), default=None), "mem_mi_max": max((mm for _, mm in pts), default=None)}
comps = {name: {"idle": stats(s, 0, 15), "under_load": stats(s, 15, load_end), "after_load": stats(s, load_end, 10**6)} for name, s in by.items()}
series = json.load(open(f"{d}/prom_head_series.json")) if open(f"{d}/prom_head_series.json").read().strip().startswith("{") else {}
rate = json.load(open(f"{d}/prom_samples_rate.json")) if open(f"{d}/prom_samples_rate.json").read().strip().startswith("{") else {}
get = lambda j: float(j["data"]["result"][0]["value"][1]) if j.get("data", {}).get("result") else None
head_series, samples_per_s = get(series), get(rate)
disk_kb = num(open(f"{d}/prom_disk_kb.txt").read().split()[0]) if open(f"{d}/prom_disk_kb.txt").read().strip()[:1].isdigit() else None
drift = [l.split("\t") for l in open(f"{d}/k8s_drift_top.tsv") if l.strip()]
drift_peak = {"cpu_m_max": max((num(x[1]) or 0 for x in drift), default=None), "mem_mi_max": max((num(x[2]) or 0 for x in drift), default=None), "samples": len(drift)}
out = {
    "hpa": {"min": 2, "max": 6, "target_cpu_pct": 70},
    "loadtest": dict(zip(["requests", "ok", "errors", "rps", "p50_ms", "p95_ms"], [float(x) if "." in x else int(x) for x in m.groups()])) if m else None,
    "seconds_from_load_start_to_first_scale_decision": first_scale,
    "max_desired_replicas": max((r["desired"] or 0) for r in rows),
    "max_ready_replicas": max((r["ready"] or 0) for r in rows),
    "peak_cpu_util_pct": max((r["cpu_util_pct"] or 0) for r in rows),
    "max_pending_pods": max((r["pending"] or 0) for r in rows),
    "peak_memory_mi_per_api_pod": max((r["mem_mi_max_pod"] or 0) for r in rows),
    "seconds_from_load_end_to_first_scale_in": (first_scale_in - load_end) if first_scale_in else None,
    "seconds_from_load_end_to_back_at_min": (back_to_min - load_end) if back_to_min else None,
    "components_cpu_m_mem_mi": comps,
    "prometheus": {"head_series": head_series, "samples_per_second": samples_per_s, "disk_kib": disk_kb,
                   "estimated_disk_mib_for_15d": round(samples_per_s * 86400 * 15 * 1.5 / 2**20) if samples_per_s else None},
    "drift_job": {"result": open(f"{d}/k8s_drift_result.txt").read().strip().splitlines()[:1], **drift_peak},
    "samples": rows,
}
json.dump(out, open(f"{d}/k8s_hpa.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))
PY
