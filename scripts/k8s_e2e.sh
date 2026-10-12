#!/usr/bin/env bash
# End-to-end check of the full Kubernetes deployment (k8s/overlays/full) on a live cluster
# (kind in CI; any cluster works). Needs: kubectl, python3, curl, and the API image already
# present in the cluster (kind load docker-image ...).
#
#   IMAGE=telecom-anomaly-api:ci scripts/k8s_e2e.sh
#
# What it proves, in order:
#   1. every workload becomes Ready (API x2, Redis, Prometheus, Pushgateway, Grafana)
#   2. history is SHARED: readings spread across both API pods still warm each cell after 20
#   3. both API pods really received traffic (otherwise (2) proves nothing)
#   4. the drift CronJob runs from the same image against the shared inference log and succeeds
#   5. Pushgateway holds its metrics; Prometheus scrapes both API pods and the drift job's metrics
#   (API-key auth is on for the run: unauthenticated /predict -> 401, /metrics open)
#   6. Grafana provisioned both dashboards and its Prometheus datasource is healthy
set -euo pipefail

NS=telecom-anomaly
IMAGE=${IMAGE:-telecom-anomaly-api:ci}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')}
REMOTE_IMAGE=ghcr.io/mohamed-amin-ai/mlops-anomaly:latest
DRIFT_IMAGE=${DRIFT_IMAGE:-telecom-anomaly-drift:ci}
REMOTE_DRIFT_IMAGE=ghcr.io/mohamed-amin-ai/mlops-anomaly-drift:latest
PF_PIDS=()
cleanup() { for p in "${PF_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

step() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*" >&2; dump; exit 1; }
dump() {
  kubectl -n "$NS" get all,pvc,cronjob,job 2>&1 || true
  kubectl -n "$NS" describe pods 2>&1 | tail -150 || true
  kubectl -n "$NS" logs -l app.kubernetes.io/name=anomaly-api --prefix --tail=60 2>&1 || true
  kubectl -n "$NS" logs -l app.kubernetes.io/name=redis --tail=30 2>&1 || true
}
# port_forward <svc> <local> <remote>
port_forward() {
  kubectl -n "$NS" port-forward "svc/$1" "$2:$3" >/dev/null 2>&1 &
  PF_PIDS+=("$!")
  for _ in $(seq 1 30); do curl -s -o /dev/null "http://127.0.0.1:$2/" && return 0; sleep 1; done
  fail "port-forward to $1 never became reachable"
}
# wait_until <description> <timeout-seconds> <command...>
wait_until() {
  local what=$1 limit=$2; shift 2
  for _ in $(seq 1 "$limit"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  fail "timed out waiting for: $what"
}

step "Deploy k8s/overlays/full with image $IMAGE"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" create secret generic grafana-admin --from-literal=admin-password="$GRAFANA_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -
# Turn API-key auth on, so steps 2-3 also prove an authenticated client works across replicas
API_KEY=${API_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')}
kubectl -n "$NS" create secret generic api-keys --from-literal=keys="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl kustomize k8s/overlays/full | sed -e "s#${REMOTE_DRIFT_IMAGE}#${DRIFT_IMAGE}#g" -e "s#${REMOTE_IMAGE}#${IMAGE}#g" | kubectl apply -f -

step "1. Workloads Ready"
for d in redis anomaly-api pushgateway prometheus grafana; do
  kubectl -n "$NS" rollout status "deploy/$d" --timeout=300s || fail "deploy/$d not ready"
done
[ "$(kubectl -n "$NS" get deploy anomaly-api -o jsonpath='{.status.readyReplicas}')" -ge 2 ] || fail "fewer than 2 API replicas ready"

step "2-3. Shared history across replicas"
# New pod each time so every request opens a fresh connection from inside the cluster (kube-proxy
# then picks a backend per connection). kubectl port-forward would pin to ONE pod, hiding the problem.
kubectl -n "$NS" run traffic --rm -i --restart=Never --image="$IMAGE" --image-pull-policy=IfNotPresent \
  --env=API_KEY="$API_KEY" -- python - --url http://anomaly-api:8000 --cells 3 --readings 32 \
  < scripts/k8s_traffic.py || fail "cells did not warm up across replicas"
for pod in $(kubectl -n "$NS" get pods -l app.kubernetes.io/name=anomaly-api -o name); do
  n=$(kubectl -n "$NS" logs "$pod" | grep -c 'POST /predict' || true)
  echo "$pod served $n /predict requests"
  [ "$n" -gt 0 ] || fail "$pod received no /predict traffic, so the spread test proves nothing"
done
port_forward anomaly-api 18000 8000
[ "$(curl -s http://127.0.0.1:18000/health | python3 -c 'import json,sys; print(json.load(sys.stdin)["history_backend"])')" = redis ] \
  || fail "/health does not report the redis history backend"

[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:18000/predict -H 'Content-Type: application/json' -d '{}')" = 401 ] \
  || fail "/predict without a key should be 401"
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18000/metrics)" = 200 ] || fail "/metrics must stay open"

step "4. Drift CronJob (real Evidently, shared inference log in Redis)"
kubectl -n "$NS" delete job drift-smoke --ignore-not-found
kubectl -n "$NS" create job drift-smoke --from=cronjob/drift-check
if ! kubectl -n "$NS" wait --for=condition=complete job/drift-smoke --timeout=420s; then
  kubectl -n "$NS" logs job/drift-smoke || true
  fail "drift job did not complete (if Evidently's result keys changed, see the banner in monitoring/drift_check.py)"
fi
kubectl -n "$NS" logs job/drift-smoke | tail -15

step "5. Pushgateway and Prometheus"
port_forward pushgateway 19091 9091
curl -s http://127.0.0.1:19091/metrics | grep -q '^data_drift_share' || fail "Pushgateway has no data_drift_share"
port_forward prometheus 19090 9090
prom() { curl -s --get http://127.0.0.1:19090/api/v1/query --data-urlencode "query=$1"; }
count_series() { prom "$1" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"]["result"]))'; }
up_api() { [ "$(count_series 'up{job="telecom-anomaly-api"} == 1')" -ge 2 ]; }
drift_seen() { [ "$(count_series 'data_drift_share{job="drift_check"}')" -ge 1 ]; }
wait_until "Prometheus to scrape both API pods (up == 1)" 120 up_api
wait_until "Prometheus to scrape the drift job's pushed metrics" 120 drift_seen
echo "Prometheus: $(count_series 'up{job="telecom-anomaly-api"} == 1') API targets up; drift metrics present"

step "6. Grafana"
port_forward grafana 13000 3000
auth=(-u "admin:${GRAFANA_PASSWORD}")
wait_until "Grafana health" 90 curl -sf http://127.0.0.1:13000/api/health
titles=$(curl -sf "${auth[@]}" "http://127.0.0.1:13000/api/search?type=dash-db" | python3 -c 'import json,sys; print(sorted(d["uid"] for d in json.load(sys.stdin)))')
echo "dashboards: $titles"
[ "$titles" = "['telecom-anomaly-api', 'telecom-anomaly-drift']" ] || fail "dashboards not provisioned: $titles"
curl -sf "${auth[@]}" http://127.0.0.1:13000/api/datasources/uid/prometheus/health | python3 -c '
import json, sys
r = json.load(sys.stdin); print("datasource:", r)
sys.exit(0 if r.get("status") == "OK" else 1)' || fail "Grafana's Prometheus datasource is unhealthy"

step "ALL CHECKS PASSED"
