# Kubernetes

Plain YAML + kustomize (built into `kubectl`), no Helm.

```
k8s/
├── base/               namespace, API Deployment (2 replicas), Service, PDB, Redis (+ NetworkPolicy)
├── overlays/scaled/    + HorizontalPodAutoscaler (2-6 replicas)
├── monitoring/         Prometheus (+RBAC), Pushgateway, Grafana (2 provisioned dashboards), drift CronJob
├── overlays/full/      scaled + monitoring  ← what you normally apply
└── optional/           ingress.yaml (HTTPS entry for the API), servicemonitor.yaml (only with the Prometheus Operator)
```

## Run it

```bash
# 1. Pin the images CI built (ghcr.io tags: :latest and :<commit sha>): the API image, and the
#    drift-check image (the same code plus Evidently) used by the drift CronJob
cd k8s/overlays/full && kustomize edit set image \
  ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection=ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection:<sha> \
  ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection-drift=ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection-drift:<sha>

# 2. Grafana's password is never in git
GRAFANA_PASSWORD='<choose>' make k8s-secret

# 3. Deploy, then look
make k8s-apply
kubectl -n telecom-anomaly get pods
make k8s-port-forward        # API  -> http://localhost:8000/docs
make k8s-grafana             # UI   -> http://localhost:3000  (user: admin)
```

**Optional hardening:**
- *API keys:* `kubectl -n telecom-anomaly create secret generic api-keys --from-literal=keys='k1,k2'`, then restart the API
  pods. `/predict`, `/predict/batch` and `/model/info` then need header `X-API-Key`; `/health` and `/metrics` stay open
  (probes and Prometheus carry no key). No Secret means open routes. Comma-separated keys allow rotation without downtime.
- *HTTPS:* edit the host in `k8s/optional/ingress.yaml` and apply it (ingress-nginx + cert-manager assumed). It routes only
  API paths, so `/metrics` is not reachable from outside. TLS alone does not authenticate anyone: turn on API keys too.

Only want the API? `make k8s-apply-base` (API + Redis, no monitoring).
**Private GHCR package:** the image pulls anonymously only if the package is public; otherwise add an `imagePullSecret`.
The CI end-to-end run on kind: `IMAGE=telecom-anomaly-api:ci DRIFT_IMAGE=telecom-anomaly-drift:ci make k8s-e2e` (needs both images loaded into the cluster).

## How it fits together

- **Shared state in Redis.** `REDIS_URL` switches `api/shared_store.py` from per-process memory to Redis for (a) each cell's
  rolling history and (b) the inference log, so any replica can serve any reading and the drift job reads one log.
  Without `REDIS_URL` (docker-compose, local dev) nothing changes: memory + a CSV file.
- **Redis is a cache, not a database:** no persistence, `noeviction` with a 256 MB cap, history keys expire after 24 h
  without a reading, the inference log keeps the newest 20,000 rows. A Redis restart empties history; each cell then
  returns `425` until it has 20 new readings again, and the drift job sees an empty log until traffic resumes.
- **Redis outage:** raw readings answer `503` (after at most a 2 s socket timeout); requests carrying pre-computed
  features, `/health` and `/metrics` are unaffected, so pods stay in rotation instead of crash-looping.
- **Scraping:** Prometheus discovers API *pods* through a namespaced Role (not through the Service, which would
  load-balance each scrape to a random replica). Pushgateway holds the drift job's metrics; Grafana's datasource and dashboards are provisioned from ConfigMaps.
- **Drift CronJob** (hourly) runs `monitoring/drift_check.py` from the `drift` image (the API image plus Evidently, `docker build --target drift .`): it compares the Redis inference log
  to `models/reference_features.csv` (written by `train.py`, baked into the image) and pushes `data_drift_*` metrics.
- The two Grafana dashboards under `k8s/monitoring/grafana/dashboards/` are copies of `grafana/dashboards/` (kustomize can't
  read outside its directory); `tests/test_k8s_manifests.py` fails if they diverge.

## What was verified, and how

| Check | Where | Real or stand-in |
|---|---|---|
| All four kustomizations render; 23 objects in `full` pass `kubeconform --strict` against the Kubernetes 1.30 schemas | sandbox | real kustomize + kubeconform |
| History/inference log against a live Redis (incl. two replicas sharing it, trimming, expiry) | sandbox + CI | real redis-server, real redis-py |
| 2 API processes + Redis + Prometheus + Pushgateway: cells warm up after exactly 20 readings while alternating replicas; every API-dashboard query returns data; `cells_tracked` is shared; Redis down -> 503 / precomputed 200 / `/health` 200; recovery | sandbox | real FastAPI app, real Prometheus 2.55.1, Pushgateway 1.9.0; model is a scikit-learn stand-in for XGBoost |
| drift job -> Pushgateway -> Prometheus -> the drift dashboard's queries | sandbox, then CI on kind | stand-in in the sandbox; **real Evidently 0.4.33 in CI** (the CronJob completes, Pushgateway holds `data_drift_share`) |
| Resource sizing (below) | CI on shared runners (`scripts/measure_resources.py`) | measured on **real XGBoost** |
| HPA scale-out under sustained load (metrics-server, 24 closed-loop threads for 240 s, `scripts/eval/k8s_hpa_eval.sh`) | CI on kind | real: scale-out decided 30 s after load began, 6 desired replicas, 5 Ready, 1 Pending (the 4-vCPU runner ran out of CPU), peak 190% of the CPU request, 496 req/s at p95 91 ms, 0 request errors |
| API-key auth: 401 without a key, either of two keys accepted, `/health` and `/metrics` open | sandbox (real FastAPI) | real |
| Everything inside a cluster (probes, kube-proxy balancing, `kubernetes_sd`, RBAC, PVCs, the CronJob, Grafana provisioning, **real XGBoost and real Evidently**) | CI `kubernetes-manifests` job on kind (`scripts/k8s_e2e.sh`) | **passing** |

## Resource sizing

Measured with `scripts/measure_resources.py` (4 concurrent clients, warmed cells, **real XGBoost**, GitHub-hosted runner; three runs):

| | measured | set to |
|---|---|---|
| API memory | 233-234 MiB idle, 240 MiB peak RSS (`kubectl top` working set under the HPA test: 174 MiB per pod) | request 320 Mi, limit 512 Mi |
| API CPU | **4.3 ms per scored request** after the NumPy feature path (51.5 ms before); one process serves 292 req/s, p50 13 ms, p95 19 ms at that load (run-to-run spread ~2x on shared runners) | request 500m, limit 1 |
| Redis memory | 5.0 MiB per ~1.2k inference-log rows, so ~85 MiB at the 20k-row cap | request 128 Mi, limit 400 Mi (`maxmemory` 256 MB) |

The HPA's 70% target applies to the 500m request. Under 24 closed-loop clients on kind the HPA decided to scale out
30 s after load began and asked for the 6-replica maximum; with 4 vCPUs shared with the control plane only 5 pods could
be scheduled, so size the node pool for `maxReplicas x 500m`. **Finding:** the cost is dominated by recomputing features
over the cell's history on every request, not by the model, so capacity scales by adding replicas rather than by a bigger
model host. The HPA has an explicit `behavior` block (scale out up to 2 pods per 30 s with no stabilisation; scale in 1 pod per 60 s
after a 180 s window). Measured: first scale-in 205 s after load ended, back at the 2-replica minimum after 388 s.

| Component | Measured peak (idle / under load) | Request / limit |
|---|---|---|
| Prometheus | 5m CPU, 35 MiB; 1.3k series, ~87 samples/s, ~160 MiB estimated for 15 d | 50m, 128 Mi / 500m, 512 Mi; PVC 2 Gi |
| Grafana | 3m CPU, 67 MiB | 50m, 128 Mi / 300m, 256 Mi |
| Pushgateway | 1m CPU, 16 MiB | 20m, 32 Mi / 100m, 64 Mi |
| Redis | 101m CPU, 91 MiB at 496 req/s (the 20k-row inference-log cap reached) | 100m, 128 Mi / 500m, 400 Mi |
| Drift CronJob | 999m CPU, 284 MiB; finished in 22 s | 250m, 384 Mi / 1, 768 Mi |

These are one run on a shared 4-vCPU runner with a small synthetic log, so they are floors, not production sizing.

## Known limits

- Redis is a single instance (no HA/Sentinel). The NetworkPolicy only restricts who may reach Redis, and is enforced
  only if your CNI supports NetworkPolicy (kind's default does not). No default-deny policy for the rest of the namespace.
- API-key auth is a shared secret: no per-client identity, quotas or rate limiting.
- Prometheus keeps 15 days on a 5 Gi PVC; Grafana state is on a 1 Gi PVC. Both need a default StorageClass.
- Airflow and Triton are not ported; Compose still runs them.
- Evidently's default dataset-drift rule (>= 50% of columns drifted) may be too coarse for 68 features from 5 KPIs.
