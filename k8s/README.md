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
# 1. Pin the image CI built (ghcr.io tags: :latest and :<commit sha>)
cd k8s/overlays/full && kustomize edit set image \
  ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection=ghcr.io/mohamed-amin-ai/telecom-network-anomaly-detection:<sha>

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
The CI end-to-end run on kind: `IMAGE=telecom-anomaly-api:ci make k8s-e2e` (needs the image loaded into the cluster).

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
- **Drift CronJob** (hourly) runs `monitoring/drift_check.py` from the API image: it compares the Redis inference log
  to `models/reference_features.csv` (written by `train.py`, baked into the image) and pushes `data_drift_*` metrics.
- The two Grafana dashboards under `k8s/monitoring/grafana/dashboards/` are copies of `grafana/dashboards/` (kustomize can't
  read outside its directory); `tests/test_k8s_manifests.py` fails if they diverge.

## What was verified, and how

| Check | Where | Real or stand-in |
|---|---|---|
| All four kustomizations render; 23 objects in `full` pass `kubeconform --strict` against the Kubernetes 1.30 schemas | sandbox | real kustomize + kubeconform |
| History/inference log against a live Redis (incl. two replicas sharing it, trimming, expiry) | sandbox + CI | real redis-server, real redis-py |
| 2 API processes + Redis + Prometheus + Pushgateway: cells warm up after exactly 20 readings while alternating replicas; every API-dashboard query returns data; `cells_tracked` is shared; Redis down -> 503 / precomputed 200 / `/health` 200; recovery | sandbox | real FastAPI app, real Prometheus 2.55.1, Pushgateway 1.9.0; model is a scikit-learn stand-in for XGBoost |
| drift job -> Pushgateway -> Prometheus -> the drift dashboard's queries | sandbox | real except **Evidently, which is a stand-in** (KS tests) |
| Resource sizing (above) | sandbox | measured, but on the stand-in model |
| API-key auth: 401 without a key, either of two keys accepted, `/health` and `/metrics` open | sandbox (real FastAPI) | real |
| Everything inside a cluster (probes, kube-proxy balancing, `kubernetes_sd`, RBAC, NetworkPolicy, PVCs, the CronJob, Grafana provisioning, **real XGBoost and real Evidently**) | CI `kubernetes-manifests` job on kind (`scripts/k8s_e2e.sh`) | **not run yet** — first CI run is the first real check |

## Resource sizing

Measured with `scripts/measure_resources.py` (4 concurrent clients, warmed cells, scikit-learn stand-in model):

| | measured | set to |
|---|---|---|
| API memory | 175 MiB idle, 181 MiB peak | request 256 Mi, limit 512 Mi |
| API CPU | **41 ms per scored request**; one process saturates ~1 core at ~27 req/s; p50 147 ms, p95 191 ms at that load | request 500m, limit 1 |
| Redis memory | 4.9 MiB per ~1.2k inference-log rows, so ~80 MiB at the 20k-row cap | request 128 Mi, limit 400 Mi (`maxmemory` 256 MB) |

The HPA's 70% target applies to the 500m request. **Finding:** the cost is dominated by recomputing features over the cell's
history on every request, not by the model, so capacity scales by adding replicas (the HPA does) rather than by a bigger
model host. Re-run the script with the real XGBoost image: its predict step differs from the stand-in. Prometheus, Grafana,
Pushgateway and the drift job (whose Evidently run here was a stand-in) are **not measured**; their values are guesses.

## Known limits

- Redis is a single instance (no HA/Sentinel). The NetworkPolicy only restricts who may reach Redis, and is enforced
  only if your CNI supports NetworkPolicy (kind's default does not). No default-deny policy for the rest of the namespace.
- API-key auth is a shared secret: no per-client identity, quotas or rate limiting.
- Prometheus keeps 15 days on a 5 Gi PVC; Grafana state is on a 1 Gi PVC. Both need a default StorageClass.
- Airflow and Triton are not ported; Compose still runs them.
- Evidently's default dataset-drift rule (>= 50% of columns drifted) may be too coarse for 68 features from 5 KPIs.
