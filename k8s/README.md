# Kubernetes

Plain YAML + kustomize (built into `kubectl`), no Helm.

```
k8s/
├── base/               namespace, API Deployment (2 replicas), Service, PDB, Redis (+ NetworkPolicy)
├── overlays/scaled/    + HorizontalPodAutoscaler (2-6 replicas)
├── monitoring/         Prometheus (+RBAC), Pushgateway, Grafana (2 provisioned dashboards), drift CronJob
├── overlays/full/      scaled + monitoring  ← what you normally apply
└── optional/           ServiceMonitor, only if you run the Prometheus Operator instead of monitoring/
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
| Everything inside a cluster (probes, kube-proxy balancing, `kubernetes_sd`, RBAC, NetworkPolicy, PVCs, the CronJob, Grafana provisioning, **real XGBoost and real Evidently**) | CI `kubernetes-manifests` job on kind (`scripts/k8s_e2e.sh`) | **not run yet** — first CI run is the first real check |

## Known limits

- Resource requests/limits and the HPA's 70% CPU target are starting guesses, not measurements.
- Redis is a single instance (no HA/Sentinel). The NetworkPolicy only restricts who may reach Redis, and is enforced
  only if your CNI supports NetworkPolicy (kind's default does not).
- No Ingress, TLS or API authentication. Grafana is reachable only inside the cluster / by port-forward.
- Prometheus keeps 15 days on a 5 Gi PVC and needs a default StorageClass; Grafana state (users, edited dashboards) is
  not persisted — provisioned dashboards reappear on restart, manual edits do not.
- Airflow and Triton are not ported; Compose still runs them.
- Evidently's default dataset-drift rule (>= 50% of columns drifted) may be too coarse for 68 features from 5 KPIs.
