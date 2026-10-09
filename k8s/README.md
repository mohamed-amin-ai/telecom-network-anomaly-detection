# Kubernetes

Manifests for running the API on a cluster. Plain YAML + kustomize (built into `kubectl`), no Helm.

```
k8s/
├── base/                 namespace, Deployment, Service, PVC (inference log)  ← what you normally apply
├── overlays/scaled/      HPA + PDB, 2–6 replicas — read "Scaling" first
└── optional/             ServiceMonitor (needs the Prometheus Operator CRDs)
```

## Run it

```bash
# 1. Pin the image CI built (kubectl alone can't edit kustomize files; use kustomize or edit k8s/base/kustomization.yaml)
#    images: newTag: <commit sha>        # tags pushed: :latest and :<sha> on ghcr.io

# 2. Deploy
make k8s-render      # inspect what will be applied
make k8s-apply
kubectl -n telecom-anomaly rollout status deploy/anomaly-api

# 3. Try it
make k8s-port-forward            # then http://localhost:8000/docs
```

**Local cluster (kind/minikube):** `docker build -t telecom-anomaly-api:local .` (run the pipeline first — the image
bakes in `models/model.json`, `scaler.pkl`, `metadata.json`), `kind load docker-image telecom-anomaly-api:local`, then set
the image in the kustomization to `telecom-anomaly-api` / `newTag: local`.

**Private GHCR package:** the image is pulled anonymously only if the package is public. Otherwise create a
`docker-registry` secret and add `imagePullSecrets` to the Deployment.

## Scaling — read before turning replicas up

`api/live_features.py` keeps each cell's recent readings **in process memory**. Raw-reading requests need ~20 prior
readings of the same cell on the *same pod*. With several replicas, requests for one cell are spread across pods, so each
pod sees a fraction of the history and the API keeps returning `425` (cold start). That is why the base has one replica,
`strategy: Recreate` (the log volume is ReadWriteOnce), and a restart empties the history.

- **Pre-computed-feature clients** (`/predict` with features supplied) are stateless: use `overlays/scaled`.
- **Raw-reading clients** at scale need the history moved to a shared store (e.g. Redis, keyed by `cell_id`) first. Not done.
- The scaled overlay replaces the log PVC with a per-pod `emptyDir`, so `monitoring/drift_check.py` can no longer read one
  shared inference log. Shipping the log to shared storage is also not done.

## What is and isn't verified

- Verified offline: YAML parses; `tests/test_k8s_manifests.py` checks selectors, ports, probe paths, image name, mount
  paths, UID and overlay targets against the rest of the repo.
- Verified by CI only (first run is the first real check): `kubectl kustomize` renders both overlays, `kubeconform`
  accepts the schema, and a kind cluster brings the Deployment to Ready and answers `/health` and `/metrics` via the Service.
- **Not verified anywhere:** CPU/memory requests and limits (250m/512Mi → 1/1Gi) are starting guesses, not measurements;
  HPA thresholds likewise; no Ingress, TLS, NetworkPolicy or auth is defined; Prometheus, Grafana, Airflow and Triton are
  **not** ported — Compose still runs them. Scraping works via the pod annotations or `optional/servicemonitor.yaml`
  (set its `release:` label to your Prometheus release).
