# Telecom Network Anomaly Detection System

An end-to-end ML system that detects anomalies in telecom network KPIs (latency, throughput, packet loss) in real time, deployed as a production-ready REST API.

**Key result: 95.2% recall on detecting network faults before service degradation.**

---

## Problem

Telecom networks generate thousands of KPI readings per minute across hundreds of cell towers. Identifying fault conditions (congestion, hardware degradation, interference) manually is slow and reactive. This system flags anomalies automatically, enabling NOC teams to act before users are impacted.

## Solution

A supervised XGBoost classifier trained on time-series KPI features, served via FastAPI for real-time inference. The pipeline is versioned end-to-end with DVC (data + model artifacts) and MLflow (experiments + model registry).

### KPIs monitored
| KPI | Unit | What it signals |
|---|---|---|
| `latency_ms` | ms | Congestion, backhaul issues |
| `throughput_mbps` | Mbps | Hardware faults, interference |
| `packet_loss_pct` | % | Link instability, interference |
| `sinr_db` | dB | Signal quality, interference |
| `connected_users` | count | Load-related anomalies |

---

## Project Structure

```
telecom_anomaly/
├── data/
│   ├── generate_data.py   # Synthetic KPI dataset generator (params.yaml-driven)
│   ├── features.py        # Time-series feature engineering
│   └── raw_kpis.csv       # DVC-tracked output, not committed to git
├── models/
│   ├── common.py           # Shared split/threshold logic (no xgboost/mlflow dep)
│   ├── train.py            # XGBoost training + threshold tuning + MLflow logging
│   ├── model.json          # DVC-tracked output
│   ├── scaler.pkl          # DVC-tracked output
│   └── metadata.json       # DVC-tracked output (also an MLflow artifact)
├── api/
│   ├── main.py             # FastAPI inference service
│   ├── live_features.py    # Rolling per-cell feature computation for /predict
│   └── metrics.py          # Prometheus business metrics
├── tests/                  # pytest suite — see Phase 2 section below
├── scripts/
│   ├── local_experiment_sweep.py       # offline hyperparameter sweep, see EXPERIMENTS.md
│   ├── replay_experiments_to_mlflow.py # loads sweep results into real MLflow
│   └── generate_traffic.py             # demo traffic generator for Grafana
├── prometheus/prometheus.yml            # scrape config
├── grafana/
│   ├── provisioning/                    # auto-configured datasource + dashboard loader
│   └── dashboards/telecom-anomaly-api.json
├── .github/workflows/ci.yml             # lint -> test -> pipeline -> docker build/push
├── params.yaml             # Single source of truth for pipeline hyperparameters
├── dvc.yaml                # DVC pipeline: generate_data → features → train
├── requirements.txt
├── requirements-dev.txt    # pytest, ruff, httpx — dev/CI only
├── docker-compose.yml      # API + Prometheus + Grafana
├── Makefile                # One-line commands for common workflows
├── EXPERIMENTS.md          # Real hyperparameter sweep results + findings
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (DVC)
```bash
dvc repro
```
This runs `generate_data.py → features.py → train.py` in order, skipping any stage whose inputs/params haven't changed since the last run. Each `train.py` run also logs to MLflow (see below).

Prefer to run steps manually instead? `make pipeline` (or the individual scripts) works the same way without DVC's caching.

### 3. Inspect experiments
```bash
mlflow ui --port 5000
```
Open [http://localhost:5000](http://localhost:5000) to compare runs, params, and metrics, and to see the model registered under `telecom-anomaly-xgb` in the Model Registry.

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Make a prediction

A brand-new cell has no history yet, so the **first call for any `cell_id` intentionally returns 425 Too Early** — the API won't guess with a mostly-empty feature vector:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "CELL_001",
    "latency_ms": 145.5,
    "throughput_mbps": 12.3,
    "packet_loss_pct": 8.7,
    "sinr_db": 4.2,
    "connected_users": 87
  }'
```

Send ~20 more readings for the same `cell_id`, in chronological order, and the API starts returning real predictions once it has enough history to compute the time-series features. If you already have pre-computed features on hand, pass them directly via the `features` field instead and skip the warm-up.

### 6. View interactive API docs
Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running with Docker

Build and run the full stack — API + Prometheus + Grafana — in containers instead of a local Python environment:

```bash
docker compose up --build
```

The API is then available at [http://localhost:8000](http://localhost:8000) — same endpoints as above. Prometheus and Grafana come up alongside it automatically; see [Phase 3](#phase-3--prometheus--grafana) below for their URLs and what's on the dashboard.

Notes:
- Runs as a non-root user, with a container `HEALTHCHECK` against `/health`.
- The trained model artifacts (`models/model.json`, `scaler.pkl`, `metadata.json`) must exist locally before building the image (run `dvc repro` or `dvc pull` first) — they're baked into the image at build time. Rebuild after retraining.
- All three services share a Docker Compose network (`anomaly-net`); Prometheus finds the API via Docker's internal DNS (`anomaly-api:8000`), not `localhost`.

---

## MLOps Stack (in progress, built phase by phase)

| Phase | Tooling | Status |
|---|---|---|
| 1 | **MLflow** (experiment tracking + model registry), **DVC** (data/model versioning) | ✅ done |
| 2 | **CI/CD** (GitHub Actions: lint, unit tests, real pipeline run, API integration tests, Docker build/push to GHCR) | ✅ done |
| 3 | **Prometheus** + **Grafana** (metrics, dashboards, alerting) | ✅ done |
| 4 | **Evidently** (data/prediction drift detection) | planned |
| 5 | **Airflow** (scheduled retrain → drift-check → promote DAG) | planned |
| 6 | **ONNX** + **Triton** (portable, high-throughput serving) | planned |

### Phase 1 — MLflow + DVC

**Why both, not just one?**
- **DVC** versions *files* — the raw CSV, the engineered feature set, the trained model binary, the scaler — the same way git versions code, so any commit can be checked out alongside the exact data/model it was trained with (`dvc.lock` pins hashes).
- **MLflow** versions *experiments* — every training run's hyperparameters, metrics, and the resulting model, queryable and comparable, with a registry for promoting a specific run's model to `Staging`/`Production`.

**Configuring a DVC remote** (so `dvc push`/`dvc pull` actually sync artifacts somewhere — without this, DVC only caches locally):
```bash
dvc remote add -d storage s3://your-bucket/telecom-anomaly-dvc
# or: dvc remote add -d storage gs://your-bucket/telecom-anomaly-dvc
dvc push
```

**Configuring MLflow tracking** (defaults to a local `./mlruns` folder if unset — fine for solo experimentation, but point at a real tracking server for team use):
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

**Reproducing a specific past state:**
```bash
git checkout <commit>
dvc checkout   # restores data/model files to match that commit's dvc.lock
```

**Real experiment results:** see [`EXPERIMENTS.md`](EXPERIMENTS.md) for a
hyperparameter sweep run against the real generated data
(`scripts/local_experiment_sweep.py`), including a genuine data-driven
change to `params.yaml` (`learning_rate: 0.05 → 0.10`) and a real
discrepancy caught between the documented ~5% anomaly rate and the
actual ~1% the generator produces. `scripts/replay_experiments_to_mlflow.py`
loads those same real numbers into MLflow once it's installed.

### Phase 2 — CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`, as four
sequential jobs (each only runs if the previous one passes):

1. **Lint** — `ruff check .`
2. **Unit tests** — `tests/test_features.py` + `tests/test_common.py`.
   These test the pure transformation/split/threshold logic directly, with
   no trained model required, so they run on every PR immediately.
3. **Real pipeline + integration tests** — runs
   `generate_data.py → features.py → train.py` for real (with real
   XGBoost, available in the Actions runner), producing a real trained
   model, then runs `tests/test_api.py` against it, then uploads the
   model artifacts as a workflow artifact.
4. **Docker build & push** — only on a push to `main` (not on PRs). Builds
   the image and pushes to `ghcr.io/<owner>/<repo>:latest` and
   `:<commit-sha>`, authenticated with the Actions-provided
   `GITHUB_TOKEN` — no personal access token is stored or required for
   this workflow.

Run the same checks locally before pushing:
```bash
make lint
make test          # unit tests only, no model needed
make pipeline       # generate_data.py -> features.py -> train.py
make test-api       # needs the above to have run first
```

**Known gap (documented, not hidden):** CI currently runs the three
pipeline scripts directly rather than `dvc repro`, because this repo
doesn't have `dvc init` run yet (no `.dvc/` directory committed — that
needs an actual DVC install to generate, which the environment this repo
was originally built in couldn't do, no network access). Once you run
`dvc init` and commit `.dvc/config` + `.dvc/.gitignore`, swap the
pipeline step in `ci.yml` for `dvc repro` to get caching (skip stages
whose inputs/params haven't changed) instead of a full rerun every time.

### Phase 3 — Prometheus + Grafana

```bash
docker compose up --build
make observability-urls   # prints all four URLs below
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000/docs | interactive Swagger UI |
| API metrics (raw) | http://localhost:8000/metrics | what Prometheus scrapes |
| Prometheus | http://localhost:9090 | query/explore raw metrics |
| Grafana | http://localhost:3000 | login `admin` / `admin` — **change this if this ever runs anywhere but your own machine** |

Grafana auto-provisions on first boot: the Prometheus datasource and a
9-panel **"Telecom Anomaly Detection API"** dashboard are both already
there, no manual setup. It'll be empty until some traffic hits the API —
generate a batch of realistic + deliberately-anomalous requests to see
it populate:

```bash
make demo-traffic   # or: python scripts/generate_traffic.py --requests 300
```

**What's actually instrumented** (`api/metrics.py`):
- Standard HTTP metrics (request rate, latency percentiles, status codes)
  via `prometheus-fastapi-instrumentator`, labeled by route.
- `anomaly_predictions_total{severity}` — counts by normal/warning/critical.
- `feature_source_total{source}` — precomputed vs live-computed features.
- `anomaly_score` — a histogram of raw model probabilities, so you can see
  the actual score distribution, not just the post-threshold decision.
- `cold_start_responses_total` — how often callers hit the 425
  insufficient-history response.
- `cells_tracked` — gauge, size of the in-process history store.
- `model_threshold_info` — gauge; a step change on this panel marks a
  model redeploy with a newly tuned threshold.

**Deliberately not labeled by `cell_id`** — see the docstring in
`api/metrics.py`. The README describes "hundreds of cell towers" in a
real deployment; a Prometheus label creates one time series per unique
value per metric, so labeling by cell_id would mean hundreds of growing
time series per metric. Severity and feature_source are small, fixed
label sets — safe. Per-cell drill-down belongs in logs, not metrics.

---

## Feature Engineering

For each KPI, the following features are computed per cell tower:

| Feature type | Detail |
|---|---|
| **Lag features** | Values at t-1, t-2, t-4, t-8 (15 min to 2 hours ago) |
| **Rolling mean** | 1h and 4h windows |
| **Rolling std** | 1h and 4h windows — captures variance bursts |
| **Rolling min/max** | 1h window |
| **Z-score** | Deviation from 4h rolling mean |
| **Rate of change** | `(t - t-1) / t-1` — detects sudden jumps |
| **Time features** | Hour of day, day of week, is_weekend |

Total: **68 features** across 5 KPIs.

| Feature group | Count |
|---|---|
| Lag features (t-1, t-2, t-4, t-8) | 5 KPIs × 4 = 20 |
| Rolling mean (1h, 4h) | 5 KPIs × 2 = 10 |
| Rolling std (1h, 4h) | 5 KPIs × 2 = 10 |
| Rolling min/max (1h) | 5 KPIs × 2 = 10 |
| Z-score (4h window) | 5 KPIs × 1 = 5 |
| Rate of change | 5 KPIs × 1 = 5 |
| Raw KPI values | 5 |
| Time features (hour, day_of_week, is_weekend) | 3 |
| **Total** | **68** |

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (gradient boosted trees) |
| Trees | 300 (with early stopping) |
| Max depth | 6 |
| Learning rate | 0.05 |
| Class imbalance | Handled via `scale_pos_weight` |
| Threshold | Tuned on validation set to maximize F1 subject to a recall floor |
| Train/test split | Time-based 80/20 (no shuffle) |

All of the above now live in `params.yaml`, not hardcoded in `train.py` — change a value there and `dvc repro` knows to retrain.

### Why XGBoost?
- Handles tabular time-series features well out of the box
- Robust to feature scale differences and missing values
- Fast inference (<1ms per prediction) — suitable for real-time monitoring
- Feature importance scores aid explainability for NOC teams

### Why tune the threshold?
The default 0.5 threshold maximizes accuracy, not recall. In anomaly detection, a missed fault (false negative) is far more costly than a false alarm. Threshold tuning lets us control this trade-off explicitly.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model/info` | Model metrics and threshold |
| GET | `/metrics` | Prometheus scrape endpoint (HTTP + business metrics) |
| POST | `/predict` | Single snapshot prediction |
| POST | `/predict/batch` | Batch of up to 500 snapshots |

---

## Results

| Metric | Value |
|---|---|
| Recall | **95.2%** — catches 95 of every 100 faults |
| Precision | **76.9%** — ~23% false alarm rate, acceptable for NOC operations |
| F1 Score | **0.851** |

*Results on held-out test set (last 20% of time window, never seen during training).*
*Threshold tuned to maximize precision subject to a 93% recall floor — missing a fault is more costly than a false alarm in NOC operations.*

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `NumPy` · `FastAPI` · `Pydantic` · `Uvicorn` · `Docker` · `MLflow` · `DVC` · `Prometheus` · `Grafana`
