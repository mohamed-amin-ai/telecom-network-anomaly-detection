# Telecom Network Anomaly Detection System

An end-to-end ML system that detects anomalies in telecom network KPIs (latency, throughput, packet loss) in real time, deployed as a production-ready REST API.

**Key result: 97.0% recall on detecting network faults before service degradation.**

---

## Problem

Telecom networks generate thousands of KPI readings per minute across hundreds of cell towers. Identifying fault conditions (congestion, hardware degradation, interference) manually is slow and reactive. This system flags anomalies automatically, enabling NOC teams to act before users are impacted.

## Solution

A supervised XGBoost classifier trained on time-series KPI features, served via FastAPI for real-time inference.

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
│   ├── generate_data.py   # Synthetic KPI dataset generator
│   ├── features.py        # Time-series feature engineering
│   └── raw_kpis.csv       # Generated after running generate_data.py
├── models/
│   ├── train.py           # XGBoost training + threshold tuning
│   ├── model.json         # Saved model (generated after training)
│   ├── scaler.pkl         # Feature scaler (generated after training)
│   └── metadata.json      # Threshold + metrics (generated after training)
├── api/
│   └── main.py            # FastAPI inference service
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python data/generate_data.py
```

### 3. Engineer features
```bash
python data/features.py
```

### 4. Train the model
```bash
python models/train.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Make a prediction

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

Send ~20 more readings for the same `cell_id`, in chronological order, and the API starts returning real predictions once it has enough history to compute the time-series features (see [How `/predict` builds its feature vector](#api-endpoints) further down). If you already have pre-computed features on hand, pass them directly via the `features` field instead and skip the warm-up.

### 7. View interactive API docs
Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running with Docker

Build and run the API in a container instead of a local Python environment:

```bash
docker compose up --build
```

The API is then available at [http://localhost:8000](http://localhost:8000) — same endpoints as above.

Or with plain Docker, no compose:

```bash
docker build -t telecom-anomaly-detection .
docker run -p 8000:8000 telecom-anomaly-detection
```

Notes:
- Runs as a non-root user, with a container `HEALTHCHECK` against `/health`.
- The trained model artifacts (`models/model.json`, `scaler.pkl`, `metadata.json`) are baked into the image at build time — rebuild the image after retraining.

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
| Threshold | Tuned on validation set to reach a 96.5% recall target |
| Train/test split | Time-based 80/20 (no shuffle) |

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
| POST | `/predict` | Single snapshot prediction |
| POST | `/predict/batch` | Batch of up to 500 snapshots |

---

## Results

| Metric | Value |
|---|---|
| Recall | **97.0%** — catches 97 of every 100 faults |
| Precision | **79.5%** — ~20% false alarm rate, acceptable for NOC operations |
| F1 Score | **0.874** |

*Results on held-out test set (last 20% of time window, never seen during training).*
*Threshold tuned to a 96.5% recall target on the validation set, with a 93% recall floor required on the test set for a model to be promoted — missing a fault is more costly than a false alarm in NOC operations.*
*Default dataset (`data.seed: 42`). Across 10 independent synthetic datasets the median is 96.4% recall / 78.4% precision, and individual runs range from 94.8–97.2% recall and 59.8–92.7% precision.*

---

## MLOps

The model, API and results above are unchanged in purpose; the MLOps layer below tracks, tests, ships, serves and monitors them.

| Capability | Tool |
|---|---|
| Experiment tracking & model registry | **MLflow** |
| Data / model / pipeline versioning | **DVC** |
| CI/CD (lint, tests, pipeline run, image build & push) | **GitHub Actions** → GHCR |
| Metrics & dashboards | **Prometheus** · **Grafana** (2 dashboards) |
| Data & prediction drift detection | **Evidently** (+ Pushgateway) |
| Scheduled retrain → drift check → promote | **Airflow** |
| Portable, high-throughput serving | **ONNX** · **Triton** |
| Container orchestration | **Docker** · **Kubernetes** (kustomize) · **Redis** (shared state) |

### MLOps results

| Area | Result |
|---|---|
| Alert quality (incident level) | **100%** of fault incidents detected, **0 min** median detection delay, **0.33** false alerts per cell per day |
| Detector ranking quality | PR-AUC **0.977** |
| Automated tests | **137** passing |
| Serving (per replica) | **~27 req/s**, p95 latency **191 ms**, **181 MiB** memory |
| Horizontal scaling | 2–6 API replicas share per-cell history; a cell is ready after exactly **20** readings regardless of which replica receives them |
| Resilience | Redis outage → raw readings return `503`, pre-computed requests, `/health` and `/metrics` keep serving; recovers without restart |
| Drift detection | Unshifted traffic flags **1 / 68** features; a 1-σ shift flags **68 / 68** and sets the dataset-drift flag |
| Security | Optional API-key auth on prediction routes; non-root, read-only containers; optional HTTPS Ingress |

*Figures measured on the project's synthetic benchmark with a scikit-learn gradient-boosting stand-in for XGBoost (drift checks with a KS-test stand-in for Evidently); a run with XGBoost will differ slightly. Full tables: [EXPERIMENTS.md](EXPERIMENTS.md); deployment: [k8s/README.md](k8s/README.md).*

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `NumPy` · `FastAPI` · `Pydantic` · `Uvicorn` · `Docker` · `MLflow` · `DVC` · `GitHub Actions` · `Prometheus` · `Grafana` · `Evidently` · `Airflow` · `ONNX` · `Triton` · `Kubernetes` · `Redis`
