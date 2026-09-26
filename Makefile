.PHONY: install pipeline repro train mlflow-ui serve docker-build docker-up clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest tests/test_features.py tests/test_common.py tests/test_inference_log.py -v

test-api:
	pytest tests/test_api.py -v

test-all: test test-api

# Full data → features → train pipeline, DVC-tracked and skip-if-unchanged
repro:
	dvc repro

# Force a full pipeline run regardless of DVC's cache
pipeline:
	python data/generate_data.py
	python data/features.py
	python models/train.py

train:
	python models/train.py

# Local MLflow UI at http://localhost:5000 (uses ./mlruns if
# MLFLOW_TRACKING_URI is unset)
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

serve:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up --build

observability-urls:
	@echo "API:        http://localhost:8000/docs"
	@echo "Metrics:    http://localhost:8000/metrics"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (login: admin / admin)"

demo-traffic:
	python scripts/generate_traffic.py --url http://localhost:8000 --requests 300

drift-check:
	python monitoring/drift_check.py

clean:
	rm -rf mlruns mlartifacts .pytest_cache **/__pycache__
