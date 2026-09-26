.PHONY: install pipeline repro train mlflow-ui serve docker-build docker-up clean

install:
	pip install -r requirements.txt

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

clean:
	rm -rf mlruns mlartifacts .pytest_cache **/__pycache__
