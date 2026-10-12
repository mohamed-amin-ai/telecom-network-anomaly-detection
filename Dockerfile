# Telecom Network Anomaly Detection System
#   docker build .                  -> `api` image (default, last stage): the FastAPI service
#   docker build --target drift .   -> `drift` image: the same code plus Evidently, for the drift CronJob
FROM python:3.12-slim AS base

# Keep pip/python quiet and predictable
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so this layer is cached unless requirements-serving.txt changes (runtime deps only; see that file)
COPY requirements-serving.txt .
# the base image ships an old pip whose resolver fails on the opentelemetry pins that the CI runner resolves fine
RUN pip install --upgrade pip && pip install -r requirements-serving.txt

# Copy only what the service actually needs at runtime
COPY api/ ./api
COPY data/ ./data
COPY models/ ./models
# drift_check.py runs as a Kubernetes CronJob from the `drift` target below (k8s/monitoring/)
COPY monitoring/ ./monitoring

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser

# ── drift: base + Evidently (only the CronJob needs it; it is most of the API image's old weight) ──
FROM base AS drift
COPY requirements-drift.txt .
RUN pip install -r requirements-drift.txt
USER appuser

# ── api: the default target ──
FROM base AS api
USER appuser

EXPOSE 8000

# Basic liveness check against the app's own /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
