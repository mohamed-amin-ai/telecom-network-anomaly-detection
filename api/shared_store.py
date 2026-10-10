"""
shared_store.py
----------------
One place that decides whether this process uses a shared Redis or stays
single-process.

  REDIS_URL unset  -> in-memory history + local-file inference log (docker-compose, local dev)
  REDIS_URL set    -> history and inference log live in Redis, shared by all API replicas
                      and readable by the drift-check job (see k8s/)

Other settings (all optional):
  HISTORY_TTL_SECONDS       a cell's history expires after this long without a reading (default 86400)
  INFERENCE_LOG_MAX_ROWS    cap on the shared inference log, newest kept (default 20000)

`redis` is imported lazily so the package is only needed when REDIS_URL is set.
"""

import os

DEFAULT_HISTORY_TTL_SECONDS = 86400
DEFAULT_INFERENCE_LOG_MAX_ROWS = 20000


def redis_url() -> str | None:
    return os.environ.get("REDIS_URL") or None


def get_redis_client():
    """A redis client for REDIS_URL, or None when it is unset. Short socket timeouts: a
    slow Redis must fail a request quickly (-> 503), not hang the worker."""
    url = redis_url()
    if url is None:
        return None
    import redis
    return redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, health_check_interval=30)


def history_ttl_seconds() -> int:
    return int(os.environ.get("HISTORY_TTL_SECONDS", DEFAULT_HISTORY_TTL_SECONDS))


def inference_log_max_rows() -> int:
    return int(os.environ.get("INFERENCE_LOG_MAX_ROWS", DEFAULT_INFERENCE_LOG_MAX_ROWS))


def build_history_backend():
    """Memory backend by default; Redis-backed when REDIS_URL is set."""
    from api.live_features import MemoryHistoryBackend, RedisHistoryBackend
    client = get_redis_client()
    if client is None:
        return MemoryHistoryBackend()
    return RedisHistoryBackend(client, ttl_seconds=history_ttl_seconds())
