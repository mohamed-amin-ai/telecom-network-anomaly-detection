"""
auth.py
--------
Optional API-key authentication for the prediction routes.

  API_KEYS unset/empty  -> routes are open (local dev, docker-compose, the default)
  API_KEYS="k1,k2"      -> /predict, /predict/batch and /model/info require header `X-API-Key: <one of them>`

Always open, on purpose: /health (kubelet probes carry no key), /metrics (Prometheus scrapes inside the
cluster) and the OpenAPI docs. Keep /metrics off the public internet at the Ingress instead.
Several keys are allowed so one can be rotated without downtime: add the new key, move clients, remove the old one.
Keys are compared in constant time. This is shared-secret auth, not per-user identity or rate limiting.
"""

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


def configured_keys() -> list[str]:
    """Read per call (not at import) so a rotated Secret takes effect on the next pod start and tests can set it."""
    return [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    keys = configured_keys()
    if not keys:
        return
    if x_api_key is not None and any(hmac.compare_digest(x_api_key.encode(), k.encode()) for k in keys):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid API key (header X-API-Key).",
                        headers={"WWW-Authenticate": "ApiKey"})
