#!/usr/bin/env bash
# Docker: build time, image size, layers, non-root user, and the container's own HEALTHCHECK going healthy.
set -u
EVAL_DIR=${EVAL_DIR:-eval/core}
mkdir -p "$EVAL_DIR"
t0=$(date +%s); docker build -t eval-api . > /tmp/dockerbuild.log 2>&1 || { tail -n 20 /tmp/dockerbuild.log; exit 1; }; t1=$(date +%s)
docker build --target drift -t eval-drift . > /tmp/dockerbuild_drift.log 2>&1 || tail -n 20 /tmp/dockerbuild_drift.log
drift_size=$(docker image inspect eval-drift --format '{{.Size}}' 2>/dev/null || echo 0)
size=$(docker image inspect eval-api --format '{{.Size}}')
layers=$(docker image inspect eval-api --format '{{len .RootFS.Layers}}')
cid=$(docker run -d -p 18030:8000 eval-api)
health=none; secs=0
for i in $(seq 1 60); do
  health=$(docker inspect --format '{{.State.Health.Status}}' "$cid"); [ "$health" = healthy ] && break; sleep 2; secs=$((i*2))
done
uid=$(docker exec "$cid" id -u)
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18030/health)
mem=$(docker stats --no-stream --format '{{.MemUsage}}' "$cid")
top=$(docker exec "$cid" sh -c "du -sm /usr/local/lib/python3.12/site-packages/* 2>/dev/null | sort -rn | head -14" | tr '\n' ';' | tr -s '\t' ' ')
docker rm -f "$cid" > /dev/null
python3 - "$EVAL_DIR" "$((t1-t0))" "$size" "$layers" "$health" "$secs" "$uid" "$code" "$mem" "$top" "$drift_size" <<'PY'
import json, sys
d, build, size, layers, health, secs, uid, code, mem, top, drift_size = sys.argv[1:12]
json.dump({"build_seconds_cold": int(build), "image_mib": round(int(size) / 2**20, 1), "drift_image_mib": round(int(drift_size) / 2**20, 1), "layers": int(layers),
           "healthcheck_status": health, "seconds_to_healthy": int(secs), "runs_as_uid": int(uid),
           "health_http": int(code), "container_memory_idle": mem,
           "largest_site_packages_mib": [t.strip() for t in top.split(";") if t.strip()]}, open(f"{d}/docker.json", "w"), indent=2)
PY
cat "$EVAL_DIR/docker.json"
