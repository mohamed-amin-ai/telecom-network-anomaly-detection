"""
test_k8s_manifests.py
----------------------
Offline checks that the Kubernetes manifests agree with the rest of the repo and with each other.
They cannot prove a deploy works (scripts/k8s_e2e.sh in CI's kind job does that); they catch the
drift that breaks one: a selector matching no pod, a probe path the API doesn't serve, a Redis
hostname that isn't the Service's name, a dashboard copy that diverged from grafana/, a CronJob
running a script the image doesn't contain.
"""

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
K8S = REPO / "k8s"
BASE = K8S / "base"
MONITORING = K8S / "monitoring"
KUSTOMIZATIONS = [BASE, K8S / "overlays" / "scaled", K8S / "overlays" / "full", MONITORING]
API_IMAGE = "ghcr.io/mohamed-amin-ai/mlops-anomaly"


def _load(path: Path) -> list:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _kustomization(directory: Path) -> dict:
    return yaml.safe_load((directory / "kustomization.yaml").read_text())


def _docs(directory: Path) -> dict:
    """(kind, name) -> manifest, for every resource file in a directory."""
    out = {}
    for f in directory.glob("*.yaml"):
        if f.name == "kustomization.yaml":
            continue
        for d in _load(f):
            out[(d["kind"], d["metadata"]["name"])] = d
    return out


def _all_docs() -> dict:
    return {**_docs(BASE), **_docs(MONITORING)}


def _pod_spec(doc: dict) -> dict:
    if doc["kind"] == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return doc["spec"]["template"]["spec"]


def _pod_labels(doc: dict) -> dict:
    if doc["kind"] == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["metadata"]["labels"]
    return doc["spec"]["template"]["metadata"]["labels"]


def _api() -> dict:
    return _docs(BASE)[("Deployment", "anomaly-api")]


def _api_container() -> dict:
    return _pod_spec(_api())["containers"][0]


def _env(container: dict) -> dict:
    return {e["name"]: e.get("value") for e in container.get("env", [])}


def _service_for(name: str) -> dict:
    return _all_docs()[("Service", name)]


# ── kustomize wiring ────────────────────────────────────────────────────────

def test_every_kustomization_resource_and_generator_file_exists():
    for kdir in KUSTOMIZATIONS:
        k = _kustomization(kdir)
        for res in k.get("resources", []):
            assert (kdir / res).exists(), f"{kdir.name}/kustomization.yaml lists missing {res}"
        for gen in k.get("configMapGenerator", []):
            for f in gen["files"]:
                assert (kdir / f.split("=")[-1]).exists(), f"{kdir.name}: generator {gen['name']} lists missing {f}"


def test_every_resource_file_is_in_its_kustomization():
    for kdir in (BASE, MONITORING):
        listed = set(_kustomization(kdir)["resources"])
        on_disk = {f.name for f in kdir.glob("*.yaml")} - {"kustomization.yaml"}
        assert listed == on_disk, f"{kdir.name}: {listed ^ on_disk}"


def test_all_kustomizations_share_one_namespace_and_one_api_image():
    namespaces = {_kustomization(d)["namespace"] for d in KUSTOMIZATIONS}
    assert len(namespaces) == 1, namespaces
    for kdir in (BASE, MONITORING):
        assert _kustomization(kdir)["images"][0]["name"] == API_IMAGE
    for doc in (_api(), _docs(MONITORING)[("CronJob", "drift-check")]):
        assert _pod_spec(doc)["containers"][0]["image"].rsplit(":", 1)[0] == API_IMAGE


def test_image_name_matches_what_ci_pushes():
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    assert "ghcr.io/${{ github.repository }}" in ci
    assert API_IMAGE.startswith("ghcr.io/") and API_IMAGE == API_IMAGE.lower()


def test_full_overlay_composes_api_and_monitoring_and_scaled_adds_the_hpa():
    assert set(_kustomization(K8S / "overlays" / "full")["resources"]) == {"../scaled", "../../monitoring"}
    scaled = _kustomization(K8S / "overlays" / "scaled")
    assert set(scaled["resources"]) == {"../../base", "hpa.yaml"}
    hpa = _load(K8S / "overlays" / "scaled" / "hpa.yaml")[0]
    assert ("Deployment", hpa["spec"]["scaleTargetRef"]["name"]) in _docs(BASE)
    assert hpa["spec"]["minReplicas"] >= 2
    # The HPA must own the replica count, so the Deployment's own `replicas` is patched away.
    patch = scaled["patches"][0]
    assert patch["target"]["name"] == "anomaly-api" and "/spec/replicas" in patch["patch"]


# ── API deployment ──────────────────────────────────────────────────────────

def test_selectors_match_pod_labels_and_services_target_pods():
    docs = _all_docs()
    for (kind, name), d in docs.items():
        if kind in ("Deployment",):
            for k, v in d["spec"]["selector"]["matchLabels"].items():
                assert _pod_labels(d).get(k) == v, f"{name}: selector {k}={v} matches no pod label"
    templates = [_pod_labels(d) for (kind, _), d in docs.items() if kind in ("Deployment", "CronJob")]
    for (kind, name), d in docs.items():
        if kind == "Service":
            assert any(all(t.get(k) == v for k, v in d["spec"]["selector"].items()) for t in templates), \
                f"Service {name} selects no pod template"


def test_container_port_matches_dockerfile_and_service():
    port = _api_container()["ports"][0]
    exposed = int(re.search(r"^EXPOSE\s+(\d+)", (REPO / "Dockerfile").read_text(), re.M).group(1))
    assert port["containerPort"] == exposed
    assert _service_for("anomaly-api")["spec"]["ports"][0]["targetPort"] == port["name"]


def test_probes_hit_an_endpoint_the_api_serves_and_use_the_named_port():
    container = _api_container()
    api_src = (REPO / "api" / "main.py").read_text()
    for name in ("startupProbe", "readinessProbe", "livenessProbe"):
        probe = container[name]["httpGet"]
        assert f'"{probe["path"]}"' in api_src, f"{name} path {probe['path']} not served by api/main.py"
        assert probe["port"] == container["ports"][0]["name"]


def test_pod_runs_as_the_images_non_root_uid_with_a_read_only_root():
    spec = _pod_spec(_api())
    uid = int(re.search(r"--uid\s+(\d+)", (REPO / "Dockerfile").read_text()).group(1))
    assert spec["securityContext"]["runAsUser"] == uid and spec["securityContext"]["runAsNonRoot"] is True
    container = _api_container()
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert "/tmp" in {m["mountPath"] for m in container["volumeMounts"]}


def test_prometheus_annotations_match_the_metrics_endpoint():
    ann = _api()["spec"]["template"]["metadata"]["annotations"]
    assert ann["prometheus.io/path"] == "/metrics"
    assert int(ann["prometheus.io/port"]) == _api_container()["ports"][0]["containerPort"]
    assert "/metrics" in (REPO / "api" / "metrics.py").read_text() + (REPO / "api" / "main.py").read_text()


# ── Redis: what makes more than one replica correct ─────────────────────────

def test_more_than_one_api_replica_requires_the_shared_store():
    # Without REDIS_URL, history is per-process (api/live_features.py) and replicas answer 425 forever.
    api = _api()
    if api["spec"]["replicas"] > 1:
        assert "REDIS_URL" in _env(_api_container())
    assert "class MemoryHistoryBackend" in (REPO / "api" / "live_features.py").read_text()   # the default when REDIS_URL is unset
    assert "REDIS_URL" in (REPO / "api" / "shared_store.py").read_text()


def test_redis_url_points_at_the_redis_service_everywhere():
    svc = _service_for("redis")
    expected = f"redis://{svc['metadata']['name']}:{svc['spec']['ports'][0]['port']}/0"
    assert _env(_api_container())["REDIS_URL"] == expected
    assert _env(_pod_spec(_docs(MONITORING)[("CronJob", "drift-check")])["containers"][0])["REDIS_URL"] == expected


def test_redis_is_a_bounded_non_persistent_cache_and_only_clients_may_reach_it():
    redis = _docs(BASE)[("Deployment", "redis")]
    args = redis["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args[args.index("--maxmemory-policy") + 1] == "noeviction", "evicting history silently would cause cold starts"
    assert "--maxmemory" in args and args[args.index("--appendonly") + 1] == "no"
    assert redis["spec"]["replicas"] == 1
    policy = _docs(BASE)[("NetworkPolicy", "redis-allow-clients")]
    allowed = policy["spec"]["ingress"][0]["from"][0]["podSelector"]["matchLabels"]
    assert policy["spec"]["podSelector"]["matchLabels"] == redis["spec"]["selector"]["matchLabels"]
    (label, value), = allowed.items()
    for doc in (_api(), _docs(MONITORING)[("CronJob", "drift-check")]):
        assert _pod_labels(doc).get(label) == value, f"{doc['metadata']['name']} would be blocked from Redis"
    assert _pod_labels(redis).get(label) is None


def test_memory_headroom_for_the_capped_inference_log():
    # 20000 rows * ~1.5KB JSON ~= 30MB (api/shared_store.py default); well under the 256mb maxmemory.
    from api import shared_store
    assert shared_store.DEFAULT_INFERENCE_LOG_MAX_ROWS * 2048 < 256 * 1024 * 1024 / 2


# ── Monitoring ──────────────────────────────────────────────────────────────

def test_prometheus_config_discovers_api_pods_in_this_namespace_and_pushgateway_by_service():
    cfg = yaml.safe_load((MONITORING / "prometheus.yml").read_text())
    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    api = jobs["telecom-anomaly-api"]
    assert api["kubernetes_sd_configs"][0]["namespaces"]["names"] == [_kustomization(BASE)["namespace"]]
    keep = {r["regex"] for r in api["relabel_configs"] if r.get("action") == "keep"}
    assert _api()["spec"]["selector"]["matchLabels"]["app.kubernetes.io/name"] in keep
    assert api["metrics_path"] == "/metrics"
    pg = _service_for("pushgateway")
    assert jobs["pushgateway"]["static_configs"][0]["targets"] == [f"pushgateway:{pg['spec']['ports'][0]['port']}"]
    assert jobs["pushgateway"]["honor_labels"] is True      # keeps job="drift_check" for the Grafana drift panels
    role = _docs(MONITORING)[("Role", "prometheus")]
    assert {"pods"} <= set(role["rules"][0]["resources"]) and "list" in role["rules"][0]["verbs"]


def test_grafana_copies_are_identical_to_the_compose_files():
    pairs = [
        (MONITORING / "grafana/dashboards/telecom-anomaly-api.json", REPO / "grafana/dashboards/telecom-anomaly-api.json"),
        (MONITORING / "grafana/dashboards/drift-monitoring.json", REPO / "grafana/dashboards/drift-monitoring.json"),
        (MONITORING / "grafana/datasources/prometheus.yml", REPO / "grafana/provisioning/datasources/prometheus.yml"),
        (MONITORING / "grafana/provider/dashboards.yml", REPO / "grafana/provisioning/dashboards/dashboards.yml"),
    ]
    for copy, original in pairs:
        assert copy.read_bytes() == original.read_bytes(), f"{copy.name} drifted from {original.relative_to(REPO)}"
    assert {f.name for f in (REPO / "grafana/dashboards").glob("*.json")} == {f.name for f in (MONITORING / "grafana/dashboards").glob("*.json")}


def test_grafana_datasource_and_dashboard_paths_match_the_pod():
    ds = yaml.safe_load((MONITORING / "grafana/datasources/prometheus.yml").read_text())["datasources"][0]
    prom = _service_for("prometheus")
    assert ds["url"] == f"http://prometheus:{prom['spec']['ports'][0]['port']}"
    provider = yaml.safe_load((MONITORING / "grafana/provider/dashboards.yml").read_text())["providers"][0]
    mounts = {m["name"]: m["mountPath"] for m in _pod_spec(_docs(MONITORING)[("Deployment", "grafana")])["containers"][0]["volumeMounts"]}
    assert mounts["dashboards"] == provider["options"]["path"]
    # configMapGenerator names are what the Deployment's volumes reference
    gens = {g["name"] for g in _kustomization(MONITORING)["configMapGenerator"]}
    for v in _pod_spec(_docs(MONITORING)[("Deployment", "grafana")])["volumes"]:
        if "configMap" in v:
            assert v["configMap"]["name"] in gens


def test_grafana_password_comes_from_a_secret_not_from_git():
    env = _pod_spec(_docs(MONITORING)[("Deployment", "grafana")])["containers"][0]["env"]
    pw = next(e for e in env if e["name"] == "GF_SECURITY_ADMIN_PASSWORD")
    assert "secretKeyRef" in pw["valueFrom"] and "value" not in pw
    for f in K8S.rglob("*.yaml"):
        assert "kind: Secret\n" not in f.read_text(), f"{f} commits a Secret"


def test_dashboards_stay_correct_with_several_api_replicas():
    d = json.loads((REPO / "grafana/dashboards/telecom-anomaly-api.json").read_text())
    exprs = {t["expr"] for p in d["panels"] for t in p.get("targets", [])}
    # Gauges are reported by every replica; a bare selector would show one number per pod.
    assert "max(cells_tracked)" in exprs and "max(model_threshold_info)" in exprs
    assert "cells_tracked" not in exprs and "model_threshold_info" not in exprs


def test_drift_cronjob_runs_something_the_image_contains_and_reaches_the_pushgateway():
    cron = _docs(MONITORING)[("CronJob", "drift-check")]
    container = _pod_spec(cron)["containers"][0]
    script = container["command"][1]
    assert (REPO / script).exists()
    dockerfile = (REPO / "Dockerfile").read_text()
    assert f"COPY {script.split('/')[0]}/" in dockerfile, f"Dockerfile does not copy {script.split('/')[0]}/ into the image"
    url = container["command"][container["command"].index("--pushgateway-url") + 1]
    pg = _service_for("pushgateway")
    assert url == f"http://pushgateway:{pg['spec']['ports'][0]['port']}"
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"
    reports = {m["mountPath"] for m in container["volumeMounts"]}
    assert "/app/monitoring/reports" in reports       # drift_check writes its HTML report there; root fs is read-only


def test_drift_reference_file_is_produced_shipped_and_tracked():
    from models.common import REFERENCE_FILENAME
    assert "COPY models/ ./models" in (REPO / "Dockerfile").read_text()
    assert f"models/{REFERENCE_FILENAME}" in (REPO / "dvc.yaml").read_text()
    assert f"models/{REFERENCE_FILENAME}" in (REPO / ".gitignore").read_text()
    assert f"models/{REFERENCE_FILENAME}" in (REPO / ".github/workflows/ci.yml").read_text()
    assert "REFERENCE_FILENAME" in (REPO / "models/train.py").read_text()
    assert "REFERENCE_FILENAME" in (REPO / "monitoring/drift_check.py").read_text()


# ── CI and the e2e script ───────────────────────────────────────────────────

def test_ci_validates_and_deploys_the_manifests():
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    for d in ("base", "overlays/scaled", "monitoring", "overlays/full"):
        assert d in ci
    assert "kubeconform" in ci and "kind-action" in ci and "scripts/k8s_e2e.sh" in ci
    assert "REDIS_TEST_URL" in ci


def test_e2e_script_expects_what_the_manifests_define():
    sh = (REPO / "scripts/k8s_e2e.sh").read_text()
    uids = sorted(json.loads(p.read_text())["uid"] for p in (REPO / "grafana/dashboards").glob("*.json"))
    assert str(uids) in sh, f"e2e script's dashboard uids must be {uids}"
    for name in ("redis", "anomaly-api", "pushgateway", "prometheus", "grafana"):
        assert ("Deployment", name) in _all_docs()
    for secret in ("grafana-admin",):
        assert secret in sh and secret in (MONITORING / "grafana.yaml").read_text()
    assert "cronjob/drift-check" in sh and ("CronJob", "drift-check") in _all_docs()
    assert (REPO / "scripts/k8s_traffic.py").exists()


def test_api_keys_come_from_an_optional_secret_and_probes_need_no_key():
    ref = {e["name"]: e for e in _api_container()["env"]}["API_KEYS"]["valueFrom"]["secretKeyRef"]
    assert ref["optional"] is True            # no Secret -> open routes, pod still starts
    api_src = (REPO / "api" / "main.py").read_text()
    assert api_src.count("Depends(require_api_key)") == 3
    # probes hit /health and Prometheus hits /metrics: neither route may be key-protected
    for open_route in ('@app.get("/health")',):
        assert open_route in api_src


def test_ingress_exposes_only_api_paths_never_metrics():
    ing = _load(K8S / "optional" / "ingress.yaml")[0]
    rules = ing["spec"]["rules"][0]["http"]["paths"]
    svc_ports = {p["name"] for p in _service_for("anomaly-api")["spec"]["ports"]}
    for p in rules:
        assert p["backend"]["service"]["name"] == "anomaly-api" and p["backend"]["service"]["port"]["name"] in svc_ports
        assert not p["path"].startswith("/metrics")
    assert {p["path"] for p in rules} >= {"/predict", "/health"}
    assert ing["spec"]["tls"][0]["secretName"]


def test_grafana_state_is_on_a_volume_not_an_emptydir():
    docs = [d for d in _load(K8S / "monitoring" / "grafana.yaml")]
    dep = next(d for d in docs if d["kind"] == "Deployment")
    vol = next(v for v in dep["spec"]["template"]["spec"]["volumes"] if v["name"] == "data")
    claim = vol["persistentVolumeClaim"]["claimName"]
    assert any(d["kind"] == "PersistentVolumeClaim" and d["metadata"]["name"] == claim for d in docs)


def test_resource_requests_never_exceed_limits_and_are_all_set():
    def q(v):
        v = str(v)
        for suf, mult in (("Gi", 1024), ("Mi", 1), ("m", 0.001)):
            if v.endswith(suf):
                return float(v[: -len(suf)]) * mult
        return float(v)
    for kdir in (BASE, K8S / "monitoring"):
        for f in kdir.glob("*.yaml"):
            for d in _load(f):
                spec = d.get("spec", {})
                pod = spec.get("template", {}).get("spec") or spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec")
                for c in (pod or {}).get("containers", []):
                    r = c.get("resources", {})
                    assert r.get("requests") and r.get("limits"), f"{d['metadata']['name']}/{c['name']} missing resources"
                    for k in ("cpu", "memory"):
                        assert q(r["requests"][k]) <= q(r["limits"][k]), f"{d['metadata']['name']} {k}"
