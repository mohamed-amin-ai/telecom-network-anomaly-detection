"""
test_k8s_manifests.py
----------------------
Offline checks that the Kubernetes manifests agree with the rest of the repo.
They cannot prove a deploy works (CI's kind job does that); they catch the drift
that breaks one: a selector that matches no pod, a probe path the API doesn't
serve, a port the image doesn't listen on, a kustomization naming a missing file.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
K8S = REPO / "k8s"
BASE = K8S / "base"


def _load(path: Path) -> list:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _kustomization(directory: Path) -> dict:
    return yaml.safe_load((directory / "kustomization.yaml").read_text())


def _base_docs() -> dict:
    return {d["kind"]: d for f in BASE.glob("*.yaml") if f.name != "kustomization.yaml" for d in _load(f)}


def test_every_kustomization_resource_exists():
    for kdir in (BASE, K8S / "overlays" / "scaled"):
        for res in _kustomization(kdir).get("resources", []):
            assert (kdir / res).exists(), f"{kdir.name}/kustomization.yaml lists missing {res}"


def test_every_base_file_is_in_the_kustomization():
    listed = set(_kustomization(BASE)["resources"])
    on_disk = {f.name for f in BASE.glob("*.yaml")} - {"kustomization.yaml"}
    assert listed == on_disk, listed ^ on_disk


def test_selectors_match_pod_labels_and_service_targets_them():
    docs = _base_docs()
    dep, svc = docs["Deployment"], docs["Service"]
    pod_labels = dep["spec"]["template"]["metadata"]["labels"]
    for k, v in dep["spec"]["selector"]["matchLabels"].items():
        assert pod_labels.get(k) == v
    for k, v in svc["spec"]["selector"].items():
        assert pod_labels.get(k) == v


def test_container_port_matches_dockerfile_and_service():
    docs = _base_docs()
    container = docs["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    port = container["ports"][0]
    exposed = int(re.search(r"^EXPOSE\s+(\d+)", (REPO / "Dockerfile").read_text(), re.M).group(1))
    assert port["containerPort"] == exposed
    svc_port = docs["Service"]["spec"]["ports"][0]
    assert svc_port["targetPort"] == port["name"]


def test_probes_hit_an_endpoint_the_api_serves_and_uses_the_named_port():
    container = _base_docs()["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    api_src = (REPO / "api" / "main.py").read_text()
    for name in ("startupProbe", "readinessProbe", "livenessProbe"):
        probe = container[name]["httpGet"]
        assert f'"{probe["path"]}"' in api_src, f"{name} path {probe['path']} not served by api/main.py"
        assert probe["port"] == container["ports"][0]["name"]


def test_prometheus_annotations_match_the_metrics_endpoint():
    ann = _base_docs()["Deployment"]["spec"]["template"]["metadata"]["annotations"]
    assert ann["prometheus.io/path"] == "/metrics"
    # prometheus-fastapi-instrumentator exposes /metrics (see api/metrics.py, README Phase 3)
    assert "/metrics" in (REPO / "api" / "metrics.py").read_text() + (REPO / "api" / "main.py").read_text()


def test_image_name_matches_what_ci_pushes():
    image = _base_docs()["Deployment"]["spec"]["template"]["spec"]["containers"][0]["image"]
    name = image.rsplit(":", 1)[0]
    assert _kustomization(BASE)["images"][0]["name"] == name
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    assert "ghcr.io/${{ github.repository }}" in ci
    remote = re.search(r"mohamed-amin-ai/telecom-network-anomaly-detection", name)
    assert remote, "image path should be the lower-cased GitHub repository"


def test_log_mount_matches_where_the_api_writes():
    container = _base_docs()["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert "/app/logs" in mounts      # api/inference_log.py: LOG_DIR = <repo root>/logs, WORKDIR /app
    assert "/tmp" in mounts           # read-only root filesystem needs scratch space


def test_pod_runs_as_the_images_non_root_uid():
    spec = _base_docs()["Deployment"]["spec"]["template"]["spec"]
    uid = int(re.search(r"--uid\s+(\d+)", (REPO / "Dockerfile").read_text()).group(1))
    assert spec["securityContext"]["runAsUser"] == uid
    assert spec["securityContext"]["runAsNonRoot"] is True


def test_base_stays_single_replica_while_history_is_in_process_memory():
    # api/live_features.py documents that history is per-process. If this changes, the
    # live-feature design must change first (shared store), and this test should be updated with it.
    assert _base_docs()["Deployment"]["spec"]["replicas"] == 1
    assert "in-process memory" in (REPO / "api" / "live_features.py").read_text()


def test_ci_validates_and_deploys_the_manifests():
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    assert "kubectl kustomize k8s/base" in ci and "kubectl kustomize k8s/overlays/scaled" in ci
    assert "kubeconform" in ci and "kind-action" in ci


def test_scaled_overlay_patches_target_what_exists():
    overlay = _kustomization(K8S / "overlays" / "scaled")
    # The JSON patch replaces volumes/0 by index, so the log volume must stay first in the base.
    assert _base_docs()["Deployment"]["spec"]["template"]["spec"]["volumes"][0]["name"] == "logs"
    assert overlay["namespace"] == _kustomization(BASE)["namespace"]
    targets = {p["target"]["name"] for p in overlay["patches"]}
    assert targets == {"anomaly-api", "anomaly-api-logs"}
