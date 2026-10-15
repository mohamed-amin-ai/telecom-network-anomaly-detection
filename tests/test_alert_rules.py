"""
test_alert_rules.py
--------------------
Offline checks for the Prometheus alert rules (prometheus/alert_rules.yml).

They cannot prove an alert fires (that needs a running Prometheus), but they catch the mistakes that
make alerting silently useless: a rule that names a metric the project never emits, a rules file the
Prometheus configs or the Kubernetes ConfigMap never load, a Kubernetes copy that diverged, a rule
with no summary for whoever gets paged.
"""

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
RULES = REPO / "prometheus" / "alert_rules.yml"
K8S_RULES = REPO / "k8s" / "monitoring" / "alert_rules.yml"
RULES_PATH_IN_CONTAINER = "/etc/prometheus/alert_rules.yml"

# Names exposed by prometheus-fastapi-instrumentator (api/metrics.py wires it up) and by Prometheus
# itself. They are not defined in this repository's code.
BUILT_IN_METRICS = {
    "up",
    "http_requests_total",
    "http_request_duration_seconds_bucket",
}
# PromQL words that appear outside braces and brackets: functions, operators, grouping keywords.
PROMQL_WORDS = {"sum", "rate", "increase", "histogram_quantile", "time", "by", "and", "or", "unless", "le"}


def _rules():
    doc = yaml.safe_load(RULES.read_text())
    return [(g["name"], r) for g in doc["groups"] for r in g["rules"]]


def _metrics_the_project_emits() -> set[str]:
    names = set(BUILT_IN_METRICS)
    api = (REPO / "api" / "metrics.py").read_text()
    for kind, name in re.findall(r'\b(Counter|Gauge|Histogram)\(\s*"([a-z_]+)"', api):
        names.add(name)
        if kind == "Histogram":
            names.add(name + "_bucket")
    drift = (REPO / "monitoring" / "drift_check.py").read_text()
    names.update(re.findall(r'\bGauge\(\s*"([a-z_]+)"', drift))
    return names


def _metric_names_in(expr: str) -> set[str]:
    expr = re.sub(r"\{[^}]*\}", "", expr)       # label matchers
    expr = re.sub(r"\[[^\]]*\]", "", expr)      # range selectors
    words = set(re.findall(r"[A-Za-z_:][A-Za-z0-9_:]*", expr))
    return words - PROMQL_WORDS


def test_every_rule_has_the_fields_an_on_call_person_needs():
    rules = _rules()
    assert len(rules) >= 8
    names = [r["alert"] for _, r in rules]
    assert len(names) == len(set(names)), "alert names must be unique"
    for group, rule in rules:
        where = f"{group}/{rule['alert']}"
        assert rule["expr"].strip(), where
        assert rule["labels"]["severity"] in {"warning", "critical"}, where
        assert rule["annotations"]["summary"].strip(), where
        assert rule["annotations"]["description"].strip(), where
        assert "for" in rule, f"{where}: state explicitly how long the condition must hold (0m is allowed)"


def test_expressions_and_templates_are_balanced():
    pairs = {"(": ")", "[": "]", "{": "}"}
    for _, rule in _rules():
        for text in (rule["expr"], *rule["annotations"].values()):
            stack = []
            for ch in re.sub(r'"[^"]*"', '""', text):
                if ch in pairs:
                    stack.append(pairs[ch])
                elif ch in pairs.values():
                    assert stack and stack.pop() == ch, f"{rule['alert']}: unbalanced {ch!r} in {text!r}"
            assert not stack, f"{rule['alert']}: unclosed bracket in {text!r}"
            assert text.count("{{") == text.count("}}"), f"{rule['alert']}: unbalanced template in {text!r}"


def test_rules_only_use_metrics_the_project_emits():
    known = _metrics_the_project_emits()
    for _, rule in _rules():
        unknown = _metric_names_in(rule["expr"]) - known
        assert not unknown, f"{rule['alert']} uses metrics nothing emits: {sorted(unknown)}"


def test_drift_rules_use_the_job_label_the_drift_check_pushes():
    pushed = re.search(r'push_to_gateway\([^)]*job="([a-z_]+)"', (REPO / "monitoring" / "drift_check.py").read_text())
    assert pushed, "could not find the job name drift_check.py pushes under"
    for _, rule in _rules():
        if "data_drift" in rule["expr"] or "drift_check_last_run" in rule["expr"]:
            assert f'job="{pushed.group(1)}"' in rule["expr"], rule["alert"]


def test_both_prometheus_configs_and_the_deployments_load_the_rules():
    for cfg in (REPO / "prometheus" / "prometheus.yml", REPO / "k8s" / "monitoring" / "prometheus.yml"):
        assert RULES_PATH_IN_CONTAINER in yaml.safe_load(cfg.read_text())["rule_files"], cfg

    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    volumes = compose["services"]["prometheus"]["volumes"]
    assert f"./prometheus/alert_rules.yml:{RULES_PATH_IN_CONTAINER}:ro" in volumes

    kustomization = yaml.safe_load((REPO / "k8s" / "monitoring" / "kustomization.yaml").read_text())
    generators = {g["name"]: g["files"] for g in kustomization["configMapGenerator"]}
    assert "alert_rules.yml" in generators["prometheus-config"]


def test_kubernetes_copy_of_the_rules_is_identical():
    assert K8S_RULES.read_text() == RULES.read_text(), "k8s/monitoring/alert_rules.yml must match prometheus/alert_rules.yml"


def test_the_api_dashboard_shows_firing_alerts():
    for path in (REPO / "grafana" / "dashboards", REPO / "k8s" / "monitoring" / "grafana" / "dashboards"):
        panels = json.loads((path / "telecom-anomaly-api.json").read_text())["panels"]
        exprs = [t["expr"] for p in panels for t in p.get("targets", [])]
        assert any('ALERTS{alertstate="firing"}' in e for e in exprs), path
