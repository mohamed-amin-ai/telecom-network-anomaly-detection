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
PROMQL_WORDS = {"sum", "rate", "increase", "histogram_quantile", "time", "by", "and", "or", "unless", "le", "offset"}


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
    expr = re.sub(r"offset\s+\w+", "offset", expr)  # offset 5m
    words = set(re.findall(r"[A-Za-z_:][A-Za-z0-9_:]*", expr))
    return words - PROMQL_WORDS


def test_every_rule_has_the_fields_an_on_call_person_needs():
    rules = _rules()
    assert len(rules) >= 10
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


def test_every_detected_anomaly_alerts_per_cell_and_catches_a_new_cells_first_count():
    by_name = {r["alert"]: r for _, r in _rules()}
    for name, sev in (("AnomalyDetectedCritical", "critical"), ("AnomalyDetectedWarning", "warning")):
        rule = by_name[name]
        assert f'anomaly_detections_total{{severity="{sev}"}}' in rule["expr"] and rule["for"] == "0m"
        assert "increase(" in rule["expr"] and " unless " in rule["expr"] and " offset " in rule["expr"]
        assert "$labels.cell_id" in rule["annotations"]["summary"]
    for sev in ("normal", "warning", "critical"):   # severities the API emits are the ones the rules match
        assert f'"{sev}"' in (REPO / "api" / "main.py").read_text()


def test_cell_label_is_bounded_so_clients_cannot_grow_prometheus_memory():
    from api import metrics

    metrics._labelled_cells.clear()
    assert metrics._cell_label("CELL_001") == "CELL_001"
    assert metrics._cell_label("not a valid id!") == "other"
    assert metrics._cell_label("x" * 65) == "other"
    for i in range(metrics.MAX_CELL_LABELS + 50):
        metrics._cell_label(f"C{i}")
    assert len(metrics._labelled_cells) == metrics.MAX_CELL_LABELS
    assert metrics._cell_label("brand-new-after-cap") == "other"
    assert metrics._cell_label("CELL_001") == "CELL_001"     # already-known cells keep their label
    metrics._labelled_cells.clear()


def test_only_flagged_readings_are_counted_per_cell():
    from api import metrics

    metrics._labelled_cells.clear()

    def before(cell, severity):
        return metrics.ANOMALY_DETECTIONS_TOTAL.labels(cell_id=cell, severity=severity)._value.get()

    base = before("T_CELL", "critical")
    metrics.record_prediction("critical", "live", 0.99, cell_id="T_CELL")
    metrics.record_prediction("normal", "live", 0.01, cell_id="T_CELL_NORMAL")
    assert before("T_CELL", "critical") == base + 1
    assert "T_CELL_NORMAL" not in metrics._labelled_cells      # normal readings never get a per-cell series
    metrics._labelled_cells.clear()



def test_the_api_dashboard_has_a_live_view_and_a_history_view_of_flagged_cells():
    assert "anomaly_detections_total" in _metrics_the_project_emits()
    for path in (REPO / "grafana" / "dashboards", REPO / "k8s" / "monitoring" / "grafana" / "dashboards"):
        panels = json.loads((path / "telecom-anomaly-api.json").read_text())["panels"]
        rows = [p["title"] for p in panels if p["type"] == "row"]
        assert any(t.startswith("Live") for t in rows) and any(t.startswith("History") for t in rows), path
        by_title = {p["title"]: p for p in panels}

        live = " ".join(t["expr"] for t in by_title["Cells With Anomalies (last hour)"]["targets"])
        assert "anomaly_detections_total" in live and "by (cell_id" in live
        for window in ("[1h]", "[7d]", "[15d]"):   # fixed windows, not the picker: flagged now, and how often before
            assert window in live, window

        for title in ("Most-Flagged Cells (selected time range)", "Flagged Readings (trailing hour), Top 10 Cells"):
            assert "anomaly_detections_total" in by_title[title]["targets"][0]["expr"], title
        history = by_title["Most-Flagged Cells (selected time range)"]["targets"][0]["expr"]
        assert "$__range" in history and "by (cell_id" in history   # follows the time picker
        y_live = by_title["Cells With Anomalies (last hour)"]["gridPos"]["y"]
        y_hist = by_title["Most-Flagged Cells (selected time range)"]["gridPos"]["y"]
        assert y_live < y_hist
