"""
test_repo_consistency.py
--------------------------
Cheap, dependency-light checks that files written in different phases
still agree with each other. This repo was built incrementally, and the
failure mode is drift: a script imports a package nobody added to
requirements, a test file exists but CI never runs it, two services claim
the same host port, a config hardcodes a number the code no longer
produces. Each check below is a real assertion against the real files.
"""

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

LOCAL_PACKAGES = {"api", "data", "models", "monitoring", "scripts", "tests"}
# Import name -> distribution name, where they differ.
ALIASES = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
# Third-party modules that are deliberately NOT in requirements.txt.
#   airflow: provided by the apache/airflow base image (Dockerfile.airflow)
NOT_IN_REQUIREMENTS = {"airflow"}


def _req_names(path: Path) -> set:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def _imported_top_level_modules(py_file: Path) -> set:
    tree = ast.parse(py_file.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _python_files():
    return [p for p in REPO.rglob("*.py") if ".git" not in p.parts]


def test_every_third_party_import_is_declared_in_requirements():
    runtime = _req_names(REPO / "requirements.txt")
    dev = runtime | _req_names(REPO / "requirements-dev.txt")
    missing = []
    for py in _python_files():
        allowed = dev if "tests" in py.relative_to(REPO).parts else runtime
        for mod in _imported_top_level_modules(py):
            if mod in sys.stdlib_module_names or mod in LOCAL_PACKAGES or mod in NOT_IN_REQUIREMENTS:
                continue
            dist = ALIASES.get(mod, mod).lower().replace("_", "-")
            if dist not in allowed:
                missing.append(f"{py.relative_to(REPO)} imports '{mod}' (expected '{dist}' in requirements)")
    assert not missing, "Undeclared third-party imports:\n  " + "\n  ".join(missing)


def _test_files_named_in(text: str) -> set:
    return set(re.findall(r"tests/(test_\w+\.py)", text))


def test_ci_and_makefile_reference_only_existing_test_files():
    existing = {p.name for p in (REPO / "tests").glob("test_*.py")}
    for name in (".github/workflows/ci.yml", "Makefile"):
        referenced = _test_files_named_in((REPO / name).read_text())
        assert referenced <= existing, f"{name} references missing test files: {referenced - existing}"


def test_no_test_file_is_orphaned_from_ci():
    existing = {p.name for p in (REPO / "tests").glob("test_*.py")}
    in_ci = _test_files_named_in((REPO / ".github/workflows/ci.yml").read_text())
    orphaned = existing - in_ci
    assert not orphaned, f"Test files that CI never runs: {sorted(orphaned)}"


def test_ci_and_makefile_run_the_same_unit_tests():
    ci = _test_files_named_in((REPO / ".github/workflows/ci.yml").read_text())
    mk = _test_files_named_in((REPO / "Makefile").read_text())
    assert ci == mk, f"CI and Makefile disagree on which tests to run: {ci ^ mk}"


def test_dvc_tracked_params_exist_in_params_yaml():
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    stages = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]
    for stage, cfg in stages.items():
        for key in cfg.get("params", []):
            node = params
            for part in key.split("."):
                assert part in node, f"dvc stage '{stage}' tracks missing param '{key}'"
                node = node[part]


def test_dvc_dependencies_exist_and_train_stage_tracks_its_local_imports():
    stages = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]
    for stage, cfg in stages.items():
        for dep in cfg.get("deps", []):
            # generated data files legitimately don't exist until the pipeline runs
            if dep.endswith(".csv"):
                continue
            assert (REPO / dep).exists(), f"dvc stage '{stage}' depends on missing file {dep}"
    train_deps = set(stages["train"]["deps"])
    assert {"models/common.py", "data/features.py"} <= train_deps, (
        "train.py imports models/common.py and data/features.py; DVC must track them "
        "or it won't retrain when they change"
    )


def test_host_ports_are_unique_across_all_compose_files():
    seen = {}
    for name in ("docker-compose.yml", "docker-compose.airflow.yml", "docker-compose.triton.yml"):
        services = yaml.safe_load((REPO / name).read_text())["services"]
        for svc, cfg in services.items():
            for mapping in cfg.get("ports", []):
                host_port = str(mapping).split(":")[0]
                assert host_port not in seen, (
                    f"host port {host_port} claimed by both {seen[host_port]} and {name}:{svc}"
                )
                seen[host_port] = f"{name}:{svc}"


def test_drift_check_flags_used_by_ci_and_dag_exist():
    source = (REPO / "monitoring/drift_check.py").read_text()
    declared = set(re.findall(r'add_argument\("(--[\w-]+)"', source))
    for path in (".github/workflows/ci.yml", "airflow/dags/telecom_anomaly_pipeline.py"):
        text = (REPO / path).read_text()
        for line in text.splitlines():
            if "drift_check.py" in line or line.strip().startswith('"--'):
                for flag in re.findall(r"--[\w-]+", line):
                    assert flag in declared, f"{path} uses drift_check flag {flag}, not declared in drift_check.py"


def _real_feature_count() -> int:
    from data.features import engineer_features, get_feature_columns

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "cell_id": "C",
        "latency_ms": 40 + rng.normal(0, 3, n),
        "throughput_mbps": 100 + rng.normal(0, 8, n),
        "packet_loss_pct": np.abs(rng.normal(0.5, 0.2, n)),
        "sinr_db": 15 + rng.normal(0, 2, n),
        "connected_users": rng.integers(10, 90, n),
        "anomaly": 0,
    })
    return len(get_feature_columns(engineer_features(df, verbose=False)))


def test_triton_config_dims_match_real_feature_count():
    raw = (REPO / "triton_model_repository/telecom_anomaly_xgb/config.pbtxt").read_text()
    # Strip comment lines first — the header prose mentions "dims: [2]" and
    # must not be mistaken for the real declaration.
    text = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#"))
    dims = [int(m) for m in re.findall(r"dims:\s*\[\s*(\d+)\s*\]", text)]
    assert len(dims) == 2, f"expected one input and one output dims declaration, found {dims}"
    n_features = _real_feature_count()
    assert dims[0] == n_features, (
        f"config.pbtxt input dims {dims[0]} != feature count {n_features} — Triton would reject every request"
    )
    assert dims[1] == 2, "output dims should be [2] (binary class probabilities, zipmap disabled)"


def _slug(heading: str) -> str:
    """GitHub's heading-anchor rule: lowercase, drop punctuation, spaces -> hyphens."""
    heading = heading.strip().lower()
    heading = re.sub(r"[^a-z0-9 _-]", "", heading)
    return heading.replace(" ", "-")


def test_readme_internal_anchor_links_resolve():
    text = (REPO / "README.md").read_text()
    in_code = False
    anchors = set()
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,6} ", line):
            anchors.add(_slug(re.sub(r"^#+\s*", "", line)))
    links = set(re.findall(r"\]\(#([^)]+)\)", text))
    broken = links - anchors
    assert not broken, f"README links to missing anchors: {sorted(broken)}"


def test_model_registry_name_and_recall_floor_used_by_code_exist_in_params():
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    assert "registered_model_name" in params["mlflow"]
    assert "min_recall" in params["threshold"]
    assert "val_fraction" in params["split"] and "train_ratio" in params["split"]
