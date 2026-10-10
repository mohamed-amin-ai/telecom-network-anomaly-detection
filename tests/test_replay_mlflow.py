"""The sweep replay script, against a recording stand-in for mlflow (real mlflow isn't needed to test the logic)."""
import sys
import types
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def _fake_mlflow():
    calls = {"experiment": None, "runs": [], "cur": None}
    m = types.ModuleType("mlflow")
    m.set_experiment = lambda name: calls.__setitem__("experiment", name)

    @contextmanager
    def start_run(run_name=None):
        calls["cur"] = {"name": run_name, "params": {}, "metrics": {}, "tags": {}}
        yield
        calls["runs"].append(calls["cur"])
    m.start_run = start_run
    m.set_tags = lambda t: calls["cur"]["tags"].update(t)
    m.log_params = lambda p: calls["cur"]["params"].update(p)
    m.log_metrics = lambda p: calls["cur"]["metrics"].update(p)
    return m, calls


def test_every_sweep_run_is_logged_with_its_metrics_and_the_benchmark_setting():
    import json
    import importlib
    fake, calls = _fake_mlflow()
    saved = sys.modules.get("mlflow")
    sys.modules["mlflow"] = fake
    try:
        sys.modules.pop("scripts.replay_experiments_to_mlflow", None)
        mod = importlib.import_module("scripts.replay_experiments_to_mlflow")
        mod.main()
    finally:
        sys.modules.pop("scripts.replay_experiments_to_mlflow", None)
        if saved is not None:
            sys.modules["mlflow"] = saved
        else:
            sys.modules.pop("mlflow", None)
    results = json.loads((REPO / "results" / "experiments.json").read_text())
    assert [r["name"] for r in calls["runs"]] == [r["run_name"] for r in results["runs"]]
    for logged, src in zip(calls["runs"], results["runs"]):
        assert logged["metrics"]["recall"] == src["metrics"]["recall"]
        assert logged["tags"]["is_xgboost_stand_in"] == "true"
        assert logged["params"]["data_seed"] == 42 and logged["params"]["data_periods_days"] == 730
    assert calls["experiment"].endswith("-sweep")
