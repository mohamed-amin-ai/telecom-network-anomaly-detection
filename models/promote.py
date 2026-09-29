"""
promote.py
----------
MLflow model-registry promotion logic for the Airflow DAG
(airflow/dags/telecom_anomaly_pipeline.py). Pulled out of the DAG file
so `should_promote` — the actual decision — is unit-testable without
either Airflow or a live MLflow tracking server, and so
`promote_latest_model`'s MLflow I/O is isolated to one place.

═══════════════════════════════════════════════════════════════════════
CAVEAT
═══════════════════════════════════════════════════════════════════════
`promote_latest_model` uses `MlflowClient.get_latest_versions()` and
`transition_model_version_stage()` — MLflow's "stages" model-registry
API (None → Staging → Production). MLflow has been moving toward a
newer "aliases" API since ~2.9, and stages are soft-deprecated: still
functional, but liable to emit deprecation warnings, and not guaranteed
to exist forever. This was written without network access to confirm
current behavior against the exact pinned version in requirements.txt
(mlflow==2.14.1) — if `transition_model_version_stage` starts raising
rather than just warning, migrate to `client.set_registered_model_alias`
and `client.get_model_version_by_alias` instead; the decision logic in
`should_promote` doesn't change either way.
═══════════════════════════════════════════════════════════════════════

Which recall is checked: the `recall` metric logged by models/train.py is
measured on the held-out TEST window at a threshold tuned on the separate
validation slice. It is NOT tuned to hit the floor, so it will often fall
below `threshold.min_recall` on small synthetic data (see EXPERIMENTS.md) —
in which case this declines to promote. That is the intended, honest gate.

Known gap this doesn't solve (documented in README's Phase 5 section):
promoting a version to "Production" in the MLflow registry has NO EFFECT
on what api/main.py actually serves — that code loads models/model.json
directly off disk, written unconditionally by every models/train.py run,
regardless of registry stage. Wiring the API to load from the registry
(and reload on promotion) is real future work, not done here.
"""


def should_promote(recall: float | None, min_recall: float) -> bool:
    """Pure decision logic, no MLflow I/O — trivially unit-testable.
    Deliberately has no mlflow import anywhere in its call path, so it's
    importable and testable in environments without mlflow installed
    (e.g. the sandbox this project was partly built in)."""
    if recall is None:
        return False
    return recall >= min_recall


def promote_latest_model(model_name: str, min_recall: float) -> dict:
    """
    Finds the most recent not-yet-Production version of `model_name`,
    checks its logged 'recall' metric against `min_recall`, and promotes
    it to Production (archiving whatever was there before) if it clears
    the bar.

    Returns a small dict describing the outcome — never raises on a
    "didn't promote" result, only on genuine MLflow errors (e.g. can't
    reach the tracking server), so callers can log the reason without
    wrapping every call in a try/except.
    """
    # Imported here, not at module level, so `should_promote` above stays
    # importable/testable without mlflow installed at all.
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    candidates = client.get_latest_versions(model_name, stages=["None", "Staging"])
    if not candidates:
        return {"promoted": False, "reason": "no_candidate_versions"}

    latest = max(candidates, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    recall = run.data.metrics.get("recall")

    if not should_promote(recall, min_recall):
        return {
            "promoted": False,
            "reason": "below_recall_floor",
            "version": latest.version,
            "recall": recall,
            "min_recall": min_recall,
        }

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return {"promoted": True, "version": latest.version, "recall": recall}
