"""
test_generate_data.py
-----------------------
Regression tests for data/generate_data.py's determinism.

Background: the generator originally seeded each cell's RNG from
`hash(cell) % 1000`. Python randomizes str hash() per process
(PYTHONHASHSEED), so every fresh run silently produced a DIFFERENT
dataset despite a pinned seed — which undermined every "reproducible"
claim downstream (DVC, the experiment log, the Airflow DAG docstring).
It was only caught by rerunning the script and noticing the anomaly
count changed. These tests make that class of bug fail loudly.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

_SNIPPET = (
    "import sys; sys.path.insert(0, %r);"
    "import pandas as pd;"
    "from data.generate_data import build_dataset;"
    "df = build_dataset();"
    "print(int(pd.util.hash_pandas_object(df, index=True).sum()) %% (2**63), len(df), int(df['anomaly'].sum()))"
) % str(REPO_ROOT)


def _run_with_hash_seed(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=True,
    )
    return result.stdout.strip()


def test_dataset_identical_across_processes_with_different_hash_seeds():
    """The core regression: two separate processes, deliberately given
    different string-hash seeds, must produce byte-identical datasets."""
    outputs = {seed: _run_with_hash_seed(seed) for seed in ("1", "2", "12345")}
    assert len(set(outputs.values())) == 1, (
        f"Dataset differs across PYTHONHASHSEED values — something is "
        f"seeding from hash() or other per-process state: {outputs}"
    )


def test_build_dataset_is_idempotent_within_one_process():
    """The anomaly injectors use a module-level rng; without reseeding in
    build_dataset(), a second call would continue the first call's random
    stream and return different data."""
    sys.path.insert(0, str(REPO_ROOT))
    import pandas as pd
    from data.generate_data import build_dataset

    first = build_dataset()
    second = build_dataset()
    pd.testing.assert_frame_equal(first, second)
