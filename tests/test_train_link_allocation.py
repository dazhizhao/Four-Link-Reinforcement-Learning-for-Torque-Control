from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest


def test_train_link_allocation_smoke_creates_expected_outputs(tmp_path: Path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/train_link_allocation.py",
        "--total-timesteps",
        "32",
        "--eval-episodes",
        "2",
        "--run-name",
        "smoke",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    run_dirs = list((output_dir / "rl_link_alloc").glob("smoke_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    artifacts_dir = run_dir / "artifacts"

    assert (run_dir / "model_final.zip").exists()
    assert (run_dir / "progress.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (artifacts_dir / "best_workspace_samples.npz").exists()
    assert (artifacts_dir / "best_workspace.png").exists()
    assert (artifacts_dir / "best_workspace.mp4").exists()
    assert (artifacts_dir / "training_curves.png").exists()
    assert (artifacts_dir / "best_workspace.mp4").stat().st_size > 0

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["resolved"]["resolved_run_name"].startswith("smoke_")
    assert "best_lengths" in summary["evaluation"]

    samples = np.load(artifacts_dir / "best_workspace_samples.npz")
    assert set(samples.files) >= {"points", "lengths", "occupied_ratio", "xy_bounds"}
