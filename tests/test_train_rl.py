from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest


def test_train_rl_smoke_creates_expected_outputs(tmp_path: Path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/train_rl.py",
        "--total-timesteps",
        "32",
        "--eval-episodes",
        "1",
        "--eval-freq",
        "16",
        "--run-name",
        "smoke",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    run_dirs = list((output_dir / "rl_torque_control").glob("smoke_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    artifacts_dir = run_dir / "artifacts"

    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "progress.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (artifacts_dir / "best_rollout.npz").exists()
    assert (artifacts_dir / "best_pose.png").exists()
    assert (artifacts_dir / "best_rollout.mp4").exists()
    assert (artifacts_dir / "best_joint_torques.png").exists()
    assert (artifacts_dir / "training_curves.png").exists()
    assert (artifacts_dir / "best_rollout.mp4").stat().st_size > 0
    assert (artifacts_dir / "best_joint_torques.png").stat().st_size > 0

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["resolved"]["resolved_run_name"].startswith("smoke_")
    evaluation = summary["evaluation"]
    periodic_evaluations = summary["periodic_evaluations"]
    assert summary["resolved"]["eval_freq"] == 16
    assert set(evaluation.keys()) >= {
        "success_rate",
        "mean_reward",
        "mean_final_distance",
        "mean_episode_length",
        "mean_torque_norm",
        "mean_motion_penalty",
        "mean_smoothness_penalty",
    }
    assert "rollout_video" in evaluation["artifact_paths"]
    assert evaluation["artifact_paths"]["rollout_video"].endswith("best_rollout.mp4")
    assert evaluation["artifact_paths"]["joint_torque_plot"].endswith("best_joint_torques.png")
    assert evaluation["artifact_paths"]["best_model"].endswith("best_model.zip")
    assert evaluation["evaluation_source"] == "best_model.zip"
    assert periodic_evaluations

    checkpoint_dirs = sorted((artifacts_dir / "evals").glob("step_*"))
    assert checkpoint_dirs
    checkpoint_dir = checkpoint_dirs[0]
    checkpoint_summary = json.loads((checkpoint_dir / "summary.json").read_text(encoding="utf-8"))
    assert checkpoint_summary["artifact_paths"]["rollout_video"].endswith("best_rollout.mp4")
    assert checkpoint_summary["artifact_dir"] == str(checkpoint_dir)
    assert (checkpoint_dir / "best_rollout.mp4").exists()
    assert (checkpoint_dir / "best_rollout.mp4").stat().st_size > 0

    samples = np.load(artifacts_dir / "best_rollout.npz")
    assert set(samples.files) >= {"step", "reward", "distance_to_target", "applied_action"}
