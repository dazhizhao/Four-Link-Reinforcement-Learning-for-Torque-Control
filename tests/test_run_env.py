from pathlib import Path

import numpy as np

from scripts.run_env import save_rollout_npz


def test_save_rollout_npz_has_expected_structure(tmp_path: Path):
    output_path = save_rollout_npz(
        output_path=tmp_path / "run_env_rollout.npz",
        steps=[1, 2],
        rewards=[-1.0, -0.5],
        distances=[1.2, 0.8],
        terminated=[False, True],
        truncated=[False, False],
        end_effector_pos=[np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        target_pos=[np.array([3.0, 4.0]), np.array([3.0, 4.0])],
        joint_torques=[np.ones(4), np.ones(4) * 2.0],
        joint_power=[np.ones(4) * 3.0, np.ones(4) * 4.0],
        applied_action=[np.zeros(4), np.ones(4)],
        seed=7,
        policy="random",
    )

    data = np.load(output_path)
    assert set(data.files) == {
        "step",
        "reward",
        "distance_to_target",
        "terminated",
        "truncated",
        "end_effector_pos",
        "target_pos",
        "joint_torques",
        "joint_power",
        "applied_action",
        "seed",
        "policy",
    }
    assert data["step"].shape == (2,)
    assert data["end_effector_pos"].shape == (2, 2)
    assert data["joint_torques"].shape == (2, 4)
    assert data["joint_power"].shape == (2, 4)
    assert data["applied_action"].shape == (2, 4)
    assert int(data["seed"]) == 7
    assert str(data["policy"]) == "random"
