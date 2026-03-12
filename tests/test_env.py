import numpy as np

from env.bridge_robot_env import (
    BridgeRobotEnv,
    EnvConfig,
    RewardConfig,
    SimConfig,
    TaskConfig,
)


def make_static_env(*, success_hold_steps: int = 3, success_tolerance: float = 1e-6, max_steps: int = 10):
    env = BridgeRobotEnv()
    config = EnvConfig(
        sim=SimConfig(
            dt=env.config.sim.dt,
            max_steps=max_steps,
            gravity=0.0,
            integrator=env.config.sim.integrator,
        ),
        robot=env.config.robot,
        task=TaskConfig(
            home_pose=env.config.task.home_pose,
            target_sampling=env.config.task.target_sampling,
            success_tolerance=success_tolerance,
            success_hold_steps=success_hold_steps,
        ),
        reward=RewardConfig(
            distance_weight=env.config.reward.distance_weight,
            torque_weight=env.config.reward.torque_weight,
            motion_weight=env.config.reward.motion_weight,
            smoothness_weight=env.config.reward.smoothness_weight,
            hold_bonus_weight=env.config.reward.hold_bonus_weight,
            success_bonus=env.config.reward.success_bonus,
        ),
        render=env.config.render,
        io=env.config.io,
    )
    return BridgeRobotEnv(config=config)


def test_reset_seed_reproduces_target():
    env_a = BridgeRobotEnv()
    env_b = BridgeRobotEnv()

    obs_a = env_a.reset(seed=11)
    obs_b = env_b.reset(seed=11)

    assert np.allclose(obs_a["target_pos"], obs_b["target_pos"])


def test_step_returns_expected_fields():
    env = BridgeRobotEnv()
    observation = env.reset(seed=3)
    assert set(observation.keys()) == {
        "q",
        "qd",
        "qdd",
        "end_effector_pos",
        "end_effector_vel",
        "target_pos",
        "distance_to_target",
        "joint_torques",
        "joint_power",
        "applied_action_norm",
        "consecutive_success_steps",
        "hold_progress",
        "success_ready",
    }

    result = env.step(np.zeros(4, dtype=float))
    assert set(result.info.keys()) >= {
        "reward_terms",
        "action_clipped",
        "success",
        "success_ready",
        "workspace_violation",
        "consecutive_success_steps",
        "hold_progress",
        "applied_action_norm",
    }


def test_terminated_only_after_hold_steps():
    env = make_static_env(success_hold_steps=3, success_tolerance=1e-6)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()

    result_a = env.step(np.zeros(4, dtype=float))
    result_b = env.step(np.zeros(4, dtype=float))
    result_c = env.step(np.zeros(4, dtype=float))

    assert result_a.terminated is False
    assert result_b.terminated is False
    assert result_c.terminated is True
    assert result_c.info["consecutive_success_steps"] == 3
    assert np.isclose(result_c.info["hold_progress"], 1.0)


def test_hold_counter_resets_when_leaving_tolerance_ball():
    env = make_static_env(success_hold_steps=3, success_tolerance=1e-9)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()

    inside_result = env.step(np.zeros(4, dtype=float))
    outside_result = env.step(np.array([env.config.robot.torque_limits[0], 0.0, 0.0, 0.0], dtype=float))

    assert inside_result.info["consecutive_success_steps"] == 1
    assert outside_result.info["consecutive_success_steps"] == 0
    assert outside_result.info["success_ready"] is False


def test_truncated_at_max_steps_without_false_success():
    env = make_static_env(success_hold_steps=5, max_steps=1)
    env.reset(seed=5)
    result = env.step(np.zeros(4, dtype=float))

    assert result.truncated is True
    assert result.terminated is False


def test_reward_breakdown_uses_motion_and_hold_not_power():
    env = make_static_env(success_hold_steps=2, success_tolerance=1e-6)
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()
    result = env.step(np.zeros(4, dtype=float))

    reward_terms = result.info["reward_terms"]
    assert "motion_penalty" in reward_terms
    assert "hold_bonus" in reward_terms
    assert "power_penalty" not in reward_terms
