import numpy as np

from env.bridge_robot_env import BridgeRobotEnv, EnvConfig, SimConfig


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
    }

    result = env.step(np.zeros(4, dtype=float))
    assert set(result.info.keys()) >= {
        "reward_terms",
        "action_clipped",
        "success",
        "workspace_violation",
    }


def test_terminated_when_target_is_reached():
    env = BridgeRobotEnv()
    obs = env.reset(seed=5)
    env.state.target_pos = obs["end_effector_pos"].copy()
    result = env.step(np.zeros(4, dtype=float))
    assert result.terminated is True


def test_truncated_at_max_steps():
    env = BridgeRobotEnv()
    env.reset(seed=5)
    env.config = EnvConfig(
        sim=SimConfig(
            dt=env.config.sim.dt,
            max_steps=1,
            gravity=env.config.sim.gravity,
            integrator=env.config.sim.integrator,
        ),
        robot=env.config.robot,
        task=env.config.task,
        reward=env.config.reward,
        render=env.config.render,
        io=env.config.io,
    )
    result = env.step(np.zeros(4, dtype=float))
    assert result.truncated is True
