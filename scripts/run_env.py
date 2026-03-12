from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.bridge_robot_env import BridgeRobotEnv, EnvConfig, IOConfig


def save_rollout_npz(
    output_path: str | Path,
    steps: list[int],
    rewards: list[float],
    distances: list[float],
    terminated: list[bool],
    truncated: list[bool],
    end_effector_pos: list[np.ndarray],
    target_pos: list[np.ndarray],
    joint_torques: list[np.ndarray],
    joint_power: list[np.ndarray],
    applied_action: list[np.ndarray],
    seed: int,
    policy: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        step=np.asarray(steps, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float64),
        distance_to_target=np.asarray(distances, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        end_effector_pos=np.asarray(end_effector_pos, dtype=np.float64),
        target_pos=np.asarray(target_pos, dtype=np.float64),
        joint_torques=np.asarray(joint_torques, dtype=np.float64),
        joint_power=np.asarray(joint_power, dtype=np.float64),
        applied_action=np.asarray(applied_action, dtype=np.float64),
        seed=np.asarray(seed, dtype=np.int64),
        policy=np.asarray(policy),
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a smoke rollout for the bridge robot environment.")
    parser.add_argument("--config", default=None, help="Path to the environment config file.")
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Override config.io.output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EnvConfig.load(args.config)
    if args.output_dir is not None:
        config = EnvConfig(
            sim=config.sim,
            robot=config.robot,
            task=config.task,
            reward=config.reward,
            render=config.render,
            io=IOConfig(output_dir=args.output_dir),
        )

    output_dir = Path(config.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = BridgeRobotEnv(config=config)
    observation = env.reset(seed=args.seed)

    max_steps = args.steps if args.steps is not None else config.sim.max_steps
    rollout_steps: list[int] = []
    rewards: list[float] = []
    distances: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    end_effector_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    joint_torque_history: list[np.ndarray] = []
    joint_power_history: list[np.ndarray] = []
    applied_action_history: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)

    print(f"seed={args.seed} policy={args.policy} target={observation['target_pos']}")
    for _ in range(max_steps):
        if args.policy == "zero":
            action = np.zeros(4, dtype=float)
        else:
            limits = np.asarray(config.robot.torque_limits, dtype=float)
            action = rng.uniform(-limits, limits)

        step_result = env.step(action)
        obs = step_result.observation
        print(
            "step={step:03d} distance={distance:.3f} reward={reward:.3f} terminated={terminated} truncated={truncated}".format(
                step=env.state.step_count,
                distance=obs["distance_to_target"],
                reward=step_result.reward,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
            )
        )

        rollout_steps.append(int(env.state.step_count))
        rewards.append(float(step_result.reward))
        distances.append(float(obs["distance_to_target"]))
        terminated_flags.append(bool(step_result.terminated))
        truncated_flags.append(bool(step_result.truncated))
        end_effector_positions.append(np.asarray(obs["end_effector_pos"], dtype=float))
        target_positions.append(np.asarray(obs["target_pos"], dtype=float))
        joint_torque_history.append(np.asarray(obs["joint_torques"], dtype=float))
        joint_power_history.append(np.asarray(obs["joint_power"], dtype=float))
        applied_action_history.append(
            np.asarray(step_result.info["applied_action"], dtype=float)
        )

        if step_result.terminated or step_result.truncated:
            break

    npz_path = save_rollout_npz(
        output_path=output_dir / "run_env_rollout.npz",
        steps=rollout_steps,
        rewards=rewards,
        distances=distances,
        terminated=terminated_flags,
        truncated=truncated_flags,
        end_effector_pos=end_effector_positions,
        target_pos=target_positions,
        joint_torques=joint_torque_history,
        joint_power=joint_power_history,
        applied_action=applied_action_history,
        seed=args.seed,
        policy=args.policy,
    )
    print(f"saved rollout npz to {npz_path}")


if __name__ == "__main__":
    main()
