from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bridge_robot_env import BridgeRobotEnv, EnvConfig


class TorqueControlEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, config_path: str | Path | None = None) -> None:
        self.base_env = BridgeRobotEnv(config=config, config_path=config_path)
        self.config = self.base_env.config
        self._torque_limits = np.asarray(self.config.robot.torque_limits, dtype=np.float32)
        self._joint_limits = np.asarray(self.config.robot.joint_limits, dtype=np.float32)
        self._joint_mid = np.mean(self._joint_limits, axis=1)
        self._joint_half_range = np.maximum(
            0.5 * (self._joint_limits[:, 1] - self._joint_limits[:, 0]),
            1e-6,
        )
        damping = np.asarray(self.config.robot.joint_damping, dtype=np.float32)
        self._velocity_scale = np.maximum(self._torque_limits / np.maximum(damping, 1e-6), 1.0)
        self._workspace_radius = max(float(self.base_env.workspace_radius), 1e-6)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        observation = self.base_env.reset(seed=seed)
        return self._flatten_observation(observation), {
            "target_pos": np.asarray(observation["target_pos"], dtype=np.float32),
            "distance_to_target": float(observation["distance_to_target"]),
            "hold_progress": float(observation["hold_progress"]),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (4,):
            raise ValueError("TorqueControlEnv expects a 4D normalized action.")

        action_norm = np.clip(action_arr, self.action_space.low, self.action_space.high)
        scaled_action = action_norm * self._torque_limits
        result = self.base_env.step(scaled_action)
        info = dict(result.info)
        info["action_norm"] = action_norm.copy()
        info["distance_to_target"] = float(result.observation["distance_to_target"])
        return (
            self._flatten_observation(result.observation),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            info,
        )

    def close(self) -> None:
        self.base_env.close()

    def _flatten_observation(self, observation: dict[str, Any]) -> np.ndarray:
        q = np.asarray(observation["q"], dtype=np.float32)
        qd = np.asarray(observation["qd"], dtype=np.float32)
        end_effector_pos = np.asarray(observation["end_effector_pos"], dtype=np.float32)
        target_pos = np.asarray(observation["target_pos"], dtype=np.float32)
        distance = float(observation["distance_to_target"])
        previous_action = np.asarray(observation["applied_action_norm"], dtype=np.float32)
        hold_progress = float(observation["hold_progress"])

        q_normalized = (q - self._joint_mid) / self._joint_half_range
        qd_normalized = np.tanh(qd / self._velocity_scale)
        ee_pos_normalized = end_effector_pos / self._workspace_radius
        target_delta_normalized = (target_pos - end_effector_pos) / self._workspace_radius
        distance_normalized = np.array([distance / self._workspace_radius], dtype=np.float32)
        hold_progress_arr = np.array([hold_progress], dtype=np.float32)

        return np.concatenate(
            [
                q_normalized.astype(np.float32),
                qd_normalized.astype(np.float32),
                ee_pos_normalized.astype(np.float32),
                target_delta_normalized.astype(np.float32),
                distance_normalized,
                previous_action.astype(np.float32),
                hold_progress_arr,
            ]
        ).astype(np.float32)
