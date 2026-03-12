from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None


@dataclass(frozen=True)
class LinkAllocationConfig:
    total_length: float
    default_link_lengths: list[float]
    min_link_lengths: list[float]
    max_link_lengths: list[float]

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "LinkAllocationConfig":
        path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parents[1] / "configs" / "link_allocation_env.yaml"
        )
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return cls(**raw)

    def validate(self) -> None:
        default = np.asarray(self.default_link_lengths, dtype=float)
        lower = np.asarray(self.min_link_lengths, dtype=float)
        upper = np.asarray(self.max_link_lengths, dtype=float)

        if default.shape != (4,) or lower.shape != (4,) or upper.shape != (4,):
            raise ValueError("Link allocation config expects exactly four link lengths.")
        if not np.all(lower <= upper):
            raise ValueError("min_link_lengths must be less than or equal to max_link_lengths.")
        if self.total_length <= 0.0:
            raise ValueError("total_length must be positive.")
        if float(np.sum(lower)) > self.total_length or float(np.sum(upper)) < self.total_length:
            raise ValueError("total_length must lie within the feasible bounded-simplex range.")
        if np.any(default < lower) or np.any(default > upper):
            raise ValueError("default_link_lengths must satisfy per-link bounds.")
        if not np.isclose(float(np.sum(default)), self.total_length):
            raise ValueError("default_link_lengths must sum to total_length.")


@dataclass(frozen=True)
class WorkspaceMetrics:
    inner_radius: float
    outer_radius: float
    workspace_area: float
    normalized_reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class LinkAllocationEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: LinkAllocationConfig | None = None, config_path: str | Path | None = None) -> None:
        self.config = config if config is not None else LinkAllocationConfig.load(config_path)
        self.config.validate()

        self._default = np.asarray(self.config.default_link_lengths, dtype=np.float32)
        self._lower = np.asarray(self.config.min_link_lengths, dtype=np.float64)
        self._upper = np.asarray(self.config.max_link_lengths, dtype=np.float64)
        self._total_length = float(self.config.total_length)
        self._observation = self._build_observation()
        self.action_space = spaces.Box(
            low=np.asarray(self.config.min_link_lengths, dtype=np.float32),
            high=np.asarray(self.config.max_link_lengths, dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )
        self.last_allocated_lengths = self._default.copy()
        self.last_metrics = self._compute_workspace_metrics(self.last_allocated_lengths)
        self._episode_active = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.last_allocated_lengths = self._default.copy()
        self.last_metrics = self._compute_workspace_metrics(self.last_allocated_lengths)
        self._episode_active = True
        info = {
            "allocated_lengths": self.last_allocated_lengths.copy(),
            "workspace_area": self.last_metrics.workspace_area,
            "inner_radius": self.last_metrics.inner_radius,
            "outer_radius": self.last_metrics.outer_radius,
            "projection_applied": False,
        }
        return self._observation.copy(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._episode_active:
            raise RuntimeError("Call reset() before step().")

        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (4,):
            raise ValueError("LinkAllocationEnv expects a 4D action.")

        allocated = project_bounded_simplex(
            raw_action,
            target_sum=self._total_length,
            lower=self._lower,
            upper=self._upper,
        ).astype(np.float32)
        metrics = self._compute_workspace_metrics(allocated)
        projection_applied = not np.allclose(allocated, raw_action, atol=1e-6, rtol=1e-6)

        self.last_allocated_lengths = allocated
        self.last_metrics = metrics
        self._episode_active = False

        info = {
            "allocated_lengths": allocated.copy(),
            "raw_action": raw_action.astype(np.float32),
            "workspace_area": metrics.workspace_area,
            "inner_radius": metrics.inner_radius,
            "outer_radius": metrics.outer_radius,
            "projection_applied": projection_applied,
        }
        return self._observation.copy(), float(metrics.normalized_reward), True, False, info

    def _build_observation(self) -> np.ndarray:
        total = np.float32(self._total_length)
        return np.concatenate(
            [
                self._default / total,
                self._lower.astype(np.float32) / total,
                self._upper.astype(np.float32) / total,
                np.array([total], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _compute_workspace_metrics(self, lengths: Sequence[float]) -> WorkspaceMetrics:
        lengths_arr = np.asarray(lengths, dtype=np.float64)
        outer_radius = float(np.sum(lengths_arr))
        inner_radius = float(max(0.0, 2.0 * float(np.max(lengths_arr)) - outer_radius))
        workspace_area = float(np.pi * (outer_radius**2 - inner_radius**2))
        normalized_reward = float(workspace_area / (np.pi * (self._total_length**2)))
        return WorkspaceMetrics(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            workspace_area=workspace_area,
            normalized_reward=normalized_reward,
        )


def project_bounded_simplex(
    values: Sequence[float],
    target_sum: float,
    lower: Sequence[float],
    upper: Sequence[float],
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    if raw.shape != lower_arr.shape or raw.shape != upper_arr.shape:
        raise ValueError("values, lower, and upper must share the same shape.")
    if float(np.sum(lower_arr)) > target_sum or float(np.sum(upper_arr)) < target_sum:
        raise ValueError("target_sum is infeasible for the provided bounds.")

    if np.all(raw >= lower_arr) and np.all(raw <= upper_arr) and np.isclose(np.sum(raw), target_sum):
        return raw.copy()

    left = float(np.min(raw - upper_arr))
    right = float(np.max(raw - lower_arr))
    for _ in range(max_iterations):
        midpoint = 0.5 * (left + right)
        projected = np.clip(raw - midpoint, lower_arr, upper_arr)
        error = float(np.sum(projected) - target_sum)
        if abs(error) <= tolerance:
            return projected
        if error > 0.0:
            left = midpoint
        else:
            right = midpoint

    projected = np.clip(raw - 0.5 * (left + right), lower_arr, upper_arr)
    residual = target_sum - float(np.sum(projected))
    if abs(residual) > 1e-6:
        free_mask = (projected > lower_arr + 1e-8) & (projected < upper_arr - 1e-8)
        if np.any(free_mask):
            projected = projected.copy()
            projected[free_mask] += residual / float(np.sum(free_mask))
            projected = np.clip(projected, lower_arr, upper_arr)
    return projected
