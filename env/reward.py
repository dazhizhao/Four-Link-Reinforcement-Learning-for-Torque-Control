from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    distance: float
    torque_penalty: float
    power_penalty: float
    smoothness_penalty: float
    success_bonus: float

    @property
    def total(self) -> float:
        return (
            self.distance
            + self.torque_penalty
            + self.power_penalty
            + self.smoothness_penalty
            + self.success_bonus
        )

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total"] = self.total
        return payload


def compute_reward(
    distance_to_target: float,
    applied_action: Sequence[float],
    joint_power: Sequence[float],
    previous_action: Sequence[float],
    success: bool,
    distance_weight: float,
    torque_weight: float,
    power_weight: float,
    smoothness_weight: float,
    success_bonus: float,
) -> RewardBreakdown:
    action = np.asarray(applied_action, dtype=float)
    prev_action = np.asarray(previous_action, dtype=float)
    power = np.asarray(joint_power, dtype=float)

    return RewardBreakdown(
        distance=-distance_weight * float(distance_to_target),
        torque_penalty=-torque_weight * float(np.sum(action**2)),
        power_penalty=-power_weight * float(np.sum(np.abs(power))),
        smoothness_penalty=-smoothness_weight * float(np.sum((action - prev_action) ** 2)),
        success_bonus=success_bonus if success else 0.0,
    )
