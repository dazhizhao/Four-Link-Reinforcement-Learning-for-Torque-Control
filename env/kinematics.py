from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np


@dataclass(frozen=True)
class KinematicsResult:
    joint_positions: np.ndarray
    link_centers: np.ndarray
    end_effector_pos: np.ndarray


def forward_kinematics(
    joint_angles: Sequence[float], link_lengths: Sequence[float]
) -> KinematicsResult:
    q = np.asarray(joint_angles, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)

    if q.shape != lengths.shape:
        raise ValueError("joint_angles and link_lengths must have the same shape")

    joint_positions = np.zeros((len(lengths) + 1, 2), dtype=float)
    for idx, (theta, length) in enumerate(zip(q, lengths)):
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        joint_positions[idx + 1] = joint_positions[idx] + length * direction

    link_centers = 0.5 * (joint_positions[:-1] + joint_positions[1:])
    return KinematicsResult(
        joint_positions=joint_positions,
        link_centers=link_centers,
        end_effector_pos=joint_positions[-1].copy(),
    )


def total_reach(link_lengths: Sequence[float]) -> float:
    return float(np.sum(np.asarray(link_lengths, dtype=float)))
