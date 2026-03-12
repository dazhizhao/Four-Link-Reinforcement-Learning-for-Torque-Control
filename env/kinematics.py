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


def cumulative_joint_angles(joint_angles: Sequence[float]) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float)
    if q.ndim != 1:
        raise ValueError("joint_angles must be a 1D sequence")
    return np.cumsum(q)


def forward_kinematics(
    joint_angles: Sequence[float], link_lengths: Sequence[float]
) -> KinematicsResult:
    q = np.asarray(joint_angles, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)

    if q.shape != lengths.shape:
        raise ValueError("joint_angles and link_lengths must have the same shape")

    world_angles = cumulative_joint_angles(q)
    joint_positions = np.zeros((len(lengths) + 1, 2), dtype=float)
    for idx, (theta, length) in enumerate(zip(world_angles, lengths)):
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


def is_pose_above_ground(
    joint_positions: Sequence[Sequence[float]] | np.ndarray,
    ground_y: float = 0.0,
) -> bool:
    positions = np.asarray(joint_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("joint_positions must have shape (N, 2)")
    return bool(np.all(positions[:, 1] >= float(ground_y)))
