from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kinematics import KinematicsResult, forward_kinematics


@dataclass(frozen=True)
class DynamicsResult:
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    joint_positions: np.ndarray
    link_centers: np.ndarray
    end_effector_pos: np.ndarray
    end_effector_vel: np.ndarray
    end_effector_acc: np.ndarray
    joint_torques: np.ndarray
    joint_power: np.ndarray
    applied_action: np.ndarray
    gravity_torques: np.ndarray
    equivalent_inertia: np.ndarray
    action_clipped: bool
    joint_limit_clipped: bool


def compute_gravity_torques(
    joint_angles: Sequence[float],
    link_lengths: Sequence[float],
    link_masses: Sequence[float],
    payload_mass: float,
    gravity: float,
) -> np.ndarray:
    lengths = np.asarray(link_lengths, dtype=float)
    masses = np.asarray(link_masses, dtype=float)
    kin = forward_kinematics(joint_angles, lengths)
    torques = np.zeros_like(lengths, dtype=float)

    payload_pos = kin.end_effector_pos
    for joint_idx in range(len(lengths)):
        pivot_x = kin.joint_positions[joint_idx, 0]
        downstream_centers = kin.link_centers[joint_idx:, 0]
        downstream_masses = masses[joint_idx:]
        torques[joint_idx] = -gravity * np.sum(
            downstream_masses * (downstream_centers - pivot_x)
        )
        torques[joint_idx] += -gravity * payload_mass * (payload_pos[0] - pivot_x)
    return torques


def compute_equivalent_inertia(
    joint_angles: Sequence[float],
    link_lengths: Sequence[float],
    link_masses: Sequence[float],
    payload_mass: float,
) -> np.ndarray:
    lengths = np.asarray(link_lengths, dtype=float)
    masses = np.asarray(link_masses, dtype=float)
    kin = forward_kinematics(joint_angles, lengths)
    inertia = np.zeros_like(lengths, dtype=float)

    for joint_idx in range(len(lengths)):
        pivot = kin.joint_positions[joint_idx]
        downstream_centers = kin.link_centers[joint_idx:]
        downstream_masses = masses[joint_idx:]
        center_r2 = np.sum((downstream_centers - pivot) ** 2, axis=1)
        payload_r2 = float(np.sum((kin.end_effector_pos - pivot) ** 2))
        inertia[joint_idx] = np.sum(downstream_masses * center_r2) + payload_mass * payload_r2

    return np.maximum(inertia, 0.1)


def step_dynamics(
    joint_angles: Sequence[float],
    joint_velocities: Sequence[float],
    action: Sequence[float],
    dt: float,
    gravity: float,
    link_lengths: Sequence[float],
    link_masses: Sequence[float],
    payload_mass: float,
    joint_damping: Sequence[float],
    torque_limits: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
    prev_end_effector_vel: Sequence[float] | None = None,
) -> DynamicsResult:
    q = np.asarray(joint_angles, dtype=float)
    qd = np.asarray(joint_velocities, dtype=float)
    tau_cmd = np.asarray(action, dtype=float)
    damping = np.asarray(joint_damping, dtype=float)
    torque_limits_arr = np.asarray(torque_limits, dtype=float)
    joint_limits_arr = np.asarray(joint_limits, dtype=float)

    tau_applied = np.clip(tau_cmd, -torque_limits_arr, torque_limits_arr)
    action_clipped = not np.allclose(tau_applied, tau_cmd)

    gravity_torques = compute_gravity_torques(
        joint_angles=q,
        link_lengths=link_lengths,
        link_masses=link_masses,
        payload_mass=payload_mass,
        gravity=gravity,
    )
    equivalent_inertia = compute_equivalent_inertia(
        joint_angles=q,
        link_lengths=link_lengths,
        link_masses=link_masses,
        payload_mass=payload_mass,
    )

    damping_load = damping * qd
    qdd = (tau_applied + gravity_torques - damping_load) / equivalent_inertia
    qd_next = qd + dt * qdd
    q_next = q + dt * qd_next

    q_next_clipped = np.clip(q_next, joint_limits_arr[:, 0], joint_limits_arr[:, 1])
    clipped_mask = ~np.isclose(q_next_clipped, q_next)
    joint_limit_clipped = bool(np.any(clipped_mask))
    if joint_limit_clipped:
        qd_next = qd_next.copy()
        qdd = qdd.copy()
        qd_next[clipped_mask] = 0.0
        qdd[clipped_mask] = 0.0
    q_next = q_next_clipped

    kin_current: KinematicsResult = forward_kinematics(q, link_lengths)
    kin_next: KinematicsResult = forward_kinematics(q_next, link_lengths)
    ee_vel = (kin_next.end_effector_pos - kin_current.end_effector_pos) / dt
    previous_velocity = (
        np.zeros(2, dtype=float)
        if prev_end_effector_vel is None
        else np.asarray(prev_end_effector_vel, dtype=float)
    )
    ee_acc = (ee_vel - previous_velocity) / dt

    gravity_torques_next = compute_gravity_torques(
        joint_angles=q_next,
        link_lengths=link_lengths,
        link_masses=link_masses,
        payload_mass=payload_mass,
        gravity=gravity,
    )
    equivalent_inertia_next = compute_equivalent_inertia(
        joint_angles=q_next,
        link_lengths=link_lengths,
        link_masses=link_masses,
        payload_mass=payload_mass,
    )
    joint_power = gravity_torques_next * qd_next

    return DynamicsResult(
        q=q_next,
        qd=qd_next,
        qdd=qdd,
        joint_positions=kin_next.joint_positions,
        link_centers=kin_next.link_centers,
        end_effector_pos=kin_next.end_effector_pos,
        end_effector_vel=ee_vel,
        end_effector_acc=ee_acc,
        joint_torques=gravity_torques_next,
        joint_power=joint_power,
        applied_action=tau_applied,
        gravity_torques=gravity_torques_next,
        equivalent_inertia=equivalent_inertia_next,
        action_clipped=action_clipped,
        joint_limit_clipped=joint_limit_clipped,
    )
