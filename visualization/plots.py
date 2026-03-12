from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_rollout_history(
    history: list[dict[str, Any]],
    save_path: str | Path | None = None,
    show: bool = False,
):
    if not history:
        raise ValueError("history must contain at least one entry")

    steps = np.array([item["step"] for item in history], dtype=int)
    distances = np.array([item["distance_to_target"] for item in history], dtype=float)
    torques = np.array([item["joint_torques"] for item in history], dtype=float)
    power = np.array([item["joint_power"] for item in history], dtype=float)
    positions = np.array([item["end_effector_pos"] for item in history], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    ax_distance, ax_torque, ax_power, ax_position = axes.flatten()

    ax_distance.plot(steps, distances, color="#0a9396", linewidth=2)
    ax_distance.set_title("Distance to target")
    ax_distance.set_xlabel("step")
    ax_distance.set_ylabel("distance (m)")
    ax_distance.grid(True, alpha=0.3)

    for idx in range(torques.shape[1]):
        ax_torque.plot(steps, torques[:, idx], linewidth=1.8, label=f"M{idx + 1}")
    ax_torque.set_title("Joint torques")
    ax_torque.set_xlabel("step")
    ax_torque.set_ylabel("torque (Nm)")
    ax_torque.grid(True, alpha=0.3)
    ax_torque.legend(loc="best")

    for idx in range(power.shape[1]):
        ax_power.plot(steps, power[:, idx], linewidth=1.8, label=f"P{idx + 1}")
    ax_power.set_title("Joint power")
    ax_power.set_xlabel("step")
    ax_power.set_ylabel("power (W)")
    ax_power.grid(True, alpha=0.3)
    ax_power.legend(loc="best")

    ax_position.plot(steps, positions[:, 0], color="#bb3e03", linewidth=2, label="x")
    ax_position.plot(steps, positions[:, 1], color="#005f73", linewidth=2, label="y")
    ax_position.set_title("End effector position")
    ax_position.set_xlabel("step")
    ax_position.set_ylabel("position (m)")
    ax_position.grid(True, alpha=0.3)
    ax_position.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig
