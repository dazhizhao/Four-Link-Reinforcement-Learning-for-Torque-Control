from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def render_environment_state(
    state: Any,
    history: list[dict[str, Any]],
    config: Any,
    ground_y: float = 0.0,
    save_path: str | Path | None = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=tuple(config.figsize), dpi=config.dpi)

    joints = state.joint_positions
    ax.plot(joints[:, 0], joints[:, 1], "-o", color="#005f73", linewidth=3, markersize=7)
    ax.scatter(joints[0, 0], joints[0, 1], color="#9b2226", s=120, label="base", zorder=5)
    ax.scatter(
        state.end_effector_pos[0],
        state.end_effector_pos[1],
        color="#ee9b00",
        s=180,
        label="end effector",
        zorder=6,
    )
    ax.scatter(
        state.target_pos[0],
        state.target_pos[1],
        color="#ae2012",
        s=120,
        marker="x",
        linewidths=3,
        label="target",
        zorder=6,
    )

    if history:
        trajectory = np.array([item["end_effector_pos"] for item in history], dtype=float)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#0a9396",
            linewidth=2,
            alpha=config.history_alpha,
            label="trajectory",
        )

    ax.axhline(ground_y, color="#bb3e03", linestyle="--", linewidth=2, alpha=0.9, label="ground")

    y_values = [joints[:, 1], np.asarray([state.target_pos[1]], dtype=float)]
    if history:
        y_values.append(trajectory[:, 1])
    y_min = min(float(np.min(values)) for values in y_values)
    y_max = max(float(np.max(values)) for values in y_values)
    max_radius = max(
        float(np.max(np.abs(joints))) + 0.5,
        float(np.linalg.norm(state.target_pos)) + 0.5,
    )
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(min(ground_y - 0.25, y_min - 0.25), max(y_max + 0.5, ground_y + 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Bridge Robot Environment")
    ax.legend(loc="upper left")

    torque_text = "\n".join(
        f"M{i + 1}: {value: .2f} Nm" for i, value in enumerate(np.asarray(state.joint_torques, dtype=float))
    )
    power_text = "\n".join(
        f"P{i + 1}: {value: .2f} W" for i, value in enumerate(np.asarray(state.joint_power, dtype=float))
    )
    detail_text = "\n".join(
        [
            f"step: {state.step_count}",
            f"distance: {state.distance_to_target:.3f} m",
            f"hold progress: {float(state.hold_progress):.2f}",
            torque_text,
            power_text,
        ]
    )
    ax.text(
        1.02,
        0.98,
        detail_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#94d2bd"},
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig
