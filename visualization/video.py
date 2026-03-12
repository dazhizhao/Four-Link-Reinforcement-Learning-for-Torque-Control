from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from env.kinematics import forward_kinematics


def export_rollout_video(
    history: list[dict[str, Any]],
    link_lengths: Sequence[float],
    target_pos: Sequence[float],
    output_path: str | Path,
    fps: int,
) -> Path:
    if not history:
        raise ValueError("history must contain at least one entry")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target_pos, dtype=float)

    trajectory = np.array([frame["end_effector_pos"] for frame in history], dtype=float)
    max_frame_radius = 0.0
    for frame in history:
        kin = forward_kinematics(frame["q"], link_lengths)
        max_frame_radius = max(max_frame_radius, float(np.max(np.abs(kin.joint_positions))))
    max_radius = max(max_frame_radius, float(np.linalg.norm(target))) + 0.5

    with imageio.get_writer(output, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p") as writer:
        for idx, frame in enumerate(history):
            kin = forward_kinematics(frame["q"], link_lengths)
            fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

            ax.plot(
                kin.joint_positions[:, 0],
                kin.joint_positions[:, 1],
                "-o",
                color="#005f73",
                linewidth=3,
                markersize=7,
            )
            ax.scatter(0.0, 0.0, color="#9b2226", s=120, label="base", zorder=5)
            ax.scatter(
                kin.end_effector_pos[0],
                kin.end_effector_pos[1],
                color="#ee9b00",
                s=180,
                label="end effector",
                zorder=6,
            )
            ax.scatter(
                target[0],
                target[1],
                color="#ae2012",
                s=120,
                marker="x",
                linewidths=3,
                label="target",
                zorder=6,
            )

            partial_traj = trajectory[: idx + 1]
            ax.plot(
                partial_traj[:, 0],
                partial_traj[:, 1],
                color="#0a9396",
                linewidth=2,
                alpha=0.85,
                label="trajectory",
            )

            ax.set_xlim(-max_radius, max_radius)
            ax.set_ylim(-max_radius, max_radius)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Bridge Robot Rollout")
            ax.legend(loc="upper left")

            torque_text = "\n".join(
                f"M{i + 1}: {value: .2f} Nm"
                for i, value in enumerate(np.asarray(frame["joint_torques"], dtype=float))
            )
            detail_text = "\n".join(
                [
                    f"step: {frame['step']}",
                    f"distance: {float(frame['distance_to_target']):.3f} m",
                    torque_text,
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

            fig.tight_layout()
            fig.canvas.draw()
            frame_image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame_image)
            plt.close(fig)

    return output
