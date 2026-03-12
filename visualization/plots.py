from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def save_joint_torque_subplots(
    history: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    if not history:
        raise ValueError("history must contain at least one entry")

    steps = np.array([item["step"] for item in history], dtype=int)
    torques = np.array([item["joint_torques"] for item in history], dtype=float)
    if torques.ndim != 2 or torques.shape[1] != 4:
        raise ValueError("joint_torques must have shape (N, 4)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120, sharex=True)
    colors = ["#005f73", "#0a9396", "#ca6702", "#ae2012"]

    for idx, ax in enumerate(axes.flatten()):
        ax.plot(steps, torques[:, idx], color=colors[idx], linewidth=2)
        ax.set_title(f"Joint {idx + 1} torque")
        ax.set_xlabel("step")
        ax.set_ylabel("torque (Nm)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Best-policy joint torques", fontsize=14)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_training_curves(
    output_path: str | Path,
    monitor_csv_path: str | Path,
    progress_csv_path: str | Path | None = None,
) -> Path:
    monitor_path = Path(monitor_csv_path)
    if not monitor_path.exists():
        raise FileNotFoundError(f"monitor log not found: {monitor_path}")

    monitor_df = pd.read_csv(monitor_path, comment="#")
    progress_df = None
    if progress_csv_path is not None:
        progress_path = Path(progress_csv_path)
        if progress_path.exists() and progress_path.stat().st_size > 0:
            progress_df = pd.read_csv(progress_path)
            if progress_df.empty:
                progress_df = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    ax_reward, ax_length, ax_loss, ax_aux = axes.flatten()

    if not monitor_df.empty and {"r", "l"}.issubset(monitor_df.columns):
        episode_index = np.arange(1, len(monitor_df) + 1)
        reward_values = monitor_df["r"].to_numpy(dtype=float)
        length_values = monitor_df["l"].to_numpy(dtype=float)

        ax_reward.plot(episode_index, reward_values, color="#0a9396", linewidth=2)
        ax_reward.set_title("Episode reward")
        ax_reward.set_xlabel("episode")
        ax_reward.set_ylabel("reward")
        ax_reward.grid(True, alpha=0.3)

        ax_length.plot(episode_index, length_values, color="#bb3e03", linewidth=2)
        ax_length.set_title("Episode length")
        ax_length.set_xlabel("episode")
        ax_length.set_ylabel("steps")
        ax_length.grid(True, alpha=0.3)
    else:
        ax_reward.text(0.5, 0.5, "episode logs unavailable", ha="center", va="center", transform=ax_reward.transAxes)
        ax_reward.set_title("Episode reward")
        ax_reward.set_axis_off()

        ax_length.text(0.5, 0.5, "episode logs unavailable", ha="center", va="center", transform=ax_length.transAxes)
        ax_length.set_title("Episode length")
        ax_length.set_axis_off()

    plotted_loss = False
    plotted_aux = False
    if progress_df is not None and "time/total_timesteps" in progress_df.columns:
        x_values = progress_df["time/total_timesteps"].to_numpy(dtype=float)

        for key, label, color in [
            ("train/actor_loss", "actor loss", "#005f73"),
            ("train/critic_loss", "critic loss", "#9b2226"),
            ("train/ent_coef_loss", "entropy coef loss", "#ca6702"),
        ]:
            if key in progress_df.columns:
                ax_loss.plot(x_values, progress_df[key].to_numpy(dtype=float), linewidth=1.8, label=label, color=color)
                plotted_loss = True

        for key, label, color in [
            ("rollout/ep_rew_mean", "mean reward", "#0a9396"),
            ("rollout/ep_len_mean", "mean length", "#ae2012"),
            ("train/ent_coef", "entropy coef", "#6c757d"),
        ]:
            if key in progress_df.columns:
                ax_aux.plot(x_values, progress_df[key].to_numpy(dtype=float), linewidth=1.8, label=label, color=color)
                plotted_aux = True

    if plotted_loss:
        ax_loss.set_title("Training losses")
        ax_loss.set_xlabel("timesteps")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="best")
    else:
        ax_loss.text(0.5, 0.5, "loss logs unavailable", ha="center", va="center", transform=ax_loss.transAxes)
        ax_loss.set_title("Training losses")
        ax_loss.set_axis_off()

    if plotted_aux:
        ax_aux.set_title("Training diagnostics")
        ax_aux.set_xlabel("timesteps")
        ax_aux.grid(True, alpha=0.3)
        ax_aux.legend(loc="best")
    else:
        ax_aux.text(0.5, 0.5, "progress logs unavailable", ha="center", va="center", transform=ax_aux.transAxes)
        ax_aux.set_title("Training diagnostics")
        ax_aux.set_axis_off()

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
