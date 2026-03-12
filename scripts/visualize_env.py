from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.bridge_robot_env import BridgeRobotEnv, EnvConfig, IOConfig
from visualization.plots import plot_rollout_history
from visualization.video import export_rollout_video

CONFIG_PATH = None
OUTPUT_DIR_OVERRIDE = None
ROLLOUT_SEED = 7
ROLLOUT_STEPS = 80
ROLLOUT_POLICY = "random"
VIDEO_FPS = 15
VIDEO_NAME = "rollout.mp4"


def main() -> None:
    config = EnvConfig.load(CONFIG_PATH)
    if OUTPUT_DIR_OVERRIDE is not None:
        config = EnvConfig(
            sim=config.sim,
            robot=config.robot,
            task=config.task,
            reward=config.reward,
            render=config.render,
            io=IOConfig(output_dir=OUTPUT_DIR_OVERRIDE),
        )

    output_dir = Path(config.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = BridgeRobotEnv(config=config)
    env.reset(seed=ROLLOUT_SEED)
    rng = np.random.default_rng(ROLLOUT_SEED)
    limits = np.asarray(config.robot.torque_limits, dtype=float)

    for _ in range(ROLLOUT_STEPS):
        action = (
            np.zeros(4, dtype=float)
            if ROLLOUT_POLICY == "zero"
            else rng.uniform(-limits, limits)
        )
        step_result = env.step(action)
        if step_result.terminated or step_result.truncated:
            break

    pose_path = output_dir / "pose.png"
    timeseries_path = output_dir / "timeseries.png"
    video_path = output_dir / VIDEO_NAME
    env.render(save_path=pose_path, show=False)
    plot_rollout_history(env.history, save_path=timeseries_path, show=False)
    export_rollout_video(
        history=env.history,
        link_lengths=config.robot.link_lengths,
        target_pos=env.state.target_pos,
        output_path=video_path,
        fps=VIDEO_FPS,
        ground_y=config.task.ground_y,
    )

    print(f"saved pose figure to {pose_path}")
    print(f"saved timeseries figure to {timeseries_path}")
    print(f"saved rollout video to {video_path}")


if __name__ == "__main__":
    main()
