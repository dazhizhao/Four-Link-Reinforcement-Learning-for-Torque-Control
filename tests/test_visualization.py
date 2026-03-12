from pathlib import Path

from env.bridge_robot_env import BridgeRobotEnv
from visualization.video import export_rollout_video


def test_export_rollout_video_creates_mp4(tmp_path: Path):
    env = BridgeRobotEnv()
    env.reset(seed=7)
    for _ in range(3):
        env.step([0.0, 0.0, 0.0, 0.0])

    output_path = tmp_path / "rollout.mp4"
    export_rollout_video(
        history=env.history,
        link_lengths=env.config.robot.link_lengths,
        target_pos=env.state.target_pos,
        output_path=output_path,
        fps=5,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
