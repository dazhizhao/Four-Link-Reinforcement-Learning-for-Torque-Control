import numpy as np

from env.kinematics import forward_kinematics


def test_forward_kinematics_zero_pose_reaches_total_length():
    result = forward_kinematics([0.0, 0.0, 0.0, 0.0], [1.2, 1.0, 0.8, 0.6])
    assert np.allclose(result.end_effector_pos, np.array([3.6, 0.0]))


def test_forward_kinematics_known_pose_positions():
    result = forward_kinematics(
        [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    assert np.allclose(result.joint_positions, expected)
