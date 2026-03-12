import numpy as np

from env.dynamics import compute_gravity_torques, step_dynamics


def test_gravity_torques_are_nonzero_at_horizontal_pose():
    torques = compute_gravity_torques(
        joint_angles=[0.0, 0.0, 0.0, 0.0],
        link_lengths=[1.2, 1.0, 0.8, 0.6],
        link_masses=[8.0, 6.0, 4.0, 2.0],
        payload_mass=1.5,
        gravity=9.81,
    )
    assert np.all(np.abs(torques) > 0.0)


def test_upstream_joints_carry_larger_gravity_torque():
    torques = compute_gravity_torques(
        joint_angles=[0.0, 0.0, 0.0, 0.0],
        link_lengths=[1.2, 1.0, 0.8, 0.6],
        link_masses=[8.0, 6.0, 4.0, 2.0],
        payload_mass=1.5,
        gravity=9.81,
    )
    magnitudes = np.abs(torques)
    assert magnitudes[0] >= magnitudes[1] >= magnitudes[2] >= magnitudes[3]


def test_step_clips_actions_to_torque_limits():
    result = step_dynamics(
        joint_angles=[0.2, 0.4, 0.2, 0.0],
        joint_velocities=[0.0, 0.0, 0.0, 0.0],
        action=[100.0, -100.0, 100.0, -100.0],
        dt=0.02,
        gravity=9.81,
        link_lengths=[1.2, 1.0, 0.8, 0.6],
        link_masses=[8.0, 6.0, 4.0, 2.0],
        payload_mass=1.5,
        joint_damping=[1.2, 1.0, 0.8, 0.6],
        torque_limits=[40.0, 30.0, 20.0, 10.0],
        joint_limits=[[-3.1416, 3.1416]] * 4,
    )
    assert result.action_clipped is True
    assert np.allclose(result.applied_action, np.array([40.0, -30.0, 20.0, -10.0]))
