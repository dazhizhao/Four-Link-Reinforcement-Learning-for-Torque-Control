from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from env.link_allocation_env import LinkAllocationEnv, project_bounded_simplex


def test_reset_returns_expected_observation_shape_and_dtype():
    env = LinkAllocationEnv()
    observation, info = env.reset(seed=7)

    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert np.allclose(info["allocated_lengths"], np.array([1.2, 1.0, 0.8, 0.6], dtype=np.float32))


def test_step_terminates_immediately_without_truncation():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, _, terminated, truncated, _ = env.step([0.9, 0.9, 0.9, 0.9])

    assert terminated is True
    assert truncated is False


def test_projected_lengths_respect_sum_and_bounds():
    projected = project_bounded_simplex(
        values=[1.4, 1.4, 1.4, 1.4],
        target_sum=3.6,
        lower=[0.4, 0.4, 0.4, 0.4],
        upper=[1.4, 1.4, 1.4, 1.4],
    )

    assert np.isclose(np.sum(projected), 3.6)
    assert np.all(projected >= 0.4 - 1e-8)
    assert np.all(projected <= 1.4 + 1e-8)


def test_balanced_lengths_outperform_unbalanced_lengths():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, balanced_reward, _, _, balanced_info = env.step([0.9, 0.9, 0.9, 0.9])

    env.reset(seed=7)
    _, unbalanced_reward, _, _, unbalanced_info = env.step([1.4, 1.4, 0.4, 0.4])

    assert balanced_reward > unbalanced_reward
    assert balanced_info["inner_radius"] == 0.0
    assert unbalanced_info["inner_radius"] > 0.0


def test_inner_radius_is_zero_when_no_link_dominates():
    env = LinkAllocationEnv()
    env.reset(seed=7)
    _, _, _, _, info = env.step([1.2, 0.8, 0.8, 0.8])

    assert info["inner_radius"] == 0.0
