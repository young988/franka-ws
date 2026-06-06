import math

import numpy as np
import pytest

from franka_policy_runtime.utils.pose_math import compose_pose_xyzw


def test_compose_pose_rotates_child_offset_into_parent_frame():
    parent_position = np.array([1.0, 2.0, 3.0])
    half = math.sqrt(0.5)
    parent_quat = np.array([0.0, 0.0, half, half])
    child_position = np.array([0.1, 0.0, 0.0])
    child_quat = np.array([0.0, 0.0, 0.0, 1.0])

    position, quat = compose_pose_xyzw(
        parent_position,
        parent_quat,
        child_position,
        child_quat,
    )

    np.testing.assert_allclose(position, [1.0, 2.1, 3.0], atol=1.0e-9)
    np.testing.assert_allclose(quat, parent_quat, atol=1.0e-9)


def test_compose_pose_combines_orientation_quaternions():
    half = math.sqrt(0.5)
    _, quat = compose_pose_xyzw(
        np.zeros(3),
        np.array([half, 0.0, 0.0, half]),
        np.zeros(3),
        np.array([0.0, half, 0.0, half]),
    )

    assert np.linalg.norm(quat) == pytest.approx(1.0)
    np.testing.assert_allclose(quat, [0.5, 0.5, 0.5, 0.5], atol=1.0e-9)
