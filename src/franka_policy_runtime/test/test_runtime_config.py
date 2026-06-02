import numpy as np
import pytest

from franka_policy_runtime.runtime_config import FR3_JOINT_NAMES
from franka_policy_runtime.reference import (
    _quat_xyzw_from_axis_angle,
    apply_tcp_delta,
    gripper_width_from_binary_action,
    split_policy_action,
    step_toward_pose,
)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    quat = np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=float)
    return quat / np.linalg.norm(quat)


def _same_orientation(actual: np.ndarray, expected: np.ndarray) -> bool:
    return abs(float(np.dot(actual, expected))) == pytest.approx(1.0)


def test_fr3_joint_names_has_7_joints():
    assert len(FR3_JOINT_NAMES) == 7
    assert all(name.startswith("fr3_joint") for name in FR3_JOINT_NAMES)
    assert FR3_JOINT_NAMES == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]


def test_split_policy_action_separates_tcp_delta_and_gripper():
    action = np.array([0.01, 0.02, -0.03, 0.1, -0.2, 0.3, 0.04], dtype=float)

    tcp_delta, gripper_delta = split_policy_action(action)

    assert tcp_delta.tolist() == [0.01, 0.02, -0.03, 0.1, -0.2, 0.3]
    assert gripper_delta == 0.04


def test_apply_tcp_delta_matches_isaaclab_scale_and_axis_angle_semantics():
    position = np.zeros(3, dtype=float)
    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    action = np.array([0.2, -0.2, 0.04, 0.0, 0.0, np.pi, 1.0], dtype=float)

    target_position, target_quat = apply_tcp_delta(
        position,
        quat_xyzw,
        action,
        action_scale=0.5,
    )

    assert target_position.tolist() == pytest.approx([0.1, -0.1, 0.02])
    expected_delta = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi * 0.5], dtype=float))
    assert _same_orientation(target_quat, expected_delta)


def test_apply_tcp_delta_composes_axis_angle_delta_in_command_frame():
    position = np.zeros(3, dtype=float)
    current_quat = _quat_xyzw_from_axis_angle(np.array([0.3, -0.4, 0.2], dtype=float))
    axis_angle_delta = np.array([0.2, 0.1, -0.3], dtype=float)
    action = np.array([0.0, 0.0, 0.0, *axis_angle_delta, 0.0], dtype=float)

    _, target_quat = apply_tcp_delta(
        position,
        current_quat,
        action,
        action_scale=1.0,
    )

    delta_quat = _quat_xyzw_from_axis_angle(axis_angle_delta)
    expected_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    assert _same_orientation(target_quat, expected_quat)


def test_gripper_width_from_binary_action_matches_isaaclab_sign_semantics():
    assert gripper_width_from_binary_action(-0.1, min_width=0.0, max_width=0.08) == 0.0
    assert gripper_width_from_binary_action(0.0, min_width=0.0, max_width=0.08) == 0.08
    assert gripper_width_from_binary_action(0.2, min_width=0.0, max_width=0.08) == 0.08


def test_step_toward_pose_passes_through_small_step():
    """When the target is within limits, step_toward_pose returns it unchanged."""
    position = np.array([0.1, -0.2, 0.3], dtype=float)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    target_position = np.array([0.11, -0.21, 0.31], dtype=float)
    target_quat = _quat_xyzw_from_axis_angle(np.array([0.1, 0.0, 0.0], dtype=float))

    stepped_pos, stepped_quat = step_toward_pose(
        position, quat, target_position, target_quat,
        max_translation_step=0.1, max_rotation_step=0.5,
    )

    assert stepped_pos == pytest.approx(target_position)
    assert _same_orientation(stepped_quat, target_quat)


def test_step_toward_pose_clamps_translation_norm():
    position = np.array([0.0, 0.0, 0.0], dtype=float)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    target_position = np.array([0.03, 0.04, 0.0], dtype=float)  # norm = 0.05
    target_quat = quat.copy()

    stepped_pos, stepped_quat = step_toward_pose(
        position, quat, target_position, target_quat,
        max_translation_step=0.01, max_rotation_step=1.0,
    )

    delta = stepped_pos - position
    assert float(np.linalg.norm(delta)) == pytest.approx(0.01)
    # direction preserved
    assert delta[0] / delta[1] == pytest.approx(0.03 / 0.04)


def test_step_toward_pose_clamps_rotation_angle():
    position = np.zeros(3, dtype=float)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    big_axis_angle = np.array([0.0, 0.0, 0.5], dtype=float)  # 0.5 rad
    target_quat = _quat_xyzw_from_axis_angle(big_axis_angle)

    _, stepped_quat = step_toward_pose(
        position, quat, position, target_quat,
        max_translation_step=1.0, max_rotation_step=0.1,
    )

    # The stepped quat should represent a rotation of exactly 0.1 rad
    angle = 2.0 * np.arctan2(float(np.linalg.norm(stepped_quat[:3])), float(abs(stepped_quat[3])))
    assert angle == pytest.approx(0.1, abs=1e-6)
