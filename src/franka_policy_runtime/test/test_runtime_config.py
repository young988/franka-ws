import numpy as np
import pytest

from franka_policy_runtime.runtime_config import RuntimeConfig
from franka_policy_runtime.reference import (
    _quat_xyzw_from_rpy,
    apply_tcp_delta,
    split_policy_action,
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


def test_runtime_config_defaults_to_single_step_mode():
    config = RuntimeConfig()

    assert config.mode == "single_step"
    assert config.observer_type == "vla"
    assert config.instruction_topic == "~/instruction"
    assert config.joint_names == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]
    assert config.actions_per_chunk == 1
    assert config.command_frame == "fr3_link0"
    assert config.tcp_frame == "fr3_hand_tcp"


def test_split_policy_action_separates_tcp_delta_and_gripper():
    action = np.array([0.01, 0.02, -0.03, 0.1, -0.2, 0.3, 0.04], dtype=float)

    tcp_delta, gripper_delta = split_policy_action(action)

    assert tcp_delta.tolist() == [0.01, 0.02, -0.03, 0.1, -0.2, 0.3]
    assert gripper_delta == 0.04


def test_apply_tcp_delta_clips_translation_and_rotation():
    position = np.zeros(3, dtype=float)
    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    action = np.array([10.0, -10.0, 0.2, 1.0, -1.0, 0.1, 0.04], dtype=float)

    target_position, target_quat = apply_tcp_delta(
        position,
        quat_xyzw,
        action,
        max_translation_delta=0.05,
        max_rotation_delta=0.25,
    )

    assert target_position.tolist() == [0.05, -0.05, 0.05]
    assert np.linalg.norm(target_quat) == pytest.approx(1.0)


def test_apply_tcp_delta_composes_rotation_delta_in_command_frame():
    position = np.zeros(3, dtype=float)
    current_quat = _quat_xyzw_from_rpy(np.array([0.3, -0.4, 0.2], dtype=float))
    rotation_delta = np.array([0.2, 0.1, -0.3], dtype=float)
    action = np.array([0.0, 0.0, 0.0, *rotation_delta, 0.0], dtype=float)

    _, target_quat = apply_tcp_delta(
        position,
        current_quat,
        action,
        max_translation_delta=0.05,
        max_rotation_delta=1.0,
    )

    delta_quat = _quat_xyzw_from_rpy(rotation_delta)
    expected_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    assert _same_orientation(target_quat, expected_quat)
