"""Tests for CartesianPoseBackend — pose accumulation, stepping, and resync."""

import numpy as np
import pytest

from franka_policy_runtime.cartesian_backend import CartesianPoseBackend, PoseState
from franka_policy_runtime.reference import _quat_xyzw_from_axis_angle


def _same_orientation(actual: np.ndarray, expected: np.ndarray) -> bool:
    return abs(float(np.dot(actual, expected))) == pytest.approx(1.0)


def test_backend_initializes_from_measured_pose():
    backend = CartesianPoseBackend(
        action_scale=0.5,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 8,
    )
    measured = PoseState(
        position=np.array([0.4, 0.1, 0.2], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )

    backend.reset(measured)

    assert backend.target_pose.position.tolist() == pytest.approx([0.4, 0.1, 0.2])
    assert backend.commanded_pose.position.tolist() == pytest.approx([0.4, 0.1, 0.2])


def test_backend_accumulates_target_pose_from_previous_target_not_measured_pose():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 8,
    )
    measured = PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )
    backend.reset(measured)

    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    backend.ingest_action(action)
    backend.ingest_action(action)

    assert backend.target_pose.position.tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_backend_step_returns_commanded_pose_limited_toward_target():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 6,
    )
    backend.reset(PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))
    backend.ingest_action(np.array([0.2, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0], dtype=float))

    next_pose = backend.step_commanded_pose()

    assert next_pose.position.tolist() == pytest.approx([0.05, 0.0, 0.0])
    expected_quat = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi / 6], dtype=float))
    assert _same_orientation(next_pose.quat_xyzw, expected_quat)


def test_backend_resyncs_when_measured_pose_drift_exceeds_threshold():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 6,
        pose_sync_reset_threshold=0.2,
    )
    backend.reset(PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))
    backend.ingest_action(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    backend.maybe_resync(PoseState(
        position=np.array([0.5, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))

    assert backend.target_pose.position.tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert backend.commanded_pose.position.tolist() == pytest.approx([0.5, 0.0, 0.0])
