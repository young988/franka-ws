"""Cartesian pose backend — dual-state target/commanded pose machine.

Maintains two pose states:
  * ``target_pose`` — the goal pose accumulated from policy actions via
    ``apply_tcp_delta_in_base_frame``.
  * ``commanded_pose`` — the pose actually sent to the controller on each
    tick, stepped toward ``target_pose`` via ``step_toward_pose``.

The backend owns all Cartesian smoothing and can resync from the measured
pose when drift exceeds a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_policy_runtime.reference import apply_tcp_delta_in_base_frame, step_toward_pose


@dataclass
class PoseState:
    """A 6-DOF pose in the base/command frame.

    Attributes:
        position: 3-element (x, y, z) translation in metres.
        quat_xyzw: 4-element quaternion in ROS xyzw order (x, y, z, w).
    """

    position: np.ndarray
    quat_xyzw: np.ndarray


class CartesianPoseBackend:
    """Dual-state Cartesian pose manager.

    Call ``reset(measured_pose)`` at startup, then ``ingest_action()`` for
    each policy output and ``step_commanded_pose()`` on each control tick.

    Parameters:
        action_scale: Linear and angular scaling applied to each policy
            action delta before accumulation.
        rotation_format: Either ``"axis_angle"`` (IsaacLab convention) or
            ``"rpy"`` (OpenVLA convention).
        max_translation_step_per_tick: Maximum translation (m) the commanded
            pose may advance per tick.
        max_rotation_step_per_tick: Maximum rotation (rad) the commanded
            pose may advance per tick.
        pose_sync_reset_threshold: If the measured translation drift from
            the commanded pose exceeds this value (m), ``maybe_resync``
            resets both states.  Defaults to infinity (never resync).
    """

    def __init__(
        self,
        *,
        action_scale: float,
        rotation_format: str,
        max_translation_step_per_tick: float,
        max_rotation_step_per_tick: float,
        pose_sync_reset_threshold: float = float("inf"),
    ) -> None:
        self._action_scale = float(action_scale)
        self._rotation_format = str(rotation_format)
        self._max_translation_step = float(max_translation_step_per_tick)
        self._max_rotation_step = float(max_rotation_step_per_tick)
        self._pose_sync_reset_threshold = float(pose_sync_reset_threshold)
        self.target_pose: PoseState | None = None
        self.commanded_pose: PoseState | None = None

    def reset(self, measured_pose: PoseState) -> None:
        """Set both ``target_pose`` and ``commanded_pose`` from measured."""
        pose = PoseState(
            position=np.asarray(measured_pose.position, dtype=float).copy(),
            quat_xyzw=np.asarray(measured_pose.quat_xyzw, dtype=float).copy(),
        )
        self.target_pose = PoseState(position=pose.position.copy(), quat_xyzw=pose.quat_xyzw.copy())
        self.commanded_pose = PoseState(position=pose.position.copy(), quat_xyzw=pose.quat_xyzw.copy())

    def ingest_action(self, action: np.ndarray) -> PoseState:
        """Accumulate a policy action onto ``target_pose``.

        The new target is computed from the **previous target**, not from
        the measured or commanded pose.  This ensures smooth accumulation
        even when the commanded pose is lagging behind.
        """
        if self.target_pose is None:
            raise RuntimeError("backend must be reset before ingesting actions")
        next_position, next_quat = apply_tcp_delta_in_base_frame(
            self.target_pose.position,
            self.target_pose.quat_xyzw,
            action,
            action_scale=self._action_scale,
            rotation_format=self._rotation_format,
        )
        self.target_pose = PoseState(position=next_position, quat_xyzw=next_quat)
        return self.target_pose

    def step_commanded_pose(self) -> PoseState:
        """Advance ``commanded_pose`` toward ``target_pose`` by one tick.

        The step magnitude is limited by ``max_translation_step_per_tick``
        and ``max_rotation_step_per_tick``.
        """
        if self.target_pose is None or self.commanded_pose is None:
            raise RuntimeError("backend must be reset before stepping")
        next_position, next_quat = step_toward_pose(
            current_position=self.commanded_pose.position,
            current_quat_xyzw=self.commanded_pose.quat_xyzw,
            target_position=self.target_pose.position,
            target_quat_xyzw=self.target_pose.quat_xyzw,
            max_translation_step=self._max_translation_step,
            max_rotation_step=self._max_rotation_step,
        )
        self.commanded_pose = PoseState(position=next_position, quat_xyzw=next_quat)
        return self.commanded_pose

    def maybe_resync(self, measured_pose: PoseState) -> bool:
        """Reset both poses from measured if translation drift exceeds threshold.

        Returns ``True`` if a resync occurred.
        """
        if self.commanded_pose is None:
            self.reset(measured_pose)
            return True
        measured_position = np.asarray(measured_pose.position, dtype=float)
        drift = float(np.linalg.norm(measured_position - self.commanded_pose.position))
        if drift > self._pose_sync_reset_threshold:
            self.reset(measured_pose)
            return True
        return False
