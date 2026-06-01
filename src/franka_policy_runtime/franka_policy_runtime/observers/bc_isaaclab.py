"""IsaacLab stack BC observer — structured robot-state observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from franka_policy_runtime.observers.base import (
    BackendObservation,
    BaseObserver,
    ObjectPoseProvider,
    estimate_object_pose_in_eef,
)


class IsaacLabStackBCObserver(BaseObserver):
    """Observation schema for IsaacLab stack BC checkpoints."""

    def __init__(
        self,
        joint_names: list[str],
        object_pose_provider: ObjectPoseProvider | None = None,
        object_provider: ObjectPoseProvider | None = None,
    ) -> None:
        super().__init__(joint_names)
        self._object_pose_provider = object_pose_provider or estimate_object_pose_in_eef
        self._object_provider = object_provider

    def observe(self) -> BackendObservation:
        with self._lock:
            images = self._copy_images_locked()
            joint_pos = None if self._joint_pos is None else self._joint_pos.copy()
            joint_vel = None if self._joint_vel is None else self._joint_vel.copy()
            tcp_position = None if self._tcp_position is None else self._tcp_position.copy()
            tcp_quat = None if self._tcp_quat is None else self._tcp_quat.copy()
            last_action = None if self._last_action is None else self._last_action.copy()
            gripper_width = self._gripper_width

        object_pose = self._object_pose_provider(self)
        if object_pose is not None:
            object_pose = np.asarray(object_pose, dtype=np.float64)
            if object_pose.shape != (7,):
                raise ValueError(f"object_pose_in_eef must have shape (7,), got {object_pose.shape}")
        object_term = None if self._object_provider is None else self._object_provider(self)
        if object_term is not None:
            object_term = np.asarray(object_term, dtype=np.float64)
            if object_term.shape != (39,):
                raise ValueError(f"object must have shape (39,), got {object_term.shape}")

        ready = (
            joint_pos is not None
            and joint_vel is not None
            and tcp_position is not None
            and tcp_quat is not None
            and (self._object_provider is None or object_term is not None)
        )
        terms: dict[str, np.ndarray] = {
            "joint_pos": np.zeros(len(self._joint_names), dtype=np.float64) if joint_pos is None else joint_pos,
            "joint_vel": np.zeros(len(self._joint_names), dtype=np.float64) if joint_vel is None else joint_vel,
            "eef_pos": np.zeros(3, dtype=np.float64) if tcp_position is None else tcp_position,
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if tcp_quat is None else tcp_quat,
            "gripper_pos": np.array(
                [0.0, 0.0] if gripper_width is None else [gripper_width * 0.5, gripper_width * 0.5],
                dtype=np.float64,
            ),
            "last_action": np.zeros(7, dtype=np.float64) if last_action is None else last_action,
        }
        availability = {
            "joint_pos": joint_pos is not None,
            "joint_vel": joint_vel is not None,
            "eef_pos": tcp_position is not None,
            "eef_quat": tcp_quat is not None,
            "gripper_pos": gripper_width is not None,
            "last_action": last_action is not None,
            "object_pose_in_eef": object_pose is not None,
            "object": object_term is not None,
        }
        if object_pose is not None:
            terms["object_pose_in_eef"] = object_pose
        if object_term is not None:
            terms["object"] = object_term
        return BackendObservation(
            ready=ready,
            payload={
                "terms": terms,
                "availability": availability,
                "images": images,
            },
        )
