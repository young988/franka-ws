"""Observation providers for policy runtime modes."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VLAObservation:
    ready: bool
    instruction: str
    image: np.ndarray | None = None


@dataclass(frozen=True)
class RLObservation:
    ready: bool
    terms: dict[str, np.ndarray] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    availability: dict[str, bool] = field(default_factory=dict)


def image_msg_to_array(msg: Any) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.height and msg.width:
        arr = arr.reshape((int(msg.height), int(msg.width), -1))
    return arr.copy()


class BaseObserver:
    """Thread-safe sink for sensor updates used to assemble policy observations."""

    def __init__(self, joint_names: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._joint_names = list(joint_names or [])
        self._image: np.ndarray | None = None
        self._joint_pos: np.ndarray | None = None
        self._joint_vel: np.ndarray | None = None
        self._tcp_position: np.ndarray | None = None
        self._tcp_quat: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._gripper_width: float | None = None

    def update_image(self, msg: Any) -> None:
        with self._lock:
            self._image = image_msg_to_array(msg)

    def update_joint_state(self, msg: Any) -> None:
        if not self._joint_names:
            return
        by_name = {name: i for i, name in enumerate(msg.name)}
        if not all(name in by_name for name in self._joint_names):
            return
        positions = np.asarray(msg.position, dtype=np.float64)
        velocities = np.asarray(msg.velocity, dtype=np.float64)
        joint_pos = np.array([positions[by_name[name]] for name in self._joint_names], dtype=np.float64)
        if velocities.size:
            joint_vel = np.array([velocities[by_name[name]] for name in self._joint_names], dtype=np.float64)
        else:
            joint_vel = np.zeros(len(self._joint_names), dtype=np.float64)
        with self._lock:
            self._joint_pos = joint_pos
            self._joint_vel = joint_vel

    def latest_joint_positions(self) -> np.ndarray | None:
        with self._lock:
            return None if self._joint_pos is None else self._joint_pos.copy()

    def update_tcp_pose(self, position: np.ndarray, quat_xyzw: np.ndarray) -> None:
        position_arr = np.asarray(position, dtype=np.float64)
        quat_arr = np.asarray(quat_xyzw, dtype=np.float64)
        if position_arr.shape != (3,):
            raise ValueError(f"tcp position must have shape (3,), got {position_arr.shape}")
        if quat_arr.shape != (4,):
            raise ValueError(f"tcp quaternion must have shape (4,), got {quat_arr.shape}")
        with self._lock:
            self._tcp_position = position_arr.copy()
            self._tcp_quat = quat_arr.copy()

    def update_last_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError(f"last action must have shape (7,), got {arr.shape}")
        with self._lock:
            self._last_action = arr.copy()

    def update_gripper_width(self, width: float) -> None:
        with self._lock:
            self._gripper_width = float(width)


class VLAObserver(BaseObserver):
    """Current OpenVLA observation provider: latest RGB image only."""

    def __init__(self, joint_names: list[str] | None = None, instruction: str = "") -> None:
        super().__init__(joint_names)
        self._instruction = str(instruction)

    def update_instruction(self, msg: Any) -> None:
        with self._lock:
            self._instruction = str(msg.data)

    def observe(self) -> VLAObservation:
        with self._lock:
            image = None if self._image is None else self._image.copy()
            instruction = self._instruction
        return VLAObservation(ready=image is not None, instruction=instruction, image=image)


class RLObserver(BaseObserver):
    """IsaacLab-style low-dimensional observation provider for RL policies."""

    def __init__(self, joint_names: list[str]) -> None:
        super().__init__(joint_names)

    def observe(self) -> RLObservation:
        with self._lock:
            image = None if self._image is None else self._image.copy()
            joint_pos = None if self._joint_pos is None else self._joint_pos.copy()
            joint_vel = None if self._joint_vel is None else self._joint_vel.copy()
            tcp_position = None if self._tcp_position is None else self._tcp_position.copy()
            tcp_quat = None if self._tcp_quat is None else self._tcp_quat.copy()
            last_action = None if self._last_action is None else self._last_action.copy()
            gripper_width = self._gripper_width

        ready = joint_pos is not None and joint_vel is not None and tcp_position is not None and tcp_quat is not None
        terms: dict[str, np.ndarray] = {
            "joint_pos": np.zeros(len(self._joint_names), dtype=np.float64) if joint_pos is None else joint_pos,
            "joint_vel": np.zeros(len(self._joint_names), dtype=np.float64) if joint_vel is None else joint_vel,
            "eef_pos": np.zeros(3, dtype=np.float64) if tcp_position is None else tcp_position,
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if tcp_quat is None else tcp_quat,
            "gripper_pos": np.array([0.0 if gripper_width is None else gripper_width], dtype=np.float64),
            "last_action": np.zeros(7, dtype=np.float64) if last_action is None else last_action,
            "object_pose_in_eef": np.zeros(7, dtype=np.float64),
        }
        images = {} if image is None else {"image": image}
        availability = {
            "joint_pos": joint_pos is not None,
            "joint_vel": joint_vel is not None,
            "eef_pos": tcp_position is not None,
            "eef_quat": tcp_quat is not None,
            "gripper_pos": gripper_width is not None,
            "last_action": last_action is not None,
            "object_pose_in_eef": False,
        }
        return RLObservation(ready=ready, terms=terms, images=images, availability=availability)
