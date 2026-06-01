"""Base observer and shared utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable

import numpy as np


ObjectPoseProvider = Callable[["BaseObserver"], np.ndarray | None]


@dataclass(frozen=True)
class BackendObservation:
    ready: bool
    payload: dict[str, Any] = field(default_factory=dict)


def image_msg_to_array(msg: Any) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.height and msg.width:
        arr = arr.reshape((int(msg.height), int(msg.width), -1))
    return arr.copy()


def depth_msg_to_array(msg: Any) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "")).lower()
    dtype = np.float32 if "32f" in encoding else np.uint16
    arr = np.frombuffer(msg.data, dtype=dtype)
    if msg.height and msg.width:
        arr = arr.reshape((int(msg.height), int(msg.width)))
    return arr.copy()


def camera_info_to_k(msg: Any) -> np.ndarray:
    return np.asarray(msg.k, dtype=np.float64).reshape(3, 3)


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.size and np.nanmax(depth) > 10.0:
        return depth / 1000.0
    return depth


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def estimate_object_pose_in_eef(_observer: BaseObserver | None) -> np.ndarray | None:
    """Default obj2ee provider hook.

    Real implementations can call YOLO/depth/TF or another perception program
    and return a 7D pose [x, y, z, qx, qy, qz, qw].
    """
    return None


class BaseObserver:
    """Thread-safe sensor sink shared by backend-specific observers."""

    def __init__(self, joint_names: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._joint_names = list(joint_names or [])
        self._images: dict[str, np.ndarray] = {}
        self._depths: dict[str, np.ndarray] = {}
        self._camera_infos: dict[str, np.ndarray] = {}
        self._tf_buffer: Any | None = None
        self._joint_pos: np.ndarray | None = None
        self._joint_vel: np.ndarray | None = None
        self._tcp_position: np.ndarray | None = None
        self._tcp_quat: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._gripper_width: float | None = None

    def update_image(self, msg: Any, name: str = "eye_to_hand") -> None:
        with self._lock:
            self._images[str(name)] = image_msg_to_array(msg)

    def update_depth(self, msg: Any, name: str = "eye_to_hand") -> None:
        with self._lock:
            self._depths[str(name)] = depth_msg_to_array(msg)

    def update_camera_info(self, msg: Any, name: str = "eye_to_hand") -> None:
        with self._lock:
            self._camera_infos[str(name)] = camera_info_to_k(msg)

    def set_tf_buffer(self, tf_buffer: Any) -> None:
        self._tf_buffer = tf_buffer

    @property
    def tf_buffer(self) -> Any | None:
        return self._tf_buffer

    def sensor_snapshot(self, name: str = "eye_to_hand") -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        with self._lock:
            image = self._images.get(name)
            depth = self._depths.get(name)
            k_matrix = self._camera_infos.get(name)
            return (
                None if image is None else image.copy(),
                None if depth is None else depth.copy(),
                None if k_matrix is None else k_matrix.copy(),
            )

    def _copy_images_locked(self) -> dict[str, np.ndarray]:
        return {name: image.copy() for name, image in self._images.items()}

    def _primary_image_locked(self) -> np.ndarray | None:
        image = self._images.get("eye_to_hand")
        return None if image is None else image.copy()

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
