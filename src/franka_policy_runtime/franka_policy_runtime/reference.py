"""Reference conversion helpers for policy actions.

Policy actions are end-effector deltas:
``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.
The arm controller consumes joint references, so runtime code must convert the
first six dimensions through TF + IK before publishing to the controller.  The
seventh dimension is handled separately through the Franka gripper action API.
"""

from __future__ import annotations

import math

import numpy as np


def validate_action(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action must be finite")
    return arr


def split_policy_action(action: np.ndarray) -> tuple[np.ndarray, float]:
    arr = validate_action(action)
    return arr[:6].copy(), float(arr[6])


def _rpy_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_xyzw_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    quat = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    quat = np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def apply_tcp_delta(
    current_position: np.ndarray,
    current_quat_xyzw: np.ndarray,
    action: np.ndarray,
    *,
    max_translation_delta: float,
    max_rotation_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a clipped TCP delta in the command/base frame.

    Translation is added directly in the command frame. Rotation deltas are RPY
    parameters of an incremental command-frame rotation, which is composed with
    the current quaternion instead of added to the current Euler angles.
    """
    position = np.asarray(current_position, dtype=np.float64)
    quat_xyzw = np.asarray(current_quat_xyzw, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"current_position must have shape (3,), got {position.shape}")
    if quat_xyzw.shape != (4,):
        raise ValueError(f"current_quat_xyzw must have shape (4,), got {quat_xyzw.shape}")

    tcp_delta, _ = split_policy_action(action)
    translation = np.clip(
        tcp_delta[:3],
        -float(max_translation_delta),
        float(max_translation_delta),
    )
    rotation_delta = np.clip(
        tcp_delta[3:6],
        -float(max_rotation_delta),
        float(max_rotation_delta),
    )
    target_position = position + translation
    current_quat = quat_xyzw / np.linalg.norm(quat_xyzw)
    delta_quat = _quat_xyzw_from_rpy(rotation_delta)
    target_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    return target_position, target_quat


def make_joint_trajectory(joint_names: list[str], positions: np.ndarray, duration_sec: float):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    point = JointTrajectoryPoint()
    point.positions = np.asarray(positions, dtype=float).tolist()
    sec = int(duration_sec)
    point.time_from_start = Duration(
        sec=sec,
        nanosec=int((duration_sec - sec) * 1_000_000_000),
    )

    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    msg.points.append(point)
    return msg
