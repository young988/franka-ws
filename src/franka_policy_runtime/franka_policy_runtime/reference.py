"""Reference conversion helpers for policy actions.

Policy actions follow IsaacLab ``DifferentialInverseKinematicsAction``:
``[dx, dy, dz, ax, ay, az, gripper]``.
The first six values are relative TCP pose deltas.  The angular part is
axis-angle, and both translation and rotation are scaled before application.
The seventh value is binary gripper command: negative closes, non-negative opens.
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


def _quat_xyzw_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)


def _quat_xyzw_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1.0e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = vec / angle
    half = angle * 0.5
    sin_half = math.sin(half)
    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        math.cos(half),
    ], dtype=np.float64)


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
    action_scale: float,
    rotation_format: str = "axis_angle",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply IsaacLab-style relative TCP delta in the command/base frame.

    Translation is added directly.  Rotation delta is axis-angle by default
    (matching IsaacLab ``apply_delta_pose``).  Set *rotation_format* to
    ``"rpy"`` for OpenVLA-style delta roll/pitch/yaw.

    Quaternion order throughout is ROS xyzw.
    """
    position = np.asarray(current_position, dtype=np.float64)
    quat_xyzw = np.asarray(current_quat_xyzw, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"current_position must have shape (3,), got {position.shape}")
    if quat_xyzw.shape != (4,):
        raise ValueError(f"current_quat_xyzw must have shape (4,), got {quat_xyzw.shape}")

    tcp_delta, _ = split_policy_action(action)
    scaled_delta = tcp_delta * float(action_scale)
    translation = scaled_delta[:3]
    rotation_delta = scaled_delta[3:6]
    target_position = position + translation
    current_quat = quat_xyzw / np.linalg.norm(quat_xyzw)
    if rotation_format == "rpy":
        delta_quat = _quat_xyzw_from_rpy(rotation_delta)
    elif rotation_format == "axis_angle":
        delta_quat = _quat_xyzw_from_axis_angle(rotation_delta)
    else:
        raise ValueError(f"unknown rotation_format: {rotation_format!r}")
    target_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    return target_position, target_quat


def step_toward_pose(
    current_position: np.ndarray,
    current_quat_xyzw: np.ndarray,
    target_position: np.ndarray,
    target_quat_xyzw: np.ndarray,
    *,
    max_translation_step: float,
    max_rotation_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Step from current toward target, clamped in Cartesian space.

    Returns a pose at most *max_translation_step* in translation norm and
    *max_rotation_step* in rotation angle away from current.
    """
    position = np.asarray(current_position, dtype=np.float64)
    quat_xyzw = np.asarray(current_quat_xyzw, dtype=np.float64)
    target_position_arr = np.asarray(target_position, dtype=np.float64)
    target_quat_arr = np.asarray(target_quat_xyzw, dtype=np.float64)

    # ---- translation ----
    max_translation = abs(float(max_translation_step))
    delta_position = target_position_arr - position
    delta_norm = float(np.linalg.norm(delta_position))
    if delta_norm > max_translation:
        delta_position = delta_position * (max_translation / delta_norm)
    stepped_position = position + delta_position

    # ---- rotation ----
    current_quat = quat_xyzw / np.linalg.norm(quat_xyzw)
    target_quat = target_quat_arr / np.linalg.norm(target_quat_arr)
    if float(np.dot(current_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    delta_quat = _quat_multiply_xyzw(
        target_quat,
        np.array([-current_quat[0], -current_quat[1], -current_quat[2], current_quat[3]], dtype=np.float64),
    )
    delta_quat = delta_quat / np.linalg.norm(delta_quat)
    angle = 2.0 * math.atan2(float(np.linalg.norm(delta_quat[:3])), float(abs(delta_quat[3])))
    max_rotation = abs(float(max_rotation_step))
    if angle > max_rotation:
        axis_norm = float(np.linalg.norm(delta_quat[:3]))
        if axis_norm < 1.0e-12:
            return stepped_position, current_quat
        axis = delta_quat[:3] / axis_norm
        delta_quat = _quat_xyzw_from_axis_angle(axis * max_rotation)
    stepped_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    return stepped_position, stepped_quat


def gripper_width_from_binary_action(action_value: float, *, min_width: float, max_width: float) -> float:
    return float(min_width if float(action_value) < 0.0 else max_width)


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
