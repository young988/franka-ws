"""Pose math utilities: quaternion operations, TCP deltas, trajectory helpers.

Combines former ``tcp_pose.py`` (TF/msg conversions, quaternion ops) and
``motion_conversion.py`` (action deltas, trajectory building, gripper logic).
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# ---------------------------------------------------------------------------
# ROS message → numpy conversions
# ---------------------------------------------------------------------------


def pose_msg_to_arrays(pose) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float),
        np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ], dtype=float),
    )


def transform_msg_to_arrays(transform) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ], dtype=float),
        np.array([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ], dtype=float),
    )

# ---------------------------------------------------------------------------
# Quaternion operations (xyzw order, ROS convention)
# ---------------------------------------------------------------------------


def _quat_multiply_raw_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (xyzw) without normalization."""
    lx, ly, lz, lw = np.asarray(left, dtype=float)
    rx, ry, rz, rw = np.asarray(right, dtype=float)
    return np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=float)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (xyzw) and normalize the result."""
    quat = _quat_multiply_raw_xyzw(left, right)
    return quat / np.linalg.norm(quat)


def quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Public alias for quaternion multiplication (xyzw, normalized)."""
    return _quat_multiply_xyzw(left, right)


def rotate_vector_xyzw(quat_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate *vector* by the unit quaternion *quat_xyzw*."""
    quat = np.asarray(quat_xyzw, dtype=float)
    quat = quat / np.linalg.norm(quat)
    vec_quat = np.array([vector[0], vector[1], vector[2], 0.0], dtype=float)
    inv = np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=float)
    return _quat_multiply_raw_xyzw(_quat_multiply_raw_xyzw(quat, vec_quat), inv)[:3]


def compose_pose_xyzw(
    parent_position: np.ndarray,
    parent_quat_xyzw: np.ndarray,
    child_position: np.ndarray,
    child_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_position_arr = np.asarray(parent_position, dtype=float)
    child_position_arr = np.asarray(child_position, dtype=float)
    parent_quat = np.asarray(parent_quat_xyzw, dtype=float)
    child_quat = np.asarray(child_quat_xyzw, dtype=float)
    position = parent_position_arr + rotate_vector_xyzw(parent_quat, child_position_arr)
    quat = _quat_multiply_xyzw(parent_quat, child_quat)
    return position, quat


def invert_pose_xyzw(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(quat_xyzw, dtype=float)
    quat = quat / np.linalg.norm(quat)
    inverse_quat = np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=float)
    inverse_position = -rotate_vector_xyzw(inverse_quat, np.asarray(position, dtype=float))
    return inverse_position, inverse_quat


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to 3x3 rotation matrix."""
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


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """Convert depth array to meters (assumes mm if max > 10)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.size and np.nanmax(depth) > 10.0:
        return depth / 1000.0
    return depth


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

# ---------------------------------------------------------------------------
# Policy action helpers
# ---------------------------------------------------------------------------


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


def policy_action_to_joint_positions(action: np.ndarray) -> np.ndarray:
    """Interpret a 7D policy action as absolute FR3 joint angles in radians."""
    return validate_action(action).copy()


def policy_action_to_cartesian_delta(
    action: np.ndarray,
    *,
    action_scale: float,
    rotation_format: str = "axis_angle",
) -> np.ndarray:
    """Convert a policy action to a base-frame Cartesian delta twist."""
    tcp_delta, _ = split_policy_action(action)
    scaled_delta = tcp_delta * float(action_scale)
    if rotation_format == "axis_angle":
        return scaled_delta
    if rotation_format != "rpy":
        raise ValueError(f"unknown rotation_format: {rotation_format!r}")

    quat = _quat_xyzw_from_rpy(scaled_delta[3:6])
    vector_norm = float(np.linalg.norm(quat[:3]))
    if vector_norm < 1.0e-12:
        rotation_vector = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * math.atan2(vector_norm, float(quat[3]))
        rotation_vector = quat[:3] * (angle / vector_norm)
    return np.concatenate((scaled_delta[:3], rotation_vector))


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
    """Map a binary-policy gripper dimension to physical width.

    Uses a mid-point threshold so that both ``0.0/1.0`` (bridge_orig) and
    ``-1.0/+1.0`` (fractal) conventions translate correctly to open/close.
    """
    return float(min_width if float(action_value) < 0.5 else max_width)


def make_joint_trajectory(
    joint_names: list[str],
    positions: np.ndarray,
    duration_sec: float,
    *,
    start_positions: np.ndarray | None = None,
    start_delay_sec: float = 0.1,
):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    target_positions = np.asarray(positions, dtype=float).tolist()
    zeros = [0.0] * len(target_positions)
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)

    if start_positions is not None:
        start = JointTrajectoryPoint()
        start.positions = np.asarray(start_positions, dtype=float).tolist()
        start.velocities = zeros.copy()
        start.accelerations = zeros.copy()
        start_sec = int(start_delay_sec)
        start.time_from_start = Duration(
            sec=start_sec,
            nanosec=int((start_delay_sec - start_sec) * 1_000_000_000),
        )
        msg.points.append(start)

    point = JointTrajectoryPoint()
    point.positions = target_positions
    point.velocities = zeros.copy()
    point.accelerations = zeros.copy()
    sec = int(duration_sec)
    point.time_from_start = Duration(
        sec=sec,
        nanosec=int((duration_sec - sec) * 1_000_000_000),
    )

    msg.points.append(point)
    return msg


# ---------------------------------------------------------------------------
# Action dimension labelling (shared by runtime and test tooling)
# ---------------------------------------------------------------------------

_DIM_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def action_dim_label(action: np.ndarray) -> str:
    """Return a human-readable label for which dimensions are non-zero.

    Examples: ``"+dx"``, ``"-ry"``, ``"gripper_open"``, ``"zero"``.
    """
    arr = np.asarray(action, dtype=float)
    nonzero = [i for i in range(len(_DIM_LABELS)) if abs(float(arr[i])) > 1e-9]
    if len(nonzero) == 0:
        return "zero"
    labels: list[str] = []
    for idx in nonzero:
        val = float(arr[idx])
        name = _DIM_LABELS[idx]
        if idx == 6:
            labels.append("gripper_open" if val > 0.0 else "gripper_close")
        else:
            sign = "+" if val > 0.0 else "-"
            labels.append(f"{sign}{name}")
    return ",".join(labels)


# ---------------------------------------------------------------------------
# Dummy observer for testing (no sensor dependency)
# ---------------------------------------------------------------------------

from franka_policy_runtime.observers.base import BackendObservation, BaseObserver  # noqa: E402


class DummyObserver(BaseObserver):
    """Observer that always reports ready with an empty payload."""

    def observe(self) -> BackendObservation:
        return BackendObservation(ready=True, payload={})


# ---------------------------------------------------------------------------
# AnyGrasp helpers
# ---------------------------------------------------------------------------


def anygrasp_action_to_base_poses(
    action: np.ndarray,
    tcp_position: np.ndarray,
    tcp_quat_xyzw: np.ndarray,
    *,
    approach_distance: float = 0.1,
    grasp_to_tcp_rotvec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decompose an AnyGrasp action into pregrasp, grasp, orientation, and width.

    The action is ``[x, y, z, rx, ry, rz, width]`` where the first six
    values are an absolute position + axis-angle rotation in the TCP frame,
    and the last value is the requested gripper width.

    Returns ``(pregrasp_pos, grasp_pos, quat_xyzw, gripper_width)``.
    """
    grasp_position = np.asarray(action[:3], dtype=np.float64) + np.asarray(tcp_position, dtype=np.float64)
    gripper_width = float(action[6])

    if grasp_to_tcp_rotvec is not None:
        try:
            import cv2
            rot, _ = cv2.Rodrigues(np.asarray(grasp_to_tcp_rotvec, dtype=np.float64))
        except ImportError:
            rot = np.eye(3, dtype=np.float64)
    else:
        rot = np.eye(3, dtype=np.float64)

    # Approach direction: grasp frame z-axis mapped to TCP frame
    approach_dir = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pregrasp_position = grasp_position - approach_dir * approach_distance

    # Return TCP orientation as-is; grasp_to_tcp_rotvec affects approach direction only
    quat_xyzw = np.asarray(tcp_quat_xyzw, dtype=np.float64)

    return pregrasp_position, grasp_position, quat_xyzw, gripper_width
