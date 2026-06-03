"""TCP pose conversion helpers shared by runtime nodes."""

from __future__ import annotations

import numpy as np


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


def _quat_multiply_raw_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=float)
    rx, ry, rz, rw = np.asarray(right, dtype=float)
    return np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=float)


def quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    quat = _quat_multiply_raw_xyzw(left, right)
    return quat / np.linalg.norm(quat)


def rotate_vector_xyzw(quat_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
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
    quat = quat_multiply_xyzw(parent_quat, child_quat)
    return position, quat
