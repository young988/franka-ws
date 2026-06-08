from __future__ import annotations

import math
from collections.abc import Sequence


FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

FR3_READY_JOINTS = [-0.0058975655, -1.4723245927, 0.0058916495, -2.8, -0.0028241465, 1.6555054429, 0.8048119146]

FR3_LOWER_LIMITS = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
FR3_UPPER_LIMITS = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def map_servo_offsets_to_joints(
    servo_offsets_deg: Sequence[float],
    *,
    base_positions_rad: Sequence[float],
    arm_servo_indices: Sequence[int],
    signs: Sequence[float],
    scales: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> list[float]:
    targets: list[float] = []
    for joint_index, servo_index in enumerate(arm_servo_indices):
        base = float(base_positions_rad[joint_index])
        offset_deg = float(servo_offsets_deg[servo_index]) if servo_index < len(servo_offsets_deg) else 0.0
        delta = math.radians(offset_deg) * float(signs[joint_index]) * float(scales[joint_index])
        targets.append(clamp(base + delta, float(lower_limits[joint_index]), float(upper_limits[joint_index])))
    return targets


def map_gripper_offset_to_width(
    offset_deg: float,
    *,
    open_offset_deg: float,
    closed_offset_deg: float,
    min_width: float,
    max_width: float,
) -> float:
    span = closed_offset_deg - open_offset_deg
    if abs(span) < 1e-9:
        return max_width
    closed_ratio = clamp((offset_deg - open_offset_deg) / span, 0.0, 1.0)
    return max_width - closed_ratio * (max_width - min_width)
