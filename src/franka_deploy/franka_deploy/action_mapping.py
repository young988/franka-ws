"""OpenVLA action helpers for planner-based and servo-based deployment.

OpenVLA bridge_orig outputs a 7-d delta action in the **world/base frame**:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

where:
  - [dx, dy, dz] = position delta in world frame (metres)
                     computed as pos_world[t+1] - pos_world[t]
  - [droll, dpitch, dyaw] = Euler-angle delta in world frame (radians)
                             computed as rpy_world[t+1] - rpy_world[t]
  - gripper = normalized in [-1, +1] (-1 = close, +1 = open)

This follows the Bridge V2 convention established by relabel_bridge_actions():
  movement_actions = state[1:, :6] - state[:-1, :6]
where state = [x, y, z, roll, pitch, yaw] in world/base frame.

For the planner path we clip the delta, compute an absolute target pose,
and hand it to MoveIt for IK + planning + execution.
For the servo path we convert the delta to a twist velocity command.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Vector3
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# OpenVLA 7-d action layout
DIM_X = 0    # dx
DIM_Y = 1    # dy
DIM_Z = 2    # dz
DIM_ROLL = 3   # droll
DIM_PITCH = 4  # dpitch
DIM_YAW = 5    # dyaw
DIM_GRIP = 6   # gripper


# ---------------------------------------------------------------------------
#  Safety limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaLimits:
    """Per-step position / orientation clipping bounds (in world/base frame)."""
    max_linear_step: float   # metres
    max_angular_step: float  # radians


DEFAULT_LIMITS = DeltaLimits(max_linear_step=0.05, max_angular_step=0.25)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def validate_action(action: Iterable[float]) -> np.ndarray:
    """Return (7,) float64 array, or raise ValueError."""
    arr = np.asarray(action, dtype=float)
    if arr.shape != (7,):
        raise ValueError(f'OpenVLA action must have shape (7,), got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError('OpenVLA action contains NaN or Inf')
    return arr


# ---------------------------------------------------------------------------
#  Delta clipping
# ---------------------------------------------------------------------------

def clip_delta(
    action: Iterable[float],
    limits: Optional[DeltaLimits] = None,
) -> np.ndarray:
    """Clip each component of the 7-d VLA action to safe bounds.

    Position deltas (first 3) are clipped to ``max_linear_step``;
    rotation deltas (next 3) to ``max_angular_step``; gripper is kept as-is.
    """
    if limits is None:
        limits = DEFAULT_LIMITS
    arr = validate_action(action)
    arr[0:3] = np.clip(arr[0:3], -limits.max_linear_step, limits.max_linear_step)
    arr[3:6] = np.clip(arr[3:6], -limits.max_angular_step, limits.max_angular_step)
    return arr


# ---------------------------------------------------------------------------
#  Pose math
# ---------------------------------------------------------------------------

def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Build geometry_msgs/Quaternion from RPY (radians, intrinsic ZYX)."""
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()  # [x, y, z, w]
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def rpy_from_quaternion(q: Quaternion) -> Tuple[float, float, float]:
    """Extract intrinsic XYZ RPY (radians) from geometry_msgs/Quaternion."""
    r = Rotation.from_quat([q.x, q.y, q.z, q.w])
    roll, pitch, yaw = r.as_euler('xyz')
    return float(roll), float(pitch), float(yaw)


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Quaternion product q1 * q2 (scipy convention: r1 composed with r2)."""
    r1 = Rotation.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = Rotation.from_quat([q2.x, q2.y, q2.z, q2.w])
    r = r1 * r2
    x, y, z, w = r.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)


def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Return the inverse (conjugate) of a unit quaternion."""
    return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)


def compute_target_pose(
    current: Pose,
    delta: Iterable[float],
) -> Pose:
    """Return absolute target pose = current_pose + delta.

    **OpenVLA bridge_orig outputs delta in the WORLD/BASE frame**, not the
    TCP frame.  This is the standard bridge_data v2 convention:
        - [dx, dy, dz] = position delta in **world frame** (metres)
        - [droll, dpitch, dyaw] = euler-angle delta in **world frame** (radians)
        - gripper = normalized in [-1, 1] (unmasked in norm_stats)

    The delta is defined by relabel_bridge_actions():
        movement_actions = state[1:, :6] - state[:-1, :6]
    where state = [x, y, z, roll, pitch, yaw] in the world/base frame.

    Therefore the correct target computation is:
        target.pos  = current.pos + [dx, dy, dz]          (world-frame addition)
        target.rpy  = current.rpy + [droll, dpitch, dyaw] (Euler-angle addition)

    NOTE: We use **Euler-angle addition** (NOT quaternion pre-/post-multiplication)
    because the delta is an RPY *parameter difference*, not a rotation matrix.
    R(Δrpy) * R(rpy) ≠ R(rpy + Δrpy) in general — they are different operations.

    Parameters
    ----------
    current : geometry_msgs/Pose
        Current TCP pose in the world frame.
    delta : (7,) or (6,) array
        [dx, dy, dz, droll, dpitch, dyaw] in world/base frame (bridge_orig).

    Returns
    -------
    geometry_msgs/Pose
        Target pose in the world frame.
    """
    arr = np.asarray(delta, dtype=float).flatten()
    if arr.shape[0] < 6:
        raise ValueError(f'delta must have at least 6 elements, got {arr.shape[0]}')

    dx, dy, dz = arr[0], arr[1], arr[2]
    droll, dpitch, dyaw = arr[3], arr[4], arr[5]

    # -- position (delta is already in world frame → direct addition) --
    target = Pose()
    target.position.x = current.position.x + dx
    target.position.y = current.position.y + dy
    target.position.z = current.position.z + dz

    # -- orientation (Euler-angle addition, matching bridge_orig convention) --
    current_roll, current_pitch, current_yaw = rpy_from_quaternion(current.orientation)
    target_roll = current_roll + droll
    target_pitch = current_pitch + dpitch
    target_yaw = current_yaw + dyaw
    target.orientation = quaternion_from_rpy(target_roll, target_pitch, target_yaw)

    return target


def pose_distance(p1: Pose, p2: Pose) -> float:
    """Euclidean distance between two pose positions (metres)."""
    return float(math.sqrt(
        (p1.position.x - p2.position.x) ** 2
        + (p1.position.y - p2.position.y) ** 2
        + (p1.position.z - p2.position.z) ** 2
    ))


# ---------------------------------------------------------------------------
#  Servo (twist conversion)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwistLimits:
    """Velocity and per-step clipping bounds for servo twist commands."""
    max_linear_velocity: float   # m/s
    max_angular_velocity: float  # rad/s
    max_linear_step: float       # m per action
    max_angular_step: float      # rad per action


def action_to_twist(
    action: np.ndarray,
    dt: float,
    limits: Optional[TwistLimits] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 7-d VLA action to linear/angular velocity vectors.

    The VLA action is a world-frame delta [dx, dy, dz, droll, dpitch, dyaw].
    We divide by the actual time delta dt to get velocity commands, then
    clip to the configured limits.

    Parameters
    ----------
    action : np.ndarray
        7-d action in world/base frame (bridge_orig convention).
    dt : float
        Time delta in seconds since the last action.
    limits : TwistLimits, optional
        Velocity and step clipping bounds.

    Returns
    -------
    linear : np.ndarray (3,)
        Linear velocity [vx, vy, vz] in world frame (m/s).
    angular : np.ndarray (3,)
        Angular velocity [wx, wy, wz] (rad/s) — Euler-angle rates.
    """
    if limits is None:
        limits = TwistLimits(
            max_linear_velocity=0.05,
            max_angular_velocity=0.25,
            max_linear_step=0.002,
            max_angular_step=0.01,
        )

    arr = validate_action(action)
    linear_raw = arr[0:3].copy()
    angular_raw = arr[3:6].copy()

    # Per-step clipping (prevents large jumps from stale actions)
    linear_raw = np.clip(linear_raw, -limits.max_linear_step, limits.max_linear_step)
    angular_raw = np.clip(angular_raw, -limits.max_angular_step, limits.max_angular_step)

    # Convert delta to velocity: v = Δx / dt
    linear = linear_raw / max(dt, 1e-6)
    angular = angular_raw / max(dt, 1e-6)

    # Velocity clipping
    lin_norm = float(np.linalg.norm(linear))
    if lin_norm > limits.max_linear_velocity and lin_norm > 0:
        linear *= limits.max_linear_velocity / lin_norm

    ang_norm = float(np.linalg.norm(angular))
    if ang_norm > limits.max_angular_velocity and ang_norm > 0:
        angular *= limits.max_angular_velocity / ang_norm

    return linear, angular


# ---------------------------------------------------------------------------
#  Gripper
# ---------------------------------------------------------------------------

def gripper_should_close(action: Iterable[float], threshold: float) -> bool:
    """Decide gripper state from VLA action.

    Bridge v2 / OpenVLA gripper convention (after normalization):
        -1.0 = close,  +1.0 = open

    So we close when the gripper action is **below** the threshold.
    """
    arr = validate_action(action)
    return bool(arr[DIM_GRIP] <= threshold)
