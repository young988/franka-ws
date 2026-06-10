#!/usr/bin/env python3
"""Replay a recorded teleop episode on the FR3 robot.

Two modes:

- **impedance** (default): publishes interpolated JointState to
  /uarm_leader/joint_states, tracked by the follower_controller PD loop
  at 1000 Hz.  Smooth, no goal gaps.

- **jtc**: sends discrete FollowJointTrajectory goals.  Simpler setup
  (uses fr3_arm_controller) but has pauses between steps.

Usage:
    ros2 run franka_telep episode_replay --ros-args -p episode_id:=0 -p speed:=1.0
    ros2 run franka_telep episode_replay --ros-args \\
        -p episode_path:=/path/to/episode_000000 -p mode:=impedance
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp, Move
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_telep.franka_mapping import FR3_JOINT_NAMES

_NUM_JOINTS = 7


def _make_trajectory_msg(
    joint_names: list[str],
    positions: list[float],
    duration_sec: float,
) -> JointTrajectory:
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = list(map(float, positions))
    point.velocities = [0.0] * len(positions)
    point.accelerations = [0.0] * len(positions)
    sec = int(duration_sec)
    point.time_from_start.sec = sec
    point.time_from_start.nanosec = int((duration_sec - sec) * 1_000_000_000)
    msg.points.append(point)
    return msg


class EpisodeReplay(Node):
    """Replay recorded joint positions and gripper commands."""

    def __init__(self) -> None:
        super().__init__("episode_replay")

        self.declare_parameter("data_root", "~/franka_openvla_data/franka_teleop/raw")
        self.declare_parameter("episode_id", 0)
        self.declare_parameter("episode_path", "")
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("mode", "impedance")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("trajectory_duration_sec", 0.3)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)
        self.declare_parameter("leader_joint_state_topic", "/uarm_leader/joint_states")
        self.declare_parameter("trajectory_action", "/fr3_arm_controller/follow_joint_trajectory")
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_grasp_action", "/franka_gripper/grasp")
        self.declare_parameter("gripper_speed", 0.1)
        self.declare_parameter("gripper_force", 60.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("require_home_ready", False)
        self.declare_parameter("home_ready_topic", "/franka_teleop/home_ready")

        # ── load episode ──────────────────────────────────────────
        path_str = str(self.get_parameter("episode_path").value).strip()
        if path_str:
            ep_dir = Path(path_str).expanduser()
        else:
            root = Path(str(self.get_parameter("data_root").value)).expanduser()
            ep_id = int(self.get_parameter("episode_id").value)
            ep_dir = root / f"episode_{ep_id:06d}"

        npz = ep_dir / "steps.npz"
        if not npz.exists():
            raise FileNotFoundError(f"steps.npz not found at {npz}")

        data = np.load(npz)
        self._joints: np.ndarray = data["joint_positions"].astype(np.float64)  # (N, 7)
        self._gripper: np.ndarray = data["action"][:, 6].astype(np.float64)    # (N,)
        self._ts: np.ndarray = data["timestamp_sec"]
        self._num_steps = len(self._joints)
        self._gripper_indices = self._find_gripper_events()

        # Compute per-step interval for linear interpolation
        self._intervals = np.diff(self._ts)
        self._intervals = np.append(self._intervals, self._intervals[-1])  # last step

        self._mode = str(self.get_parameter("mode").value).lower()
        if self._mode not in ("impedance", "jtc"):
            raise ValueError(f"mode must be 'impedance' or 'jtc', got {self._mode!r}")

        self.get_logger().info(
            f"Loaded {self._num_steps} steps from {ep_dir} "
            f"(duration={self._ts[-1] - self._ts[0]:.1f}s, mode={self._mode})"
        )

        self._current_gripper_binary: float = self._gripper[0]
        self._preview_mode = False

        # ── home ready ────────────────────────────────────────────
        self._home_ready = not bool(self.get_parameter("require_home_ready").value)
        if not self._home_ready:
            ready_qos = QoSProfile(depth=1)
            ready_qos.reliability = ReliabilityPolicy.RELIABLE
            ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(
                Bool, str(self.get_parameter("home_ready_topic").value),
                self._home_ready_cb, ready_qos,
            )
            self.get_logger().info("Waiting for Franka home ready before replay...")

        # ── impedance mode ────────────────────────────────────────
        if self._mode == "impedance":
            self._leader_pub = self.create_publisher(
                JointState,
                str(self.get_parameter("leader_joint_state_topic").value),
                10,
            )
            self._preview_mode = "preview" in str(self.get_parameter("leader_joint_state_topic").value)
            self._replay_start: float | None = None
            self._last_gripper_binary: float = float("nan")
            period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
            self._timer = self.create_timer(period, self._tick_impedance)
            self.get_logger().info(f"Impedance replay: publishing at {1.0/period:.0f} Hz")

        # ── JTC mode ──────────────────────────────────────────────
        else:
            self._index = 0
            self._goal_active = False
            self._last_gripper: float = float("nan")
            self._traj_client = ActionClient(
                self, FollowJointTrajectory,
                str(self.get_parameter("trajectory_action").value),
            )
            speed = max(0.1, float(self.get_parameter("speed").value))
            base = float(np.mean(self._intervals[:-1])) if self._num_steps >= 2 else 0.2
            period = max(0.05, base / speed)
            self._timer = self.create_timer(period, self._tick_jtc)
            self.get_logger().info(f"JTC replay: step_interval={period:.3f}s")

        # ── gripper clients (both modes) ──────────────────────────
        self._gripper_move = ActionClient(self, Move, str(self.get_parameter("gripper_move_action").value))
        self._gripper_grasp = ActionClient(self, Grasp, str(self.get_parameter("gripper_grasp_action").value))

    # ── helpers ───────────────────────────────────────────────────

    def _find_gripper_events(self) -> list[int]:
        """Return indices where gripper value changes."""
        events = [0]
        prev = self._gripper[0]
        for i in range(1, len(self._gripper)):
            if abs(float(self._gripper[i]) - float(prev)) > 0.01:
                events.append(i)
                prev = self._gripper[i]
        return events

    # ═══════════════════════════════════════════════════════════════
    #  Impedance mode — publish interpolated JointState
    # ═══════════════════════════════════════════════════════════════

    def _home_ready_cb(self, msg: Bool) -> None:
        if msg.data and not self._home_ready:
            self.get_logger().info("Franka home ready — starting replay")
        self._home_ready = bool(msg.data)

    def _tick_impedance(self) -> None:
        if not self._home_ready:
            self.get_logger().warn("Waiting for Franka home ready...", throttle_duration_sec=2.0)
            return
        speed = max(0.1, float(self.get_parameter("speed").value))
        now = self.get_clock().now().nanoseconds * 1.0e-9

        if self._replay_start is None:
            self._replay_start = now

        elapsed = (now - self._replay_start) * speed
        total_duration = float(self._ts[-1] - self._ts[0])

        if elapsed >= total_duration:
            if bool(self.get_parameter("loop").value):
                self._replay_start = now
                return
            self.get_logger().info(
                f"Replay finished ({self._num_steps} steps, {total_duration:.1f}s)")
            self._timer.cancel()
            self.destroy_node()
            rclpy.shutdown()
            return

        target_epoch = float(self._ts[0]) + elapsed
        joints = self._interpolate_joints(target_epoch)
        self._current_gripper_binary = self._gripper_binary_at(target_epoch)
        self._publish_joint_state(joints)
        self._handle_gripper_impedance()

    def _interpolate_joints(self, target_epoch: float) -> np.ndarray:
        """Linear interpolation between the two nearest recorded steps."""
        # Find right boundary
        idx = np.searchsorted(self._ts, target_epoch)
        if idx <= 0:
            return self._joints[0].copy()
        if idx >= self._num_steps:
            return self._joints[-1].copy()

        t_left = float(self._ts[idx - 1])
        t_right = float(self._ts[idx])
        span = t_right - t_left
        if span <= 1e-12:
            return self._joints[idx].copy()

        alpha = (target_epoch - t_left) / span
        return self._joints[idx - 1] + alpha * (self._joints[idx] - self._joints[idx - 1])

    def _publish_joint_state(self, positions: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names = [f"fr3_joint{i}" for i in range(1, _NUM_JOINTS + 1)]
        pos = positions.tolist()
        if self._preview_mode:
            finger_val = 0.04 if self._current_gripper_binary > 0.5 else 0.0
            names += ["fr3_finger_joint1", "fr3_finger_joint2"]
            pos += [finger_val, finger_val]
        msg.name = names
        msg.position = pos
        self._leader_pub.publish(msg)

    def _gripper_binary_at(self, target_epoch: float) -> float:
        """Return the binary gripper value at the given virtual time."""
        binary = self._gripper[-1]  # default: last
        for gi in reversed(self._gripper_indices):
            if float(self._ts[gi]) <= target_epoch:
                binary = self._gripper[gi]
                break
        return float(binary)

    def _handle_gripper_impedance(self) -> None:
        """Check gripper events and send commands only on transitions."""
        speed_val = max(0.1, float(self.get_parameter("speed").value))
        now = self.get_clock().now().nanoseconds * 1.0e-9
        elapsed = (now - (self._replay_start or now)) * speed_val
        target_epoch = float(self._ts[0]) + elapsed
        binary = self._gripper_binary_at(target_epoch)

        if abs(binary - float(self._last_gripper_binary)) < 0.01:
            return
        self._last_gripper_binary = binary
        self._send_gripper(binary)

    # ═══════════════════════════════════════════════════════════════
    #  JTC mode — discrete trajectory goals
    # ═══════════════════════════════════════════════════════════════

    def _tick_jtc(self) -> None:
        if self._goal_active:
            return
        if self._index >= self._num_steps:
            if bool(self.get_parameter("loop").value):
                self._index = 0
                return
            self.get_logger().info(f"Replay finished ({self._num_steps} steps)")
            self._timer.cancel()
            self.destroy_node()
            rclpy.shutdown()
            return

        joints = self._joints[self._index].tolist()
        gripper_binary = float(self._gripper[self._index])

        duration = float(self.get_parameter("trajectory_duration_sec").value)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = _make_trajectory_msg(
            list(self.get_parameter("joint_names").value), joints, duration,
        )
        self._goal_active = True
        send_future = self._traj_client.send_goal_async(goal)
        send_future.add_done_callback(self._traj_response_cb)
        self._send_gripper(gripper_binary)
        self._index += 1

    def _traj_response_cb(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self._goal_active = False
            self.get_logger().warn(f"Trajectory goal rejected: {exc}")
            return
        if not handle.accepted:
            self._goal_active = False
            self.get_logger().warn("Trajectory goal not accepted", throttle_duration_sec=1.0)
            return
        handle.get_result_async().add_done_callback(self._traj_result_cb)

    def _traj_result_cb(self, future) -> None:
        self._goal_active = False

    # ── gripper (shared) ──────────────────────────────────────────

    def _send_gripper(self, binary: float) -> None:
        speed = float(self.get_parameter("gripper_speed").value)
        if binary <= 0.5:
            if not self._gripper_grasp.server_is_ready():
                return
            goal = Grasp.Goal()
            goal.width = 0.0
            goal.speed = speed
            goal.force = float(self.get_parameter("gripper_force").value)
            goal.epsilon.inner = 0.05
            goal.epsilon.outer = 0.05
            self._gripper_grasp.send_goal_async(goal)
            self.get_logger().info("Gripper → close", throttle_duration_sec=0.5)
        else:
            if not self._gripper_move.server_is_ready():
                return
            goal = Move.Goal()
            goal.width = 0.08
            goal.speed = speed
            self._gripper_move.send_goal_async(goal)
            self.get_logger().info("Gripper → open", throttle_duration_sec=0.5)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
