"""Publish leader JointState from uArm servo absolute angles.

Subscribes to the Zhonglin servo reader's absolute-angle topic,
applies EMA smoothing, maps servo offsets to FR3 joint positions
(using the same franka_mapping as the existing franka_teleop node),
and publishes sensor_msgs/JointState on a topic that the
TeleopFollowerController from franka_ros2_teleop can subscribe to.

The follower controller runs a joint-impedance (PD) law at 1000 Hz
and expects *exactly 7* position values in the JointState message.
"""
from __future__ import annotations

import math

from franka_msgs.action import Grasp, Move
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray

from franka_telep.franka_mapping import (
    FR3_JOINT_NAMES,
    FR3_LOWER_LIMITS,
    FR3_READY_JOINTS,
    FR3_UPPER_LIMITS,
    map_servo_offsets_to_joints,
)


class UarmLeaderPublisher(Node):
    def __init__(self) -> None:
        super().__init__("uarm_leader_publisher")

# Subscription topics.
        self.declare_parameter("servo_absolute_angles_topic", "/servo_absolute_angles")
        self.declare_parameter("home_ready_topic", "/franka_teleop/home_ready")
        self.declare_parameter("require_home_ready", True)

        # uArm home (zero) configuration.
        self.declare_parameter("use_absolute_servo_angles", True)
        self.declare_parameter("uarm_home_abs_angles_deg", [0.0] * 8)
        self.declare_parameter("latch_uarm_home_on_ready", True)

        # Servo to joint mapping parameters.
        self.declare_parameter("arm_servo_indices", [0, 1, 2, 3, 4, 5, 6])
        self.declare_parameter("joint_signs", [1.0] * 7)
        self.declare_parameter("joint_scales", [1.0] * 7)
        self.declare_parameter("initial_joint_positions", FR3_READY_JOINTS)
        self.declare_parameter("joint_lower_limits", FR3_LOWER_LIMITS)
        self.declare_parameter("joint_upper_limits", FR3_UPPER_LIMITS)
        self.declare_parameter("joint_limit_margin_rad", 0.05)

        # Publishing.
        self.declare_parameter("publish_topic", "/uarm_leader/joint_states")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("servo_filter_alpha", 0.5)
        self.declare_parameter("servo_input_timeout_sec", 0.15)
        self.declare_parameter("max_joint_velocity_rad_s", [0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5])

        # Franka gripper control, migrated from franka_teleop_node.
        self.declare_parameter("enable_gripper", True)
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_grasp_action", "/franka_gripper/grasp")
        self.declare_parameter("gripper_servo_index", 7)
        self.declare_parameter("gripper_threshold_deg", 15.0)
        self.declare_parameter("gripper_hysteresis_deg", 5.0)
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_speed", 0.1)
        self.declare_parameter("gripper_force", 60.0)
        self.declare_parameter("gripper_epsilon_inner", 0.05)
        self.declare_parameter("gripper_epsilon_outer", 0.05)
        self.declare_parameter("gripper_deadband", 0.002)
        self.declare_parameter("gripper_debounce_sec", 0.5)
        self.declare_parameter("gripper_command_topic", "/uarm_leader/gripper_command")

        self._arm_servo_indices = [int(v) for v in self.get_parameter("arm_servo_indices").value]
        self._joint_names = list(FR3_JOINT_NAMES)

        # State.
        self._latest_absolute_angles_deg: list[float] | None = None
        self._last_servo_msg_time = self.get_clock().now()
        self._latched_uarm_home: list[float] | None = None
        self._servo_filtered_offsets: list[float] | None = None
        self._limited_positions: list[float] | None = None
        self._last_limit_time = self.get_clock().now()
        self._home_ready = not bool(self.get_parameter("require_home_ready").value)
        self._last_gripper_width: float | None = None
        self._gripper_is_closed = False
        self._gripper_last_change = self.get_clock().now()

        self._gripper_move_client = ActionClient(
            self,
            Move,
            str(self.get_parameter("gripper_move_action").value),
        )
        self._gripper_grasp_client = ActionClient(
            self,
            Grasp,
            str(self.get_parameter("gripper_grasp_action").value),
        )

        # -- subscriptions --
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("servo_absolute_angles_topic").value),
            self._absolute_servo_callback,
            10,
        )
        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool,
            str(self.get_parameter("home_ready_topic").value),
            self._home_ready_callback,
            ready_qos,
        )

        self._pub = self.create_publisher(
            JointState,
            str(self.get_parameter("publish_topic").value),
            10,
        )
        self._gripper_cmd_pub = self.create_publisher(
            Float64,
            str(self.get_parameter("gripper_command_topic").value),
            10,
        )

        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(period, self._publish_tick)
        self.get_logger().info(
            f"UArm leader publisher -> {self.get_parameter('publish_topic').value}"
        )

    # ------------------------------------------------------------------
    #  callbacks
    # ------------------------------------------------------------------
    def _absolute_servo_callback(self, msg: Float64MultiArray) -> None:
        if not bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        current = [float(v) for v in msg.data]
        self._latest_absolute_angles_deg = current
        self._last_servo_msg_time = self.get_clock().now()
        if self._home_ready:
            self._try_latch_uarm_home()
        self._update_offsets(current)

    def _home_ready_callback(self, msg: Bool) -> None:
        if msg.data and not self._home_ready:
            self.get_logger().info("Franka home ready; accepting uArm leader input")
            self._try_latch_uarm_home()
        self._home_ready = bool(msg.data)

    # ------------------------------------------------------------------
    #  uArm home latching  (same logic as franka_teleop_node)
    # ------------------------------------------------------------------
    def _active_uarm_home(self) -> list[float]:
        if self._latched_uarm_home is not None:
            return self._latched_uarm_home
        return [float(v) for v in self.get_parameter("uarm_home_abs_angles_deg").value]

    def _try_latch_uarm_home(self) -> None:
        if not bool(self.get_parameter("latch_uarm_home_on_ready").value):
            return
        if self._latched_uarm_home is not None:
            return
        if self._latest_absolute_angles_deg is None:
            self.get_logger().warn(
                "Waiting for UArm absolute angles before latching home",
                throttle_duration_sec=2.0,
            )
            return
        self._latched_uarm_home = list(self._latest_absolute_angles_deg)
        self._servo_filtered_offsets = [0.0] * len(self._latched_uarm_home)
        self._limited_positions = [
            float(v) for v in self.get_parameter("initial_joint_positions").value
        ]
        self._last_limit_time = self.get_clock().now()
        self.get_logger().info(
            "Latched UArm absolute angles as teleop zero: "
            f"{[round(v, 2) for v in self._latched_uarm_home]}"
        )

    # ------------------------------------------------------------------
    #  servo to FR3 joint mapping  (reuses existing franka_mapping)
    # ------------------------------------------------------------------
    def _update_offsets(self, current_abs: list[float]) -> None:
        home = self._active_uarm_home()
        raw_offsets = [
            current_abs[i] - home[i] if i < len(current_abs) and i < len(home) else 0.0
            for i in range(max(len(current_abs), len(home)))
        ]
        alpha = max(0.0, min(1.0, float(self.get_parameter("servo_filter_alpha").value)))
        if alpha >= 1.0 or self._servo_filtered_offsets is None:
            self._servo_filtered_offsets = list(raw_offsets)
            return
        if len(self._servo_filtered_offsets) != len(raw_offsets):
            self._servo_filtered_offsets = list(raw_offsets)
            return
        self._servo_filtered_offsets = [
            prev + alpha * (raw - prev)
            for prev, raw in zip(self._servo_filtered_offsets, raw_offsets)
        ]

    def _map_to_joints(self) -> list[float]:
        lower = [float(v) for v in self.get_parameter("joint_lower_limits").value]
        upper = [float(v) for v in self.get_parameter("joint_upper_limits").value]
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        return map_servo_offsets_to_joints(
            self._servo_filtered_offsets,
            base_positions_rad=[
                float(v) for v in self.get_parameter("initial_joint_positions").value
            ],
            arm_servo_indices=self._arm_servo_indices,
            signs=[float(v) for v in self.get_parameter("joint_signs").value],
            scales=[float(v) for v in self.get_parameter("joint_scales").value],
            lower_limits=[lo + margin for lo in lower],
            upper_limits=[hi - margin for hi in upper],
        )

    # ------------------------------------------------------------------
    #  publish tick
    # ------------------------------------------------------------------
    def _publish_tick(self) -> None:
        if not self._home_ready:
            self.get_logger().warn(
                "Waiting for Franka home before publishing leader state",
                throttle_duration_sec=2.0,
            )
            return
        if (
            bool(self.get_parameter("latch_uarm_home_on_ready").value)
            and self._latched_uarm_home is None
        ):
            self.get_logger().warn(
                "Waiting for first UArm absolute angles to latch home",
                throttle_duration_sec=2.0,
            )
            return
        if self._servo_filtered_offsets is None:
            return
        if self._servo_input_timed_out():
            self.get_logger().warn(
                "UArm input timed out; stopping leader JointState publication",
                throttle_duration_sec=1.0,
            )
            return

        positions = self._limited_target(self._map_to_joints())

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"fr3_joint{i}" for i in range(1, 8)]
        msg.position = [float(p) for p in positions]
        self._pub.publish(msg)

        if bool(self.get_parameter("enable_gripper").value):
            self._gripper_cmd_pub.publish(Float64(data=self._current_gripper_width_or_default()))
            self._send_gripper_if_needed()

    def _servo_input_timed_out(self) -> bool:
        timeout_sec = float(self.get_parameter("servo_input_timeout_sec").value)
        if timeout_sec <= 0.0:
            return False
        age_sec = (self.get_clock().now() - self._last_servo_msg_time).nanoseconds * 1e-9
        return age_sec > timeout_sec

    def _limited_target(self, desired: list[float]) -> list[float]:
        max_vel = [float(v) for v in self.get_parameter("max_joint_velocity_rad_s").value]
        if not max_vel or max(max_vel) <= 0.0:
            self._limited_positions = list(desired)
            self._last_limit_time = self.get_clock().now()
            return list(desired)

        now = self.get_clock().now()
        dt = (now - self._last_limit_time).nanoseconds * 1e-9
        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        dt = max(min(dt, 0.2), period)
        self._last_limit_time = now

        if self._limited_positions is None or len(self._limited_positions) != len(desired):
            self._limited_positions = list(desired)
            return list(desired)

        limited = []
        for index, goal in enumerate(desired):
            current = self._limited_positions[index]
            velocity = max_vel[index] if index < len(max_vel) else max_vel[-1]
            step = max(0.0, velocity) * dt
            delta = max(-step, min(step, goal - current))
            limited.append(current + delta)
        self._limited_positions = limited
        return limited

    def _send_gripper_if_needed(self) -> None:
        gripper_width = self._current_gripper_width_or_nan()
        if math.isnan(gripper_width):
            return
        if self._last_gripper_width is not None:
            if abs(gripper_width - self._last_gripper_width) < float(self.get_parameter("gripper_deadband").value):
                return

        min_width = float(self.get_parameter("gripper_min_width").value)
        speed = float(self.get_parameter("gripper_speed").value)
        if gripper_width <= min_width + 0.001:
            if not self._gripper_grasp_client.server_is_ready():
                self.get_logger().warn("Gripper grasp action not available", throttle_duration_sec=2.0)
                return
            goal = Grasp.Goal()
            goal.width = gripper_width
            goal.speed = speed
            goal.force = float(self.get_parameter("gripper_force").value)
            goal.epsilon.inner = float(self.get_parameter("gripper_epsilon_inner").value)
            goal.epsilon.outer = float(self.get_parameter("gripper_epsilon_outer").value)
            self._gripper_grasp_client.send_goal_async(goal)
        else:
            if not self._gripper_move_client.server_is_ready():
                self.get_logger().warn("Gripper move action not available", throttle_duration_sec=2.0)
                return
            goal = Move.Goal()
            goal.width = gripper_width
            goal.speed = speed
            self._gripper_move_client.send_goal_async(goal)

        self._last_gripper_width = gripper_width

    def _current_gripper_width_or_nan(self) -> float:
        index = int(self.get_parameter("gripper_servo_index").value)
        offsets = self._servo_filtered_offsets
        if offsets is None or index < 0 or index >= len(offsets):
            return float("nan")

        close_threshold = float(self.get_parameter("gripper_threshold_deg").value)
        hysteresis = max(0.0, float(self.get_parameter("gripper_hysteresis_deg").value))
        open_threshold = max(0.0, close_threshold - hysteresis)
        min_width = float(self.get_parameter("gripper_min_width").value)
        max_width = float(self.get_parameter("gripper_max_width").value)
        offset = abs(offsets[index])

        now = self.get_clock().now()
        debounce_sec = max(0.0, float(self.get_parameter("gripper_debounce_sec").value))
        if (now - self._gripper_last_change).nanoseconds * 1e-9 < debounce_sec:
            return min_width if self._gripper_is_closed else max_width

        if offset > close_threshold:
            if not self._gripper_is_closed:
                self._gripper_is_closed = True
                self._gripper_last_change = now
        elif offset < open_threshold:
            if self._gripper_is_closed:
                self._gripper_is_closed = False
                self._gripper_last_change = now

        return min_width if self._gripper_is_closed else max_width

    def _current_gripper_width_or_default(self) -> float:
        width = self._current_gripper_width_or_nan()
        if math.isnan(width):
            return float(self.get_parameter("gripper_max_width").value)
        return width


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = UarmLeaderPublisher()
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
