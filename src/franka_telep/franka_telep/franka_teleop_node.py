from __future__ import annotations

import math

import rclpy
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp, Move
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_telep.franka_mapping import (
    FR3_JOINT_NAMES,
    FR3_LOWER_LIMITS,
    FR3_READY_JOINTS,
    FR3_UPPER_LIMITS,
    map_servo_offsets_to_joints,
)


class FrankaTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("franka_teleop")
        self.declare_parameter("servo_angles_topic", "/servo_angles")
        self.declare_parameter("servo_absolute_angles_topic", "/servo_absolute_angles")
        self.declare_parameter("use_absolute_servo_angles", True)
        self.declare_parameter("uarm_home_abs_angles_deg", [0.0] * 8)
        self.declare_parameter("latch_uarm_home_on_ready", True)
        self.declare_parameter("robot_action_topic", "/robot_action")
        self.declare_parameter("robot_state_topic", "/robot_state")
        self.declare_parameter("joint_state_topic", "/franka/joint_states")
        self.declare_parameter("home_ready_topic", "/franka_teleop/home_ready")
        self.declare_parameter("require_home_ready", True)
        self.declare_parameter("trajectory_action", "/fr3_arm_controller/follow_joint_trajectory")
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_grasp_action", "/franka_gripper/grasp")
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)
        self.declare_parameter("initial_joint_positions", FR3_READY_JOINTS)
        self.declare_parameter("use_current_as_initial", False)
        self.declare_parameter("arm_servo_indices", [0, 1, 2, 3, 4, 5, 6])
        self.declare_parameter("joint_signs", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("joint_scales", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("joint_lower_limits", FR3_LOWER_LIMITS)
        self.declare_parameter("joint_upper_limits", FR3_UPPER_LIMITS)
        self.declare_parameter("joint_limit_margin_rad", 0.05)
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("trajectory_duration_sec", 0.3)
        self.declare_parameter("trajectory_deadband_rad", 0.002)
        self.declare_parameter("target_filter_alpha", 0.35)
        self.declare_parameter("target_deadband_rad", 0.004)
        self.declare_parameter("cd_omega", 22.0)
        self.declare_parameter("servo_filter_alpha", 0.5)
        self.declare_parameter("max_teleop_deviation_rad", 0.0)
        self.declare_parameter("max_joint_step_rad", 0.0)
        self.declare_parameter("max_joint_velocity_rad_s", [0.45, 0.45, 0.45, 0.45, 0.65, 0.65, 0.65])
        self.declare_parameter("wait_for_action_servers", False)
        self.declare_parameter("enable_trajectory", True)
        self.declare_parameter("enable_gripper", True)
        self.declare_parameter("gripper_servo_index", 7)
        self.declare_parameter("gripper_threshold_deg", 15.0)
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_speed", 0.05)
        self.declare_parameter("gripper_deadband", 0.002)

        self._joint_names = [str(value) for value in self.get_parameter("joint_names").value]
        self._arm_servo_indices = [int(value) for value in self.get_parameter("arm_servo_indices").value]
        if len(self._joint_names) != len(self._arm_servo_indices):
            raise ValueError("joint_names and arm_servo_indices must have the same length")

        self._base_positions = [float(value) for value in self.get_parameter("initial_joint_positions").value]
        self._current_positions: list[float] | None = None
        self._target_positions = list(self._base_positions)
        self._command_positions: list[float] | None = None
        self._last_sent_positions: list[float] | None = None
        # Critically damped second-order filter state
        self._cd_positions: list[float] | None = None
        self._cd_velocities: list[float] | None = None
        self._cd_targets: list[float] | None = None
        self._cd_last_time = self.get_clock().now()
        self._servo_offsets_deg: list[float] | None = None
        self._servo_filtered_offsets_deg: list[float] | None = None
        self._latest_absolute_angles_deg: list[float] | None = None
        self._latched_uarm_home_abs_angles_deg: list[float] | None = None
        self._base_initialized_from_state = False
        self._home_ready = not bool(self.get_parameter("require_home_ready").value)
        self._last_gripper_width: float | None = None
        self._gripper_is_closed = False
        self._gripper_last_change = self.get_clock().now()
        self._last_command_time = self.get_clock().now()
        self._trajectory_goal_active = False

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
        )
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
        if bool(self.get_parameter("wait_for_action_servers").value):
            self._trajectory_client.wait_for_server(timeout_sec=3.0)
            if bool(self.get_parameter("enable_gripper").value):
                self._gripper_move_client.wait_for_server(timeout_sec=3.0)
                self._gripper_grasp_client.wait_for_server(timeout_sec=3.0)

        self._action_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("robot_action_topic").value),
            10,
        )
        self._state_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("robot_state_topic").value),
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("servo_angles_topic").value),
            self._relative_servo_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("servo_absolute_angles_topic").value),
            self._absolute_servo_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_callback,
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
        period = 1.0 / max(1.0, float(self.get_parameter("command_rate_hz").value))
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"Franka teleop listening on {self.get_parameter('servo_angles_topic').value}; "
            f"trajectory action={self.get_parameter('trajectory_action').value}"
        )

    def _relative_servo_callback(self, msg: Float64MultiArray) -> None:
        if bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        self._servo_offsets_deg = [float(value) for value in msg.data]
        self._apply_servo_filter()

    def _absolute_servo_callback(self, msg: Float64MultiArray) -> None:
        if not bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        current = [float(value) for value in msg.data]
        self._latest_absolute_angles_deg = current
        if self._home_ready:
            self._try_latch_uarm_home_zero()
        home = self._active_uarm_home_abs_angles()
        self._servo_offsets_deg = [
            current[index] - home[index] if index < len(current) and index < len(home) else 0.0
            for index in range(max(len(current), len(home)))
        ]
        self._apply_servo_filter()

    def _apply_servo_filter(self) -> None:
        """Light EMA low-pass on raw servo offsets to suppress high-frequency noise."""
        alpha = max(0.0, min(1.0, float(self.get_parameter("servo_filter_alpha").value)))
        if alpha >= 1.0 or self._servo_offsets_deg is None:
            self._servo_filtered_offsets_deg = (
                list(self._servo_offsets_deg) if self._servo_offsets_deg is not None else None
            )
            return
        if self._servo_filtered_offsets_deg is None or len(self._servo_filtered_offsets_deg) != len(self._servo_offsets_deg):
            self._servo_filtered_offsets_deg = list(self._servo_offsets_deg)
            return
        self._servo_filtered_offsets_deg = [
            prev + alpha * (raw - prev)
            for prev, raw in zip(self._servo_filtered_offsets_deg, self._servo_offsets_deg)
        ]

    def _home_ready_callback(self, msg: Bool) -> None:
        if msg.data and not self._home_ready:
            self.get_logger().info("Received Franka home ready; accepting teleop commands")
            self._last_command_time = self.get_clock().now()
            self._command_positions = list(self._current_positions or self._base_positions)
            self._last_sent_positions = list(self._current_positions or self._base_positions)
            self._init_cd_state(list(self._current_positions or self._base_positions))
            self._try_latch_uarm_home_zero()
        self._home_ready = bool(msg.data)

    def _active_uarm_home_abs_angles(self) -> list[float]:
        if self._latched_uarm_home_abs_angles_deg is not None:
            return self._latched_uarm_home_abs_angles_deg
        return [float(value) for value in self.get_parameter("uarm_home_abs_angles_deg").value]

    def _try_latch_uarm_home_zero(self) -> None:
        if not bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        if not bool(self.get_parameter("latch_uarm_home_on_ready").value):
            return
        if self._latched_uarm_home_abs_angles_deg is not None:
            return
        if self._latest_absolute_angles_deg is None:
            self.get_logger().warn("Waiting for UArm absolute angles before enabling teleop", throttle_duration_sec=2.0)
            return
        self._latched_uarm_home_abs_angles_deg = list(self._latest_absolute_angles_deg)
        self._servo_offsets_deg = [0.0] * len(self._latched_uarm_home_abs_angles_deg)
        self._servo_filtered_offsets_deg = [0.0] * len(self._latched_uarm_home_abs_angles_deg)
        self._last_gripper_width = None
        self.get_logger().info(
            "Latched current UArm absolute angles as teleop zero: "
            f"{[round(value, 2) for value in self._latched_uarm_home_abs_angles_deg]}"
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        positions = self._positions_for_configured_joints(msg)
        if positions is None:
            return
        self._current_positions = positions
        self._state_pub.publish(Float64MultiArray(data=positions))
        if self._command_positions is None:
            self._command_positions = list(positions)
        if bool(self.get_parameter("use_current_as_initial").value) and not self._base_initialized_from_state:
            self._base_positions = list(positions)
            self._target_positions = list(positions)
            self._init_cd_state(list(positions))
            self._base_initialized_from_state = True
            self.get_logger().info("Using current joint state as teleop zero pose.")

    def _positions_for_configured_joints(self, msg: JointState) -> list[float] | None:
        by_name = {name: index for index, name in enumerate(msg.name)}
        if not all(name in by_name for name in self._joint_names):
            return None
        return [float(msg.position[by_name[name]]) for name in self._joint_names]

    def _init_cd_state(self, positions: list[float]) -> None:
        self._cd_positions = list(positions)
        self._cd_velocities = [0.0] * len(positions)
        self._cd_targets = list(positions)
        self._cd_last_time = self.get_clock().now()

    def _critically_damped_step(self, desired: list[float]) -> list[float]:
        """Critically damped second-order filter for smooth continuous tracking.

        Unlike quintic polynomials (designed for point-to-point motion with
        zero velocity at endpoints), this filter is ideal for teleoperation:
        it tracks a moving target smoothly without ever forcing velocity to
        zero.  Dynamics:  p'' + 2ω p' + ω² p = ω² target  (ζ = 1).
        """
        now = self.get_clock().now()
        omega = max(1.0, float(self.get_parameter("cd_omega").value))
        deadband = max(0.0, float(self.get_parameter("target_deadband_rad").value))

        if self._cd_targets is None:
            self._init_cd_state(
                list(self._current_positions or self._base_positions)
            )
            return list(desired)

        # Update individual joint targets that moved beyond deadband
        for i, (d, t) in enumerate(zip(desired, self._cd_targets)):
            if abs(d - t) > deadband:
                self._cd_targets[i] = d

        # Integrate critically damped dynamics
        dt = (now - self._cd_last_time).nanoseconds * 1e-9
        dt = max(dt, 1e-6)
        self._cd_last_time = now
        omega2 = omega * omega
        two_omega = 2.0 * omega

        result = []
        for i, target in enumerate(self._cd_targets):
            p = self._cd_positions[i]
            v = self._cd_velocities[i]
            # Critically damped: a = ω²·(target - p) - 2ω·v
            a = omega2 * (target - p) - two_omega * v
            v += a * dt
            p += v * dt
            self._cd_positions[i] = p
            self._cd_velocities[i] = v
            result.append(p)

        return result

    def _tick(self) -> None:
        if not self._home_ready:
            self.get_logger().warn("Waiting for Franka home initializer before teleop", throttle_duration_sec=2.0)
            return
        if (
            bool(self.get_parameter("use_absolute_servo_angles").value)
            and bool(self.get_parameter("latch_uarm_home_on_ready").value)
            and self._latched_uarm_home_abs_angles_deg is None
        ):
            self._try_latch_uarm_home_zero()
            return
        if self._servo_filtered_offsets_deg is None:
            return

        lower_limits = [float(value) for value in self.get_parameter("joint_lower_limits").value]
        upper_limits = [float(value) for value in self.get_parameter("joint_upper_limits").value]
        limit_margin = float(self.get_parameter("joint_limit_margin_rad").value)
        desired = map_servo_offsets_to_joints(
            self._servo_filtered_offsets_deg,
            base_positions_rad=self._base_positions,
            arm_servo_indices=self._arm_servo_indices,
            signs=[float(value) for value in self.get_parameter("joint_signs").value],
            scales=[float(value) for value in self.get_parameter("joint_scales").value],
            lower_limits=[value + limit_margin for value in lower_limits],
            upper_limits=[value - limit_margin for value in upper_limits],
        )
        desired = self._critically_damped_step(desired)
        target = self._limited_target(desired)
        target = self._clamp_step(target)
        self._target_positions = target
        self._action_pub.publish(Float64MultiArray(data=target + [self._current_gripper_width_or_default()]))

        if bool(self.get_parameter("enable_trajectory").value):
            self._send_trajectory_if_needed(target)
        if bool(self.get_parameter("enable_gripper").value):
            self._send_gripper_if_needed()

    def _limited_target(self, desired: list[float]) -> list[float]:
        now = self.get_clock().now()
        dt = max((now - self._last_command_time).nanoseconds * 1e-9, 1.0 / max(1.0, float(self.get_parameter("command_rate_hz").value)))
        self._last_command_time = now
        if self._command_positions is None:
            self._command_positions = list(self._current_positions or self._base_positions)
        max_vel = [float(value) for value in self.get_parameter("max_joint_velocity_rad_s").value]
        limited = []
        for index, goal in enumerate(desired):
            current = self._command_positions[index]
            step_limit = max_vel[index] * dt if index < len(max_vel) else max_vel[-1] * dt
            delta = goal - current
            delta = max(-step_limit, min(step_limit, delta))
            limited.append(current + delta)
        self._command_positions = limited
        return limited

    def _clamp_step(self, target: list[float]) -> list[float]:
        """Clamp each joint's movement relative to the last sent position."""
        limit = float(self.get_parameter("max_joint_step_rad").value)
        if limit <= 0.0:
            return list(target)
        previous = self._last_sent_positions or self._current_positions or self._base_positions
        clamped = []
        for i, (goal, prev) in enumerate(zip(target, previous)):
            delta = goal - prev
            delta = max(-limit, min(limit, delta))
            clamped.append(prev + delta)
        return clamped

    def _send_trajectory_if_needed(self, target: list[float]) -> None:
        if self._trajectory_goal_active:
            return
        if not self._trajectory_client.server_is_ready():
            self.get_logger().warn("Trajectory action server is not available", throttle_duration_sec=2.0)
            return
        if self._last_sent_positions is not None:
            max_delta = max(abs(goal - old) for goal, old in zip(target, self._last_sent_positions))
            if max_delta < float(self.get_parameter("trajectory_deadband_rad").value):
                return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._make_trajectory(target)
        send_future = self._trajectory_client.send_goal_async(goal)
        self._trajectory_goal_active = True
        send_future.add_done_callback(self._trajectory_goal_response_cb)
        self._last_sent_positions = list(target)

    def _trajectory_goal_response_cb(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._trajectory_goal_active = False
            self.get_logger().warn(f"Trajectory goal send failed: {exc}")
            return
        if not goal_handle.accepted:
            self._trajectory_goal_active = False
            self.get_logger().warn("Trajectory goal was rejected")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._trajectory_goal_result_cb)

    def _trajectory_goal_result_cb(self, future) -> None:
        self._trajectory_goal_active = False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Trajectory goal result failed: {exc}")
            return
        status = getattr(result, "status", None)
        if status not in (None, 4):
            self.get_logger().warn(f"Trajectory goal finished with status={status}", throttle_duration_sec=2.0)

    def _make_trajectory(self, target: list[float]) -> JointTrajectory:
        start = self._command_positions or self._current_positions or self._base_positions
        duration_sec = float(self.get_parameter("trajectory_duration_sec").value)
        sample_dt = 0.001
        num_points = max(int(duration_sec / sample_dt) + 1, 2)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self._joint_names)

        for i in range(num_points):
            elapsed = i * sample_dt
            t_norm = max(0.0, min(1.0, elapsed / max(duration_sec, 1e-6)))
            s = 10.0 * t_norm**3 - 15.0 * t_norm**4 + 6.0 * t_norm**5
            ds_dt = (30.0 * t_norm**2 - 60.0 * t_norm**3 + 30.0 * t_norm**4) / max(duration_sec, 1e-6)

            sec = int(elapsed)
            nsec = int((elapsed - sec) * 1_000_000_000)

            point = JointTrajectoryPoint()
            point.positions = [float(p0 + s * (tg - p0)) for p0, tg in zip(start, target)]
            point.velocities = [float(ds_dt * (tg - p0)) for p0, tg in zip(start, target)]
            point.accelerations = [0.0] * len(target)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nsec
            trajectory.points.append(point)

        return trajectory

    def _send_gripper_if_needed(self) -> None:
        gripper_width = self._current_gripper_width_or_nan()
        if math.isnan(gripper_width):
            return
        if self._last_gripper_width is not None:
            if abs(gripper_width - self._last_gripper_width) < float(self.get_parameter("gripper_deadband").value):
                return
        speed = float(self.get_parameter("gripper_speed").value)
        min_width = float(self.get_parameter("gripper_min_width").value)

        if gripper_width <= min_width + 0.001:
            # Closing → use grasp (force-controlled, stops on contact)
            if not self._gripper_grasp_client.server_is_ready():
                self.get_logger().warn("Gripper grasp action not available", throttle_duration_sec=2.0)
                return
            goal = Grasp.Goal()
            goal.width = gripper_width
            goal.speed = speed
            goal.force = 60.0
            goal.epsilon.inner = 0.05
            goal.epsilon.outer = 0.05
            self._gripper_grasp_client.send_goal_async(goal)
        else:
            # Opening → use move (no object, just go to width)
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
        offsets = self._servo_filtered_offsets_deg
        if offsets is None or index < 0 or index >= len(offsets):
            return float("nan")
        close_threshold = float(self.get_parameter("gripper_threshold_deg").value)
        open_threshold = max(0.0, close_threshold - 5.0)
        min_width = float(self.get_parameter("gripper_min_width").value)
        max_width = float(self.get_parameter("gripper_max_width").value)
        offset = abs(offsets[index])

        now = self.get_clock().now()
        debounce_sec = 0.5
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
    node = FrankaTeleopNode()
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
