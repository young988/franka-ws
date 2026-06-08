from __future__ import annotations

import math

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_telep.franka_mapping import FR3_JOINT_NAMES, FR3_READY_JOINTS


class FrankaHomeInitializer(Node):
    def __init__(self) -> None:
        super().__init__("franka_home_initializer")
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)
        self.declare_parameter("home_joint_positions", FR3_READY_JOINTS)
        self.declare_parameter("joint_state_topic", "/franka/joint_states")
        self.declare_parameter("trajectory_action", "/fr3_arm_controller/follow_joint_trajectory")
        self.declare_parameter("ready_topic", "/franka_teleop/home_ready")
        self.declare_parameter("trajectory_duration_sec", 6.0)
        self.declare_parameter("goal_tolerance_rad", 0.03)
        self.declare_parameter("ready_delay_sec", 0.5)
        self.declare_parameter("hold_ready_publish_hz", 2.0)

        self._joint_names = [str(value) for value in self.get_parameter("joint_names").value]
        self._home = [float(value) for value in self.get_parameter("home_joint_positions").value]
        if len(self._joint_names) != len(self._home):
            raise ValueError("joint_names and home_joint_positions must have the same length")

        self._current: list[float] | None = None
        self._sent = False
        self._sent_time = None
        self._ready = False
        self._last_log = self.get_clock().now()
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._ready_pub = self.create_publisher(Bool, str(self.get_parameter("ready_topic").value), qos)
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_cb,
            10,
        )
        self._timer = self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"Waiting to send Franka home goal on {self.get_parameter('trajectory_action').value}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        by_name = {name: index for index, name in enumerate(msg.name)}
        if not all(name in by_name for name in self._joint_names):
            return
        self._current = [float(msg.position[by_name[name]]) for name in self._joint_names]

    def _tick(self) -> None:
        if self._ready:
            self._ready_pub.publish(Bool(data=True))
            return
        if self._current is None:
            self._throttled_info("Waiting for Franka joint state")
            return
        if not self._traj_client.server_is_ready():
            self._throttled_info("Waiting for trajectory action server")
            return
        if not self._sent:
            self._send_home_goal()
            self._sent = True
            self._sent_time = self.get_clock().now()
            return
        if not self._minimum_home_time_elapsed():
            return
        if self._at_home():
            self._ready = True
            self._ready_pub.publish(Bool(data=True))
            self.get_logger().info("Franka reached teleop home; teleoperation is enabled")

    def _send_home_goal(self) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(self._home)
        point.velocities = [0.0] * len(self._home)
        point.accelerations = [0.0] * len(self._home)
        duration_sec = float(self.get_parameter("trajectory_duration_sec").value)
        sec = int(duration_sec)
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = int((duration_sec - sec) * 1_000_000_000)
        goal.trajectory.points.append(point)
        self.get_logger().info(f"Sending Franka home goal: {[round(v, 4) for v in self._home]}")
        self._traj_client.send_goal_async(goal)

    def _minimum_home_time_elapsed(self) -> bool:
        if self._sent_time is None:
            return False
        wait_sec = (
            float(self.get_parameter("trajectory_duration_sec").value)
            + float(self.get_parameter("ready_delay_sec").value)
        )
        return self.get_clock().now() - self._sent_time >= Duration(seconds=wait_sec)

    def _at_home(self) -> bool:
        if self._current is None:
            return False
        tolerance = float(self.get_parameter("goal_tolerance_rad").value)
        max_error = max(abs(current - target) for current, target in zip(self._current, self._home))
        if max_error > tolerance:
            now = self.get_clock().now()
            if now - self._last_log > Duration(seconds=1.0):
                self._last_log = now
                self.get_logger().info(f"Waiting for home convergence: max_error={max_error:.4f} rad")
            return False
        return True

    def _throttled_info(self, message: str) -> None:
        now = self.get_clock().now()
        if now - self._last_log > Duration(seconds=2.0):
            self._last_log = now
            self.get_logger().info(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FrankaHomeInitializer()
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
