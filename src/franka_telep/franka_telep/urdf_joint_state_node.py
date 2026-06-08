from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from franka_telep.franka_mapping import (
    FR3_JOINT_NAMES,
    FR3_LOWER_LIMITS,
    FR3_READY_JOINTS,
    FR3_UPPER_LIMITS,
    map_servo_offsets_to_joints,
)


class UrdfJointStateNode(Node):
    """Publish preview joint states from UArm servo offsets."""

    def __init__(self) -> None:
        super().__init__("urdf_joint_state")
        self.declare_parameter("servo_angles_topic", "/servo_angles")
        self.declare_parameter("servo_absolute_angles_topic", "/servo_absolute_angles")
        self.declare_parameter("use_absolute_servo_angles", False)
        self.declare_parameter("uarm_home_abs_angles_deg", [0.0] * 8)
        self.declare_parameter("joint_state_topic", "/teleop_preview/joint_states")
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)
        self.declare_parameter("initial_joint_positions", FR3_READY_JOINTS)
        self.declare_parameter("arm_servo_indices", [0, 1, 2, 3, 4, 5, 6])
        self.declare_parameter("joint_signs", [1.0] * 7)
        self.declare_parameter("joint_scales", [1.0] * 7)
        self.declare_parameter("joint_lower_limits", FR3_LOWER_LIMITS)
        self.declare_parameter("joint_upper_limits", FR3_UPPER_LIMITS)
        self.declare_parameter("joint_limit_margin_rad", 0.05)
        self.declare_parameter("gripper_servo_index", 7)
        self.declare_parameter("gripper_threshold_deg", 15.0)
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)

        self._joint_names = [str(value) for value in self.get_parameter("joint_names").value]
        self._arm_servo_indices = [int(value) for value in self.get_parameter("arm_servo_indices").value]
        if len(self._joint_names) != len(self._arm_servo_indices):
            raise ValueError("joint_names and arm_servo_indices must have the same length")

        self._publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
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
        self.get_logger().info(
            f"URDF preview publishing {self.get_parameter('joint_state_topic').value} "
            f"from {self.get_parameter('servo_angles_topic').value}"
        )

    def _relative_servo_callback(self, msg: Float64MultiArray) -> None:
        if bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        self._publish_from_offsets([float(value) for value in msg.data])

    def _absolute_servo_callback(self, msg: Float64MultiArray) -> None:
        if not bool(self.get_parameter("use_absolute_servo_angles").value):
            return
        current = [float(value) for value in msg.data]
        home = [float(value) for value in self.get_parameter("uarm_home_abs_angles_deg").value]
        offsets = [
            current[index] - home[index] if index < len(current) and index < len(home) else 0.0
            for index in range(max(len(current), len(home)))
        ]
        self._publish_from_offsets(offsets)

    def _publish_from_offsets(self, offsets: list[float]) -> None:
        lower_limits = [float(value) for value in self.get_parameter("joint_lower_limits").value]
        upper_limits = [float(value) for value in self.get_parameter("joint_upper_limits").value]
        limit_margin = float(self.get_parameter("joint_limit_margin_rad").value)
        positions = map_servo_offsets_to_joints(
            offsets,
            base_positions_rad=[
                float(value) for value in self.get_parameter("initial_joint_positions").value
            ],
            arm_servo_indices=self._arm_servo_indices,
            signs=[float(value) for value in self.get_parameter("joint_signs").value],
            scales=[float(value) for value in self.get_parameter("joint_scales").value],
            lower_limits=[value + limit_margin for value in lower_limits],
            upper_limits=[value - limit_margin for value in upper_limits],
        )

        width = self._gripper_width(offsets)
        state = JointState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.name = self._joint_names + ["fr3_finger_joint1"]
        state.position = positions + [width * 0.5]
        self._publisher.publish(state)

    def _gripper_width(self, offsets: list[float]) -> float:
        index = int(self.get_parameter("gripper_servo_index").value)
        if index < 0 or index >= len(offsets) or math.isnan(offsets[index]):
            return float(self.get_parameter("gripper_max_width").value)
        threshold = float(self.get_parameter("gripper_threshold_deg").value)
        min_width = float(self.get_parameter("gripper_min_width").value)
        max_width = float(self.get_parameter("gripper_max_width").value)
        # Binary gripper: close if offset exceeds threshold, open otherwise
        return min_width if abs(offsets[index]) > threshold else max_width


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = UrdfJointStateNode()
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
