"""Hardcoded base-frame delta action test for a real FR3."""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Move
from franka_msgs.msg import FrankaRobotState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

from franka_policy_runtime.reference import (
    apply_tcp_delta,
    gripper_width_from_binary_action,
    make_joint_trajectory,
)
from franka_policy_runtime.runtime_config import FR3_JOINT_NAMES
from franka_policy_runtime.tcp_pose import (
    compose_pose_xyzw,
    pose_msg_to_arrays,
    transform_msg_to_arrays,
)


@dataclass(frozen=True)
class ActionTestStep:
    name: str
    action: np.ndarray

    @property
    def is_gripper_step(self) -> bool:
        return not np.any(np.abs(self.action[:6]) > 0.0)


def build_action_test_sequence(
    *,
    translation_step_m: float = 0.01,
    rotation_step_rad: float = 0.05,
    gripper_open_value: float = 1.0,
    gripper_close_value: float = -1.0,
) -> list[ActionTestStep]:
    sequence: list[ActionTestStep] = []
    for name, index, value in [
        ("+dx", 0, translation_step_m),
        ("-dx", 0, -translation_step_m),
        ("+dy", 1, translation_step_m),
        ("-dy", 1, -translation_step_m),
        ("+dz", 2, translation_step_m),
        ("-dz", 2, -translation_step_m),
        ("+rx", 3, rotation_step_rad),
        ("-rx", 3, -rotation_step_rad),
        ("+ry", 4, rotation_step_rad),
        ("-ry", 4, -rotation_step_rad),
        ("+rz", 5, rotation_step_rad),
        ("-rz", 5, -rotation_step_rad),
    ]:
        action = np.zeros(7, dtype=float)
        action[index] = float(value)
        sequence.append(ActionTestStep(name=name, action=action))

    open_action = np.zeros(7, dtype=float)
    open_action[6] = float(gripper_open_value)
    sequence.append(ActionTestStep(name="gripper_open", action=open_action))

    close_action = np.zeros(7, dtype=float)
    close_action[6] = float(gripper_close_value)
    sequence.append(ActionTestStep(name="gripper_close", action=close_action))
    return sequence


def pose_errors(
    target_position: np.ndarray,
    target_quat_xyzw: np.ndarray,
    actual_position: np.ndarray,
    actual_quat_xyzw: np.ndarray,
) -> tuple[float, float]:
    target_pos = np.asarray(target_position, dtype=float)
    actual_pos = np.asarray(actual_position, dtype=float)
    pos_error = float(np.linalg.norm(actual_pos - target_pos))

    target_quat = np.asarray(target_quat_xyzw, dtype=float)
    actual_quat = np.asarray(actual_quat_xyzw, dtype=float)
    target_quat = target_quat / np.linalg.norm(target_quat)
    actual_quat = actual_quat / np.linalg.norm(actual_quat)
    dot = min(1.0, max(-1.0, abs(float(np.dot(target_quat, actual_quat)))))
    rot_error = float(2.0 * math.acos(dot))
    return pos_error, rot_error


def classify_step_result(
    step: ActionTestStep,
    *,
    pos_error_m: float | None,
    rot_error_rad: float | None,
    tolerance_pos_m: float = 0.01,
    tolerance_rot_rad: float = 0.05,
) -> str:
    if step.is_gripper_step:
        return "gripper_only"
    if pos_error_m is None or rot_error_rad is None:
        return "fail"
    if pos_error_m <= float(tolerance_pos_m) and rot_error_rad <= float(tolerance_rot_rad):
        return "pass"
    return "fail"


class ActionTestNode(Node):
    """Execute a fixed 7D action sequence against the FR3 control stack."""

    def __init__(self) -> None:
        super().__init__("action_test")
        self.declare_parameter("command_frame", "fr3_link0")
        self.declare_parameter("tcp_frame", "fr3_hand_tcp")
        self.declare_parameter("tcp_pose_source", "franka_state")
        self.declare_parameter("franka_state_topic", "/franka_robot_state_broadcaster/robot_state")
        self.declare_parameter("franka_state_ee_frame", "fr3_hand_tcp")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("move_group_name", "fr3_arm")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("trajectory_duration_sec", 0.5)
        self.declare_parameter("step_interval_sec", 1.0)
        self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("translation_step_m", 0.01)
        self.declare_parameter("rotation_step_rad", 0.05)
        self.declare_parameter("tolerance_pos_m", 0.01)
        self.declare_parameter("tolerance_rot_rad", 0.05)
        self.declare_parameter("csv_output_path", "/tmp/franka_delta_test_results.csv")
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_speed", 0.05)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._latest_joints: np.ndarray | None = None
        self._latest_franka_state_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._started = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._callback_group = ReentrantCallbackGroup()
        self._ik_client = self.create_client(
            GetPositionIK,
            str(self.get_parameter("ik_service").value),
            callback_group=self._callback_group,
        )
        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
            callback_group=self._callback_group,
        )
        self._gripper_client = ActionClient(
            self,
            Move,
            str(self.get_parameter("gripper_move_action").value),
            callback_group=self._callback_group,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_cb,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            FrankaRobotState,
            str(self.get_parameter("franka_state_topic").value),
            self._franka_state_cb,
            10,
            callback_group=self._callback_group,
        )
        self.create_timer(0.5, self._start_once, callback_group=self._callback_group)

    def _joint_state_cb(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position))
        if all(name in by_name for name in self._joint_names):
            self._latest_joints = np.array([by_name[name] for name in self._joint_names], dtype=float)

    def _franka_state_cb(self, msg: FrankaRobotState) -> None:
        self._latest_franka_state_pose = pose_msg_to_arrays(msg.o_t_ee.pose)

    def _start_once(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run_sequence, daemon=True).start()

    def _run_sequence(self) -> None:
        sequence = build_action_test_sequence(
            translation_step_m=float(self.get_parameter("translation_step_m").value),
            rotation_step_rad=float(self.get_parameter("rotation_step_rad").value),
        )
        csv_path = Path(str(self.get_parameter("csv_output_path").value))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            "Starting base-frame action test: "
            f"{len(sequence)} steps, csv_output_path={csv_path}"
        )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields())
            writer.writeheader()
            for index, step in enumerate(sequence, start=1):
                row = self._execute_step(index, step)
                writer.writerow(row)
                handle.flush()
                self.get_logger().info(self._format_row_log(row))
                time.sleep(float(self.get_parameter("step_interval_sec").value))
        self.get_logger().info(f"Action test finished; results written to {csv_path}")

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "step_index",
            "step_name",
            "action",
            "status",
            "reason",
            "start_position",
            "start_quat_xyzw",
            "target_position",
            "target_quat_xyzw",
            "actual_position",
            "actual_quat_xyzw",
            "actual_pose_source",
            "position_error_m",
            "orientation_error_rad",
        ]

    def _execute_step(self, index: int, step: ActionTestStep) -> dict[str, object]:
        row = {
            "step_index": index,
            "step_name": step.name,
            "action": self._json(step.action),
            "status": "",
            "reason": "",
            "start_position": "",
            "start_quat_xyzw": "",
            "target_position": "",
            "target_quat_xyzw": "",
            "actual_position": "",
            "actual_quat_xyzw": "",
            "actual_pose_source": "",
            "position_error_m": "",
            "orientation_error_rad": "",
        }
        if step.is_gripper_step:
            accepted = self._send_gripper_goal(float(step.action[6]))
            row["status"] = "gripper_only"
            row["reason"] = "accepted" if accepted else "gripper_action_unavailable_or_rejected"
            return row

        start_pose = self._lookup_tcp_pose()
        if start_pose is None:
            row["status"] = "fail"
            row["reason"] = "start_tf_unavailable"
            return row
        start_position, start_quat, start_source = start_pose
        row["actual_pose_source"] = start_source
        row["start_position"] = self._json(start_position)
        row["start_quat_xyzw"] = self._json(start_quat)

        target_position, target_quat = apply_tcp_delta(
            start_position,
            start_quat,
            step.action,
            action_scale=float(self.get_parameter("action_scale").value),
            rotation_format="axis_angle",
        )
        row["target_position"] = self._json(target_position)
        row["target_quat_xyzw"] = self._json(target_quat)

        current_joints = self._wait_for_joint_state()
        if current_joints is None:
            row["status"] = "fail"
            row["reason"] = "joint_state_unavailable"
            return row
        target_joints = self._compute_ik(current_joints, target_position, target_quat)
        if target_joints is None:
            row["status"] = "fail"
            row["reason"] = "ik_failed"
            return row
        if not self._send_trajectory_goal(target_joints):
            row["status"] = "fail"
            row["reason"] = "trajectory_rejected_or_failed"
            return row

        actual_pose = self._lookup_tcp_pose()
        if actual_pose is None:
            row["status"] = "fail"
            row["reason"] = "actual_tf_unavailable"
            return row
        actual_position, actual_quat, actual_source = actual_pose
        pos_error, rot_error = pose_errors(target_position, target_quat, actual_position, actual_quat)
        row["actual_position"] = self._json(actual_position)
        row["actual_quat_xyzw"] = self._json(actual_quat)
        row["actual_pose_source"] = actual_source
        row["position_error_m"] = f"{pos_error:.9f}"
        row["orientation_error_rad"] = f"{rot_error:.9f}"
        row["status"] = classify_step_result(
            step,
            pos_error_m=pos_error,
            rot_error_rad=rot_error,
            tolerance_pos_m=float(self.get_parameter("tolerance_pos_m").value),
            tolerance_rot_rad=float(self.get_parameter("tolerance_rot_rad").value),
        )
        row["reason"] = "within_tolerance" if row["status"] == "pass" else "outside_tolerance"
        return row

    def _lookup_tcp_pose(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        source = str(self.get_parameter("tcp_pose_source").value).lower()
        if source == "tf":
            tf_pose = self._lookup_tcp_pose_from_tf()
            if tf_pose is None:
                return None
            position, quat = tf_pose
            return position, quat, "tf"
        if source != "franka_state":
            self.get_logger().warn(f"unknown tcp_pose_source={source!r}; falling back to TF")
            tf_pose = self._lookup_tcp_pose_from_tf()
            if tf_pose is None:
                return None
            position, quat = tf_pose
            return position, quat, "tf_fallback"
        franka_pose = self._tcp_pose_from_franka_state()
        if franka_pose is not None:
            position, quat = franka_pose
            return position, quat, "franka_state_o_t_ee"
        self.get_logger().warn("Franka robot state TCP pose is not available; falling back to TF")
        tf_pose = self._lookup_tcp_pose_from_tf()
        if tf_pose is None:
            return None
        position, quat = tf_pose
        return position, quat, "tf_fallback"

    def _tcp_pose_from_franka_state(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._latest_franka_state_pose is None:
            return None
        ee_position, ee_quat = self._latest_franka_state_pose
        ee_frame = str(self.get_parameter("franka_state_ee_frame").value)
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        if ee_frame == tcp_frame:
            return ee_position.copy(), ee_quat.copy()
        try:
            transform = self._tf_buffer.lookup_transform(ee_frame, tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup {ee_frame}->{tcp_frame} for Franka state TCP offset failed: {exc}")
            return None
        ee_to_tcp_position, ee_to_tcp_quat = transform_msg_to_arrays(transform.transform)
        return compose_pose_xyzw(ee_position, ee_quat, ee_to_tcp_position, ee_to_tcp_quat)

    def _lookup_tcp_pose_from_tf(self) -> tuple[np.ndarray, np.ndarray] | None:
        base_frame = str(self.get_parameter("command_frame").value)
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(base_frame, tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup {base_frame}->{tcp_frame} failed: {exc}")
            return None
        return transform_msg_to_arrays(transform.transform)

    def _wait_for_joint_state(self, timeout_sec: float = 3.0) -> np.ndarray | None:
        deadline = time.time() + timeout_sec
        while self._latest_joints is None and time.time() < deadline:
            time.sleep(0.02)
        if self._latest_joints is None:
            self.get_logger().warn("No complete FR3 joint state received")
            return None
        return self._latest_joints.copy()

    def _compute_ik(
        self,
        current_joints: np.ndarray,
        target_position: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> np.ndarray | None:
        if not self._ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("MoveIt IK service is not available")
            return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = str(self.get_parameter("move_group_name").value)
        request.ik_request.ik_link_name = str(self.get_parameter("tcp_frame").value)
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state = RobotState()
        request.ik_request.robot_state.joint_state.name = self._joint_names
        request.ik_request.robot_state.joint_state.position = np.asarray(current_joints, dtype=float).tolist()
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = str(self.get_parameter("command_frame").value)
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(target_position[0])
        request.ik_request.pose_stamped.pose.position.y = float(target_position[1])
        request.ik_request.pose_stamped.pose.position.z = float(target_position[2])
        request.ik_request.pose_stamped.pose.orientation.x = float(target_quat_xyzw[0])
        request.ik_request.pose_stamped.pose.orientation.y = float(target_quat_xyzw[1])
        request.ik_request.pose_stamped.pose.orientation.z = float(target_quat_xyzw[2])
        request.ik_request.pose_stamped.pose.orientation.w = float(target_quat_xyzw[3])
        request.ik_request.timeout.sec = 0
        request.ik_request.timeout.nanosec = 500_000_000

        future = self._ik_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=2.0):
            self.get_logger().warn("MoveIt IK request timed out")
            return None
        response = future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if response is None else response.error_code.val
            self.get_logger().warn(f"MoveIt IK failed with code {code}")
            return None

        by_name = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(name in by_name for name in self._joint_names):
            self.get_logger().warn("MoveIt IK response did not contain all FR3 joints")
            return None
        return np.array([by_name[name] for name in self._joint_names], dtype=float)

    def _send_trajectory_goal(self, joint_positions: np.ndarray) -> bool:
        if not self._trajectory_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("Trajectory action server is not available")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = make_joint_trajectory(
            self._joint_names,
            joint_positions,
            float(self.get_parameter("trajectory_duration_sec").value),
        )
        send_future = self._trajectory_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, timeout_sec=3.0):
            self.get_logger().warn("Timed out sending trajectory goal")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        duration = float(self.get_parameter("trajectory_duration_sec").value)
        if not self._wait_for_future(result_future, timeout_sec=max(5.0, duration + 3.0)):
            self.get_logger().warn("Timed out waiting for trajectory result")
            return False
        return result_future.result() is not None

    def _send_gripper_goal(self, action_value: float) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Franka gripper move action is not available")
            return False
        goal = Move.Goal()
        goal.width = gripper_width_from_binary_action(
            action_value,
            min_width=float(self.get_parameter("gripper_min_width").value),
            max_width=float(self.get_parameter("gripper_max_width").value),
        )
        goal.speed = float(self.get_parameter("gripper_speed").value)
        send_future = self._gripper_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, timeout_sec=2.0):
            self.get_logger().warn("Timed out sending gripper goal")
            return False
        goal_handle = send_future.result()
        return bool(goal_handle is not None and goal_handle.accepted)

    @staticmethod
    def _wait_for_future(future, *, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        return bool(future.done())

    @staticmethod
    def _json(values: np.ndarray) -> str:
        return json.dumps(np.asarray(values, dtype=float).round(9).tolist())

    @staticmethod
    def _format_row_log(row: dict[str, object]) -> str:
        return (
            f"step {row['step_index']} {row['step_name']}: "
            f"status={row['status']} reason={row['reason']} "
            f"target={row['target_position']} actual={row['actual_position']} "
            f"pos_err={row['position_error_m']} rot_err={row['orientation_error_rad']}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionTestNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
