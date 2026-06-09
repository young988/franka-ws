"""Base policy runtime node — shared observation, inference, and control loop.

Subclasses (VLAPolicyRuntime, BCCubeStackPolicyRuntime) only need to
override ``_declare_parameters()`` and ``_create_observer()``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Move
from geometry_msgs.msg import PoseStamped, Twist
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from franka_policy_runtime.observers.base import BaseObserver
from franka_policy_runtime.utils.pose_math import (
    FR3_JOINT_NAMES,
    apply_tcp_delta,
    compose_pose_xyzw,
    gripper_width_from_binary_action,
    invert_pose_xyzw,
    make_joint_trajectory,
    policy_action_to_cartesian_delta,
    policy_action_to_joint_positions,
    pose_msg_to_arrays,
    split_policy_action,
    transform_msg_to_arrays,
)


class PolicyRuntimeBase(Node):
    """Async single-step inference bridge.

    Continuous policies publish one-shot Cartesian deltas to the realtime
    effort controller. Legacy absolute-pose workflows can still use MoveIt IK
    and FollowJointTrajectory.
    """

    def __init__(self, node_name: str = "franka_policy_runtime") -> None:
        super().__init__(node_name)
        self.declare_parameter("policy_url", "http://127.0.0.1:8000/act")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("wrist_image_topic", "")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("camera_info_topic", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "~/instruction")
        self.declare_parameter("trajectory_action", "/fr3_arm_controller/follow_joint_trajectory")
        self.declare_parameter("control_mode", "cartesian_delta")
        self.declare_parameter("cartesian_delta_topic", "/policy/cartesian_delta")
        self.declare_parameter("joint_target_topic", "/policy/joint_target")
        self.declare_parameter("command_interval_sec", 0.5)
        self.declare_parameter("command_frame", "fr3_link0")
        self.declare_parameter("tcp_frame", "fr3_hand_tcp")
        self.declare_parameter("tcp_pose_source", "tf")
        self.declare_parameter("current_pose_topic", "/franka_robot_state_broadcaster/current_pose")
        self.declare_parameter("move_group_name", "fr3_arm")
        self.declare_parameter("ik_link_name", "")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("ik_request_timeout_sec", 1.0)
        self.declare_parameter("avoid_collisions", False)
        self.declare_parameter("control_retry_sec", 0.5)
        self.declare_parameter("trajectory_duration_sec", 0.5)
        self.declare_parameter("max_joint_delta_rad", 0.25)
        self.declare_parameter("action_scale", 0.5)
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_initial_width", 0.04)
        self.declare_parameter("gripper_speed", 0.05)
        self.declare_parameter("gripper_deadband", 0.002)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)

        self._declare_parameters()

        self._joint_names: list[str] = list(self.get_parameter("joint_names").value)
        self._latest_current_pose: PoseStamped | None = None
        self._latest_current_pose_received_ns: int | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._observer: BaseObserver = self._create_observer()
        self._observer.set_tf_buffer(self._tf_buffer)
        self._gripper_width = float(self.get_parameter("gripper_initial_width").value)
        self._last_gripper_goal = self._gripper_width
        self._observer.update_gripper_width(self._gripper_width)

        self._ik_callback_group = ReentrantCallbackGroup()
        self._control_callback_group = ReentrantCallbackGroup()
        self._ik_client = self.create_client(
            GetPositionIK, str(self.get_parameter("ik_service").value),
            callback_group=self._ik_callback_group,
        )
        self._gripper_client = ActionClient(
            self, Move, str(self.get_parameter("gripper_move_action").value),
            callback_group=self._control_callback_group,
        )
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
            callback_group=self._control_callback_group,
        )
        self._cartesian_delta_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter("cartesian_delta_topic").value),
            10,
        )
        self._joint_target_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_target_topic").value),
            10,
        )
        self._goal_active = False
        self._active_goal_handle = None
        self._control_timer = None

        self._create_subscriptions()
        self._schedule_control(0.0)

        # ---- timing accumulator ----
        self._timings: dict[str, list[float]] = {
            "encode": [], "inference": [], "ik": [],
        }
        self._timing_cycle = 0
        self._timing_log_every = 5

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        """Override to declare policy-specific parameters.

        Call ``super()._declare_parameters()`` **first**, then declare
        additional parameters.
        """

    def _create_observer(self) -> BaseObserver:
        """Override to instantiate the appropriate observer."""
        raise NotImplementedError("subclass must implement _create_observer()")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            lambda msg: self._image_cb(msg, "eye_to_hand"),
            10,
        )
        wrist_image_topic = str(self.get_parameter("wrist_image_topic").value)
        if wrist_image_topic:
            self.create_subscription(
                Image,
                wrist_image_topic,
                lambda msg: self._image_cb(msg, "eye_in_hand"),
                10,
            )
        depth_topic = str(self.get_parameter("depth_topic").value)
        if depth_topic:
            self.create_subscription(
                Image,
                depth_topic,
                lambda msg: self._depth_cb(msg, "eye_to_hand"),
                10,
            )
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        if camera_info_topic:
            self.create_subscription(
                CameraInfo,
                camera_info_topic,
                lambda msg: self._camera_info_cb(msg, "eye_to_hand"),
                10,
            )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            self._current_pose_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("instruction_topic").value),
            self._instruction_cb,
            10,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image, name: str = "eye_to_hand") -> None:
        self._observer.update_image(msg, name=name)

    def _depth_cb(self, msg: Image, name: str = "eye_to_hand") -> None:
        self._observer.update_depth(msg, name=name)

    def _camera_info_cb(self, msg: CameraInfo, name: str = "eye_to_hand") -> None:
        self._observer.update_camera_info(msg, name=name)

    def _joint_state_cb(self, msg: JointState) -> None:
        self._observer.update_joint_state(msg)

    def _current_pose_cb(self, msg: PoseStamped) -> None:
        self._latest_current_pose = msg
        self._latest_current_pose_received_ns = self.get_clock().now().nanoseconds

    def _instruction_cb(self, msg: String) -> None:
        if hasattr(self._observer, "update_instruction"):
            self._observer.update_instruction(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        super().destroy_node()

    # ------------------------------------------------------------------
    # Policy request
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_b64(image: np.ndarray) -> str:
        import base64
        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray(image).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @classmethod
    def _payload_from_observation(cls, observation) -> dict[str, object]:
        payload = dict(observation.payload)
        image = payload.pop("image", None)
        if image is not None:
            image_arr = np.asarray(image, dtype=np.uint8)
            payload["image_b64"] = cls._encode_image_b64(image_arr)
            payload["height"] = int(image_arr.shape[0])
            payload["width"] = int(image_arr.shape[1])
        images = payload.pop("images", None)
        if images:
            image_shapes = {}
            images_b64 = {}
            for name, value in images.items():
                image_arr = np.asarray(value, dtype=np.uint8)
                images_b64[str(name)] = cls._encode_image_b64(image_arr)
                image_shapes[str(name)] = [int(image_arr.shape[0]), int(image_arr.shape[1])]
            payload["images_b64"] = images_b64
            payload["image_shapes"] = image_shapes
        terms = payload.get("terms")
        if terms:
            payload["terms"] = {
                name: np.asarray(value, dtype=float).tolist()
                for name, value in terms.items()
            }
        return payload

    def _request_policy(self, observation) -> np.ndarray:
        import requests

        t0 = time.perf_counter()
        payload = self._payload_from_observation(observation)
        t_encode = time.perf_counter() - t0
        payload["unnorm_key"] = self._unnorm_key
        payload["actions_per_chunk"] = 1
        t1 = time.perf_counter()
        response = requests.post(
            str(self.get_parameter("policy_url").value), json=payload, timeout=120.0)
        t_infer = time.perf_counter() - t1
        response.raise_for_status()
        body = response.json()
        if isinstance(body, str):
            body = json.loads(body)
        actions = body.get("actions", body.get("action", body))
        arr = np.asarray(actions, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._timings["encode"].append(t_encode)
        self._timings["inference"].append(t_infer)
        return arr

    @property
    def _unnorm_key(self) -> str:
        return ""

    @property
    def _rotation_format(self) -> str:
        return "axis_angle"

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _retry_delay_sec(self) -> float:
        return max(0.01, float(self.get_parameter("control_retry_sec").value))

    def _schedule_control(self, delay_sec: float | None = None) -> None:
        if delay_sec is None:
            delay_sec = self._retry_delay_sec()
        if self._control_timer is not None:
            self._control_timer.cancel()

        def _timer_cb() -> None:
            if self._control_timer is not None:
                self._control_timer.cancel()
                self._control_timer = None
            self._control_tick()

        self._control_timer = self.create_timer(
            max(0.001, float(delay_sec)),
            _timer_cb,
            callback_group=self._control_callback_group,
        )

    def _control_tick(self) -> None:
        """Observe, request one action, and dispatch it to the selected controller."""
        if self._goal_active:
            return
        self._goal_active = True

        try:
            self._update_observer_tcp_pose()
            observation = self._observer.observe()
            if not observation.ready:
                self._goal_active = False
                self._schedule_control()
                return

            try:
                actions = self._request_policy(observation)
            except Exception as exc:
                self.get_logger().warn(
                    f"policy request failed: {exc}", throttle_duration_sec=2.0)
                self._goal_active = False
                self._schedule_control()
                return

            action = actions[0]
            self._observer.update_last_action(action)
            control_mode = str(self.get_parameter("control_mode").value)

            if control_mode == "cartesian_delta":
                self._handle_gripper(action)
                self._publish_cartesian_delta(action)
                self._goal_active = False
                self._schedule_control(
                    max(0.01, float(self.get_parameter("command_interval_sec").value))
                )
                return

            if control_mode == "joint_position":
                self._publish_joint_positions(action)
                self._goal_active = False
                self._schedule_control(
                    max(0.01, float(self.get_parameter("command_interval_sec").value))
                )
                return

            if control_mode != "trajectory":
                raise ValueError(
                    "control_mode must be cartesian_delta, joint_position, or trajectory; "
                    f"got {control_mode!r}"
                )

            self._handle_gripper(action)
            tcp_pose = self._update_observer_tcp_pose()
            if tcp_pose is None:
                self._goal_active = False
                self._schedule_control()
                return

            current_position, current_quat = tcp_pose
            target_position, target_quat = apply_tcp_delta(
                current_position, current_quat, action,
                action_scale=float(self.get_parameter("action_scale").value),
                rotation_format=self._rotation_format,
            )

            current_joints = self._observer.latest_joint_positions()
            if current_joints is None:
                self._goal_active = False
                self._schedule_control()
                return

            target_joints = self._compute_ik(current_joints, target_position, target_quat)
            if target_joints is None:
                self._goal_active = False
                self._schedule_control()
                return

            self._send_trajectory_goal(target_joints)
            if not self._goal_active:
                self._schedule_control()
        except Exception:
            self._goal_active = False
            raise

    def _trajectory_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Trajectory goal failed: {exc}")
            self._goal_active = False
            self._active_goal_handle = None
            self._schedule_control()
            return
        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected")
            self._goal_active = False
            self._active_goal_handle = None
            self._schedule_control()
            return
        self._active_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._trajectory_result_cb)

    def _trajectory_result_cb(self, future):
        self._goal_active = False
        self._active_goal_handle = None
        self._schedule_control(0.0)

    # ---- shared helpers ------------------------------------------------

    def _publish_cartesian_delta(self, action: np.ndarray) -> None:
        delta = policy_action_to_cartesian_delta(
            action,
            action_scale=float(self.get_parameter("action_scale").value),
            rotation_format=self._rotation_format,
        )
        message = Twist()
        message.linear.x = float(delta[0])
        message.linear.y = float(delta[1])
        message.linear.z = float(delta[2])
        message.angular.x = float(delta[3])
        message.angular.y = float(delta[4])
        message.angular.z = float(delta[5])
        self._cartesian_delta_publisher.publish(message)
        self.get_logger().info(
            "Published Cartesian delta: "
            f"linear={np.array2string(delta[:3], precision=4, suppress_small=True)} "
            f"rotation_vector={np.array2string(delta[3:], precision=4, suppress_small=True)}",
            throttle_duration_sec=1.0,
        )

    def _publish_joint_positions(self, action: np.ndarray) -> None:
        positions = policy_action_to_joint_positions(action)
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self._joint_names)
        message.position = positions.tolist()
        self._joint_target_publisher.publish(message)
        self.get_logger().info(
            "Published absolute joint target [rad]: "
            f"{np.array2string(positions, precision=4, suppress_small=True)}",
            throttle_duration_sec=1.0,
        )

    def _send_trajectory_goal(self, joint_positions: np.ndarray) -> None:
        """Send a one-point trajectory goal, preempting any in-flight goal."""
        current_joints = self._observer.latest_joint_positions()
        if current_joints is not None:
            delta = np.asarray(joint_positions, dtype=float) - np.asarray(current_joints, dtype=float)
            max_delta = float(np.max(np.abs(delta)))
            limit = float(self.get_parameter("max_joint_delta_rad").value)
            if max_delta > limit:
                self.get_logger().error(
                    "Refusing trajectory goal: IK joint jump exceeds limit "
                    f"max_delta={max_delta:.4f} rad limit={limit:.4f} rad "
                    f"delta={np.array2string(delta, precision=4, suppress_small=True)}"
                )
                self._goal_active = False
                self._active_goal_handle = None
                return
            self.get_logger().info(
                "Sending trajectory goal: "
                f"max_joint_delta={max_delta:.4f} rad "
                f"duration={float(self.get_parameter('trajectory_duration_sec').value):.3f}s",
                throttle_duration_sec=1.0,
            )
        if self._active_goal_handle is not None:
            self._active_goal_handle.cancel_goal_async()
            self._active_goal_handle = None
            self._goal_active = False

        msg = make_joint_trajectory(
            self._joint_names, joint_positions,
            float(self.get_parameter("trajectory_duration_sec").value),
            start_positions=current_joints,
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = msg
        try:
            send_future = self._trajectory_client.send_goal_async(goal)
            send_future.add_done_callback(self._trajectory_goal_response_cb)
        except Exception as exc:
            self.get_logger().warn(f"Failed to send trajectory goal: {exc}")
            self._goal_active = False
            self._active_goal_handle = None

    def _update_observer_tcp_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        tcp_pose = self._current_tcp_pose()
        if tcp_pose is None:
            return None
        current_position, current_quat = tcp_pose
        self._observer.update_tcp_pose(current_position, current_quat)
        return current_position, current_quat

    def _current_tcp_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        source = str(self.get_parameter("tcp_pose_source").value).lower()
        if source in {"current_pose", "franka_current_pose", "franka_state_broadcaster"}:
            pose = self._tcp_pose_from_current_pose_topic()
            if pose is not None:
                return pose
            self.get_logger().warn(
                "Franka current_pose is not available; falling back to TF",
                throttle_duration_sec=2.0,
            )
            return self._lookup_tcp_pose_from_tf()
        if source != "tf":
            self.get_logger().warn(
                f"unknown tcp_pose_source={source!r}; falling back to TF",
                throttle_duration_sec=2.0,
            )
        return self._lookup_tcp_pose_from_tf()

    def _tcp_pose_from_current_pose_topic(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._latest_current_pose is None:
            return None
        command_frame = str(self.get_parameter("command_frame").value)
        frame_id = self._latest_current_pose.header.frame_id
        position, quat = pose_msg_to_arrays(self._latest_current_pose.pose)
        if not frame_id or frame_id == command_frame:
            return position, quat

        try:
            transform = self._tf_buffer.lookup_transform(
                command_frame,
                frame_id,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup {command_frame}->{frame_id} for current_pose failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        frame_position, frame_quat = transform_msg_to_arrays(transform.transform)
        return compose_pose_xyzw(frame_position, frame_quat, position, quat)

    def _lookup_tcp_pose_from_tf(self) -> tuple[np.ndarray, np.ndarray] | None:
        base_frame = str(self.get_parameter("command_frame").value)
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(base_frame, tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup {base_frame}->{tcp_frame} failed: {exc}", throttle_duration_sec=2.0)
            return None

        return transform_msg_to_arrays(transform.transform)

    def _compute_ik(
        self,
        current_joints: np.ndarray,
        target_tcp_position: np.ndarray,
        target_tcp_quat_xyzw: np.ndarray,
    ) -> np.ndarray | None:
        if not self._ik_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("MoveIt IK service is not available", throttle_duration_sec=2.0)
            return None

        request = GetPositionIK.Request()
        group_name = str(self.get_parameter("move_group_name").value)
        command_frame = str(self.get_parameter("command_frame").value)
        ik_link_name = str(self.get_parameter("ik_link_name").value).strip()
        if not ik_link_name:
            ik_link_name = str(self.get_parameter("tcp_frame").value)
        avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
        timeout_sec = max(0.01, float(self.get_parameter("ik_request_timeout_sec").value))
        target_position, target_quat_xyzw = self._target_pose_for_ik_link(
            target_tcp_position,
            target_tcp_quat_xyzw,
            ik_link_name,
        )
        if target_position is None or target_quat_xyzw is None:
            return None

        request.ik_request.group_name = group_name
        request.ik_request.ik_link_name = ik_link_name
        request.ik_request.avoid_collisions = avoid_collisions
        request.ik_request.robot_state = RobotState()
        request.ik_request.robot_state.joint_state.name = self._joint_names
        request.ik_request.robot_state.joint_state.position = np.asarray(current_joints, dtype=float).tolist()
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = command_frame
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(target_position[0])
        request.ik_request.pose_stamped.pose.position.y = float(target_position[1])
        request.ik_request.pose_stamped.pose.position.z = float(target_position[2])
        request.ik_request.pose_stamped.pose.orientation.x = float(target_quat_xyzw[0])
        request.ik_request.pose_stamped.pose.orientation.y = float(target_quat_xyzw[1])
        request.ik_request.pose_stamped.pose.orientation.z = float(target_quat_xyzw[2])
        request.ik_request.pose_stamped.pose.orientation.w = float(target_quat_xyzw[3])
        request.ik_request.timeout.sec = int(timeout_sec)
        request.ik_request.timeout.nanosec = int((timeout_sec - int(timeout_sec)) * 1_000_000_000)

        self.get_logger().info(
            "IK request: "
            f"group={group_name} link={ik_link_name} frame={command_frame} "
            f"avoid_collisions={avoid_collisions} timeout={timeout_sec:.3f}s "
            f"target_pos=[{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}] "
            f"target_quat_xyzw=[{target_quat_xyzw[0]:.4f}, {target_quat_xyzw[1]:.4f}, "
            f"{target_quat_xyzw[2]:.4f}, {target_quat_xyzw[3]:.4f}]"
        )

        future = self._ik_client.call_async(request)
        deadline = time.time() + timeout_sec + 1.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done() or future.result() is None:
            self.get_logger().warn(
                f"MoveIt IK request timed out after {timeout_sec + 1.0:.3f}s"
            )
            return None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(
                f"MoveIt IK failed with code {response.error_code.val} "
                f"(group={group_name}, link={ik_link_name}, frame={command_frame})"
            )
            return None

        by_name = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(name in by_name for name in self._joint_names):
            self.get_logger().warn("MoveIt IK response did not contain all FR3 joints", throttle_duration_sec=2.0)
            return None
        return np.array([by_name[name] for name in self._joint_names], dtype=float)

    def _target_pose_for_ik_link(
        self,
        target_tcp_position: np.ndarray,
        target_tcp_quat_xyzw: np.ndarray,
        ik_link_name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        if ik_link_name == tcp_frame:
            return target_tcp_position, target_tcp_quat_xyzw

        try:
            transform = self._tf_buffer.lookup_transform(
                ik_link_name,
                tcp_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup {ik_link_name}->{tcp_frame} for IK target offset failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None, None

        ik_to_tcp_position, ik_to_tcp_quat = transform_msg_to_arrays(transform.transform)
        tcp_to_ik_position, tcp_to_ik_quat = invert_pose_xyzw(
            ik_to_tcp_position,
            ik_to_tcp_quat,
        )
        target_ik_position, target_ik_quat = compose_pose_xyzw(
            target_tcp_position,
            target_tcp_quat_xyzw,
            tcp_to_ik_position,
            tcp_to_ik_quat,
        )
        self.get_logger().info(
            "Converted TCP target for IK link: "
            f"tcp_frame={tcp_frame} ik_link={ik_link_name} "
            f"target_tcp_pos=[{target_tcp_position[0]:.4f}, {target_tcp_position[1]:.4f}, {target_tcp_position[2]:.4f}] "
            f"target_ik_pos=[{target_ik_position[0]:.4f}, {target_ik_position[1]:.4f}, {target_ik_position[2]:.4f}]",
            throttle_duration_sec=1.0,
        )
        return target_ik_position, target_ik_quat

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _maybe_log_timings(self) -> None:
        self._timing_cycle += 1
        if self._timing_cycle % self._timing_log_every != 0:
            return
        lines = [f"--- Timing (avg over last {self._timing_log_every} cycles) ---"]
        for key, label in [
            ("encode", "1. encode"), ("inference", "2. inference"), ("ik", "3. IK"),
        ]:
            vals = self._timings.get(key, [])
            recent = vals[-self._timing_log_every:]
            if recent:
                avg = sum(recent) / len(recent)
                lines.append(f"  {label:20s} {avg*1000:7.1f} ms")
            if len(vals) > 200:
                self._timings[key] = vals[-100:]
        self.get_logger().info("\n".join(lines))

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _handle_gripper(self, action: np.ndarray) -> None:
        _, gripper_action = split_policy_action(action)
        min_width = float(self.get_parameter("gripper_min_width").value)
        max_width = float(self.get_parameter("gripper_max_width").value)
        self._gripper_width = gripper_width_from_binary_action(
            gripper_action,
            min_width=min_width,
            max_width=max_width,
        )
        self._observer.update_gripper_width(self._gripper_width)
        if abs(self._gripper_width - self._last_gripper_goal) < float(self.get_parameter("gripper_deadband").value):
            return
        if not self._gripper_client.server_is_ready():
            self.get_logger().warn("Franka gripper move action is not available", throttle_duration_sec=2.0)
            return
        goal = Move.Goal()
        goal.width = self._gripper_width
        goal.speed = float(self.get_parameter("gripper_speed").value)
        self._gripper_client.send_goal_async(goal)
        self._last_gripper_goal = self._gripper_width


def run_node(node_cls, *, args=None, num_threads: int = 2) -> None:
    """Spin a policy runtime node with a MultiThreadedExecutor."""
    rclpy.init(args=args)
    node = node_cls()
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
