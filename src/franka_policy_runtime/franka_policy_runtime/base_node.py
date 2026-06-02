"""Base policy runtime node — shared control / inference / pose-command loop.

Subclasses (VLAPolicyRuntime, BCCubeStackPolicyRuntime) only need to
override ``_declare_parameters()`` and ``_create_observer()``.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from franka_msgs.action import Move
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from franka_policy_runtime.action_queue import ActionChunk, WeightedActionQueue
from franka_policy_runtime.cartesian_backend import CartesianPoseBackend, PoseState
from franka_policy_runtime.observers.base import BaseObserver
from franka_policy_runtime.reference import (
    gripper_width_from_binary_action,
    split_policy_action,
)
from franka_policy_runtime.runtime_config import FR3_JOINT_NAMES


class PolicyRuntimeBase(Node):
    """Async inference scheduler and Cartesian pose command publisher.

    Modes:
    - single_step: wait for each predicted action to be consumed before
      asking again.
    - chunk_async: request a new chunk before the current queue is
      exhausted and fuse overlap.
    - streaming: replace the queue with each latest policy output.
    """

    def __init__(self, node_name: str = "franka_policy_runtime") -> None:
        super().__init__(node_name)
        self.declare_parameter("mode", "single_step")
        self.declare_parameter("policy_url", "http://127.0.0.1:8000/act")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("wrist_image_topic", "")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("camera_info_topic", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "~/instruction")
        self.declare_parameter("cartesian_command_topic", "/franka_cartesian_pose_controller/reference")
        self.declare_parameter("command_frame", "fr3_link0")
        self.declare_parameter("policy_tcp_frame", "fr3_hand_tcp")
        self.declare_parameter("franka_ee_frame", "fr3_hand_tcp")
        self.declare_parameter("control_period_sec", 0.2)
        self.declare_parameter("actions_per_chunk", 1)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("fusion_new_weight", 0.6)
        self.declare_parameter("action_scale", 0.5)
        self.declare_parameter("max_translation_step_per_tick", 0.01)
        self.declare_parameter("max_rotation_step_per_tick", 0.1)
        self.declare_parameter("pose_sync_reset_threshold", 0.05)
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_initial_width", 0.04)
        self.declare_parameter("gripper_speed", 0.05)
        self.declare_parameter("gripper_deadband", 0.002)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)

        # Let subclass add its own parameters.
        self._declare_parameters()

        # Cache shared parameter values on attributes for convenience.
        self._joint_names: list[str] = list(self.get_parameter("joint_names").value)

        self._queue = WeightedActionQueue(action_dim=7)
        self._queue_lock = threading.Lock()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._observer: BaseObserver = self._create_observer()
        self._observer.set_tf_buffer(self._tf_buffer)
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._gripper_width = float(self.get_parameter("gripper_initial_width").value)
        self._last_gripper_goal = self._gripper_width
        self._observer.update_gripper_width(self._gripper_width)
        self._control_callback_group = ReentrantCallbackGroup()
        self._gripper_client = ActionClient(
            self,
            Move,
            str(self.get_parameter("gripper_move_action").value),
            callback_group=self._control_callback_group,
        )

        # Cartesian pose backend and publisher.
        self._cartesian_backend = CartesianPoseBackend(
            action_scale=float(self.get_parameter("action_scale").value),
            rotation_format=self._rotation_format,
            max_translation_step_per_tick=float(self.get_parameter("max_translation_step_per_tick").value),
            max_rotation_step_per_tick=float(self.get_parameter("max_rotation_step_per_tick").value),
            pose_sync_reset_threshold=float(self.get_parameter("pose_sync_reset_threshold").value),
        )
        self._cartesian_command_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("cartesian_command_topic").value),
            10,
        )

        self._create_subscriptions()
        self.create_timer(
            float(self.get_parameter("control_period_sec").value),
            self._control_tick,
            callback_group=self._control_callback_group,
        )

        # ---- timing accumulator ----
        self._timings: dict[str, list[float]] = {
            "encode": [], "inference": [], "queue_ops": [],
            "tf_lookup": [], "apply_delta": [], "publish": [],
        }
        self._timing_cycle = 0
        self._timing_log_every = 5

        self._inference_thread.start()

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

    def _instruction_cb(self, msg: String) -> None:
        if hasattr(self._observer, "update_instruction"):
            self._observer.update_instruction(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._running = False
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)
        super().destroy_node()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while self._running and rclpy.ok():
            self._update_observer_tcp_pose()
            observation = self._observer.observe()
            if not observation.ready:
                time.sleep(0.05)
                continue

            mode = str(self.get_parameter("mode").value)
            with self._queue_lock:
                queue_size = self._queue.size
            actions_per_chunk = max(1, int(self.get_parameter("actions_per_chunk").value))
            threshold = float(self.get_parameter("chunk_size_threshold").value)
            should_request = (
                mode == "streaming"
                or queue_size == 0
                or (mode == "chunk_async" and queue_size <= max(1, int(actions_per_chunk * threshold)))
            )
            if not should_request:
                time.sleep(0.02)
                continue

            try:
                actions = self._request_policy(observation, actions_per_chunk)
                chunk = ActionChunk(actions=actions)
                with self._queue_lock:
                    if mode == "chunk_async" and self._queue.size > 0:
                        self._queue.fuse(chunk, float(self.get_parameter("fusion_new_weight").value))
                    else:
                        self._queue.replace(chunk)
            except Exception as exc:
                self.get_logger().warn(f"policy request failed: {exc}", throttle_duration_sec=2.0)
                time.sleep(0.1)

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

    def _request_policy(self, observation, actions_per_chunk: int) -> np.ndarray:
        import requests

        t0 = time.perf_counter()
        payload = self._payload_from_observation(observation)
        t_encode = time.perf_counter() - t0

        payload["unnorm_key"] = self._unnorm_key
        payload["actions_per_chunk"] = actions_per_chunk
        t1 = time.perf_counter()
        response = requests.post(str(self.get_parameter("policy_url").value), json=payload, timeout=120.0)
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
        """Override to provide an unnormalization key (VLA-specific)."""
        return ""

    @property
    def _rotation_format(self) -> str:
        """Override to set the rotation delta format: ``"axis_angle"`` or ``"rpy"``."""
        return "axis_angle"

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        t0 = time.perf_counter()
        tcp_pose = self._update_observer_tcp_pose()
        if tcp_pose is None:
            return

        measured_pose = PoseState(position=tcp_pose[0], quat_xyzw=tcp_pose[1])
        if self._cartesian_backend.commanded_pose is None:
            self._cartesian_backend.reset(measured_pose)
        else:
            self._cartesian_backend.maybe_resync(measured_pose)

        with self._queue_lock:
            action = self._queue.pop_next()
        if action is not None:
            self._observer.update_last_action(action)
            self._cartesian_backend.ingest_action(action)
            self._handle_gripper(action)

        commanded_pose = self._cartesian_backend.step_commanded_pose()
        msg = PoseStamped()
        msg.header.frame_id = str(self.get_parameter("command_frame").value)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(commanded_pose.position[0])
        msg.pose.position.y = float(commanded_pose.position[1])
        msg.pose.position.z = float(commanded_pose.position[2])
        msg.pose.orientation.x = float(commanded_pose.quat_xyzw[0])
        msg.pose.orientation.y = float(commanded_pose.quat_xyzw[1])
        msg.pose.orientation.z = float(commanded_pose.quat_xyzw[2])
        msg.pose.orientation.w = float(commanded_pose.quat_xyzw[3])
        t_pub = time.perf_counter()
        self._cartesian_command_pub.publish(msg)
        self._timings["queue_ops"].append(t_pub - t0)
        self._timings["publish"].append(time.perf_counter() - t_pub)
        self._maybe_log_timings()

    def _update_observer_tcp_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        base_frame = str(self.get_parameter("command_frame").value)
        policy_tcp_frame = str(self.get_parameter("policy_tcp_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(base_frame, policy_tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup {base_frame}->{policy_tcp_frame} failed: {exc}", throttle_duration_sec=2.0)
            return None

        current_position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=float)
        current_quat = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ], dtype=float)
        self._observer.update_tcp_pose(current_position, current_quat)
        return current_position, current_quat

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _maybe_log_timings(self) -> None:
        self._timing_cycle += 1
        if self._timing_cycle % self._timing_log_every != 0:
            return
        lines = ["--- Timing (avg over last %d cycles) ---" % self._timing_log_every]
        labels = [
            ("encode",     "1. JPEG+base64"),
            ("inference",  "2. Server infer"),
            ("queue_ops",  "3. Queue/step"),
            ("tf_lookup",  "4. TF lookup"),
            ("apply_delta","5. Target update"),
            ("publish",    "6. Publish pose"),
        ]
        for key, label in labels:
            vals = self._timings.get(key, [])
            recent = vals[-self._timing_log_every:]
            if recent:
                avg = sum(recent) / len(recent)
                lines.append(f"  {label:20s} {avg*1000:7.1f} ms")
            if len(vals) > 200:
                self._timings[key] = vals[-100:]
        all_recent = []
        for key, _ in labels:
            vals = self._timings.get(key, [])
            if vals:
                all_recent.extend(vals[-self._timing_log_every:])
        if all_recent:
            total_per_cycle = sum(all_recent) / self._timing_log_every
            lines.append(f"  {'TOTAL (excl infer)':20s} {total_per_cycle*1000:7.1f} ms")
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
