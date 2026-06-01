"""Policy runtime node for Franka reference generation."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from franka_msgs.action import Move
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

from franka_policy_runtime.action_queue import ActionChunk, WeightedActionQueue
from franka_policy_runtime.reference import (
    apply_tcp_delta,
    make_joint_trajectory,
    split_policy_action,
)
from franka_policy_runtime.runtime_config import FR3_JOINT_NAMES


class PolicyRuntimeNode(Node):
    """Async inference scheduler and reference publisher.

    Modes:
    - single_step: wait for each predicted action to be consumed before asking again.
    - chunk_async: request a new chunk before the current queue is exhausted and fuse overlap.
    - streaming: replace the queue with each latest policy output.
    """

    def __init__(self) -> None:
        super().__init__("franka_policy_runtime")
        self.declare_parameter("mode", "single_step")
        self.declare_parameter("policy_url", "http://127.0.0.1:8000/act")
        self.declare_parameter("instruction", "move up slightly")
        self.declare_parameter("unnorm_key", "bridge_orig")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("reference_topic", "/franka_policy_controller/reference")
        self.declare_parameter("command_frame", "fr3_link0")
        self.declare_parameter("tcp_frame", "fr3_hand_tcp")
        self.declare_parameter("move_group_name", "fr3_arm")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("control_period_sec", 0.2)
        self.declare_parameter("actions_per_chunk", 1)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("fusion_new_weight", 0.6)
        self.declare_parameter("max_translation_delta", 0.05)
        self.declare_parameter("max_rotation_delta", 0.25)
        self.declare_parameter("gripper_move_action", "/franka_gripper/move")
        self.declare_parameter("gripper_min_width", 0.0)
        self.declare_parameter("gripper_max_width", 0.08)
        self.declare_parameter("gripper_initial_width", 0.04)
        self.declare_parameter("gripper_speed", 0.05)
        self.declare_parameter("gripper_deadband", 0.002)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)

        self._queue = WeightedActionQueue(action_dim=7)
        self._queue_lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_joint_positions: np.ndarray | None = None
        self._data_lock = threading.Lock()
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._gripper_width = float(self.get_parameter("gripper_initial_width").value)
        self._last_gripper_goal = self._gripper_width
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._ik_callback_group = ReentrantCallbackGroup()
        self._control_callback_group = ReentrantCallbackGroup()
        self._ik_client = self.create_client(
            GetPositionIK,
            str(self.get_parameter("ik_service").value),
            callback_group=self._ik_callback_group,
        )
        self._gripper_client = ActionClient(
            self,
            Move,
            str(self.get_parameter("gripper_move_action").value),
            callback_group=self._control_callback_group,
        )

        self._reference_pub = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter("reference_topic").value),
            10,
        )
        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self._image_cb, 10)
        self.create_subscription(JointState, str(self.get_parameter("joint_state_topic").value), self._joint_state_cb, 10)
        self.create_timer(
            float(self.get_parameter("control_period_sec").value),
            self._control_tick,
            callback_group=self._control_callback_group,
        )
        # ---- timing accumulator ----
        self._timings: dict[str, list[float]] = {
            "encode": [], "inference": [], "queue_ops": [],
            "tf_lookup": [], "apply_delta": [], "ik": [], "publish": [],
        }
        self._timing_cycle = 0
        self._timing_log_every = 5

        self._inference_thread.start()

    def destroy_node(self) -> None:
        self._running = False
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)
        super().destroy_node()

    def _image_cb(self, msg: Image) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.height and msg.width:
            arr = arr.reshape((msg.height, msg.width, -1))
        with self._data_lock:
            self._latest_image = arr.copy()

    def _joint_state_cb(self, msg: JointState) -> None:
        joint_names = list(self.get_parameter("joint_names").value)
        by_name = dict(zip(msg.name, msg.position))
        if not all(name in by_name for name in joint_names):
            return
        with self._data_lock:
            self._latest_joint_positions = np.array([by_name[name] for name in joint_names], dtype=float)

    def _inference_loop(self) -> None:
        while self._running and rclpy.ok():
            with self._data_lock:
                image = None if self._latest_image is None else self._latest_image.copy()
            if image is None:
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
                actions = self._request_policy(image, actions_per_chunk)
                chunk = ActionChunk(actions=actions)
                with self._queue_lock:
                    if mode == "chunk_async" and self._queue.size > 0:
                        self._queue.fuse(chunk, float(self.get_parameter("fusion_new_weight").value))
                    else:
                        self._queue.replace(chunk)
            except Exception as exc:
                self.get_logger().warn(f"policy request failed: {exc}", throttle_duration_sec=2.0)
                time.sleep(0.1)

    def _request_policy(self, image: np.ndarray, actions_per_chunk: int) -> np.ndarray:
        import base64
        import io

        import requests
        from PIL import Image as PILImage

        t0 = time.perf_counter()
        buf = io.BytesIO()
        PILImage.fromarray(image).save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        t_encode = time.perf_counter() - t0

        payload = {
            "image_b64": image_b64,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "instruction": str(self.get_parameter("instruction").value),
            "unnorm_key": str(self.get_parameter("unnorm_key").value),
            "actions_per_chunk": actions_per_chunk,
        }
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

    def _control_tick(self) -> None:
        t0 = time.perf_counter()
        with self._queue_lock:
            action = self._queue.pop_next()
        if action is None:
            return
        with self._data_lock:
            current = None if self._latest_joint_positions is None else self._latest_joint_positions.copy()
        if current is None:
            return
        target = self._action_to_joint_reference(current, action)
        if target is None:
            return
        msg = make_joint_trajectory(
            list(self.get_parameter("joint_names").value),
            target,
            float(self.get_parameter("control_period_sec").value),
        )
        t_pub = time.perf_counter()
        self._reference_pub.publish(msg)
        self._handle_gripper(action)
        self._timings["queue_ops"].append(t_pub - t0)
        self._timings["publish"].append(time.perf_counter() - t_pub)
        self._maybe_log_timings()

    def _action_to_joint_reference(self, current_joints: np.ndarray, action: np.ndarray) -> np.ndarray | None:
        base_frame = str(self.get_parameter("command_frame").value)
        tcp_frame = str(self.get_parameter("tcp_frame").value)
        t0 = time.perf_counter()
        try:
            transform = self._tf_buffer.lookup_transform(base_frame, tcp_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup {base_frame}->{tcp_frame} failed: {exc}", throttle_duration_sec=2.0)
            return None
        t_tf = time.perf_counter()

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
        target_position, target_quat = apply_tcp_delta(
            current_position,
            current_quat,
            action,
            max_translation_delta=float(self.get_parameter("max_translation_delta").value),
            max_rotation_delta=float(self.get_parameter("max_rotation_delta").value),
        )
        t_delta = time.perf_counter()
        result = self._compute_ik(current_joints, target_position, target_quat)
        t_ik = time.perf_counter()

        self._timings["tf_lookup"].append(t_tf - t0)
        self._timings["apply_delta"].append(t_delta - t_tf)
        self._timings["ik"].append(t_ik - t_delta)
        return result

    def _compute_ik(
        self,
        current_joints: np.ndarray,
        target_position: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> np.ndarray | None:
        if not self._ik_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("MoveIt IK service is not available", throttle_duration_sec=2.0)
            return None

        joint_names = list(self.get_parameter("joint_names").value)
        request = GetPositionIK.Request()
        request.ik_request.group_name = str(self.get_parameter("move_group_name").value)
        request.ik_request.ik_link_name = str(self.get_parameter("tcp_frame").value)
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state = RobotState()
        request.ik_request.robot_state.joint_state.name = joint_names
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
        request.ik_request.timeout.nanosec = 500_000_000  # 0.5 s for MoveIt collision IK

        future = self._ik_client.call_async(request)
        # Poll-wait instead of spin_until_future_complete —
        # compatible with MultiThreadedExecutor (other thread spins the response).
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done() or future.result() is None:
            self.get_logger().warn("MoveIt IK request timed out", throttle_duration_sec=2.0)
            return None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"MoveIt IK failed with code {response.error_code.val}", throttle_duration_sec=2.0)
            return None

        by_name = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(name in by_name for name in joint_names):
            self.get_logger().warn("MoveIt IK response did not contain all FR3 joints", throttle_duration_sec=2.0)
            return None
        return np.array([by_name[name] for name in joint_names], dtype=float)

    def _maybe_log_timings(self) -> None:
        self._timing_cycle += 1
        if self._timing_cycle % self._timing_log_every != 0:
            return
        lines = ["--- Timing (avg over last %d cycles) ---" % self._timing_log_every]
        labels = [
            ("encode",     "1. JPEG+base64"),
            ("inference",  "2. Server infer"),
            ("queue_ops",  "3. Queue pop"),
            ("tf_lookup",  "4. TF lookup"),
            ("apply_delta","5. Delta apply"),
            ("ik",         "6. MoveIt IK"),
            ("publish",    "7. Publish ref"),
        ]
        for key, label in labels:
            vals = self._timings.get(key, [])
            recent = vals[-self._timing_log_every:]
            if recent:
                avg = sum(recent) / len(recent)
                lines.append(f"  {label:20s} {avg*1000:7.1f} ms")
            if len(vals) > 200:
                self._timings[key] = vals[-100:]
        # sum excluding server inference (which dominates)
        all_recent = []
        for key, _ in labels:
            vals = self._timings.get(key, [])
            if vals:
                all_recent.extend(vals[-self._timing_log_every:])
        if all_recent:
            total_per_cycle = sum(all_recent) / self._timing_log_every
            lines.append(f"  {'TOTAL (excl infer)':20s} {total_per_cycle*1000:7.1f} ms")
        self.get_logger().info("\n".join(lines))

    def _handle_gripper(self, action: np.ndarray) -> None:
        _, gripper_delta = split_policy_action(action)
        if abs(gripper_delta) < float(self.get_parameter("gripper_deadband").value):
            return
        min_width = float(self.get_parameter("gripper_min_width").value)
        max_width = float(self.get_parameter("gripper_max_width").value)
        self._gripper_width = float(np.clip(self._gripper_width + gripper_delta, min_width, max_width))
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyRuntimeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
