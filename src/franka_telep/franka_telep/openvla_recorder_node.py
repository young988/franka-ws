from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

from franka_telep.franka_mapping import FR3_JOINT_NAMES
from franka_telep.openvla_dataset import (
    OpenVLAEpisodeWriter,
    center_crop_resize_rgb,
    openvla_action,
    openvla_state,
)


@dataclass
class ObservationSample:
    image_rgb: np.ndarray
    wrist_image_rgb: np.ndarray | None
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    joint_positions: np.ndarray
    gripper_width: float
    timestamp_sec: float


class OpenVLADatasetRecorder(Node):
    """Record uArm teleoperation demonstrations for OpenVLA RLDS conversion."""

    def __init__(self) -> None:
        super().__init__("openvla_dataset_recorder")
        self.declare_parameter(
            "image_topic", "/eye_to_hand_camera/eye_to_hand_camera/color/image_raw")
        self.declare_parameter("wrist_image_topic", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter(
            "gripper_joint_state_topic", "/franka_gripper/joint_states")
        self.declare_parameter(
            "tcp_pose_topic", "/franka_robot_state_broadcaster/current_pose")
        self.declare_parameter("home_ready_topic", "/franka_teleop/home_ready")
        self.declare_parameter("recording_topic", "/franka_teleop/dataset_recording")
        self.declare_parameter("instruction_topic", "/franka_teleop/dataset_instruction")
        self.declare_parameter("dataset_root", "~/franka_openvla_data")
        self.declare_parameter("dataset_name", "franka_teleop")
        self.declare_parameter("instruction", "pick up the object")
        self.declare_parameter("image_size", 256)
        self.declare_parameter("sample_rate_hz", 10.0)
        self.declare_parameter("max_data_age_sec", 0.2)
        self.declare_parameter("gripper_open_threshold", 0.04)
        self.declare_parameter("require_home_ready", True)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("joint_names", FR3_JOINT_NAMES)
        self.declare_parameter(
            "gripper_joint_names", ["fr3_finger_joint1", "fr3_finger_joint2"])

        self._lock = Lock()
        self._joint_names = [str(value) for value in self.get_parameter("joint_names").value]
        self._gripper_joint_names = [
            str(value) for value in self.get_parameter("gripper_joint_names").value
        ]
        self._latest_pose: tuple[np.ndarray, np.ndarray, float] | None = None
        self._latest_joints: tuple[np.ndarray, float] | None = None
        self._latest_gripper: tuple[float, float] | None = None
        self._latest_wrist_image: tuple[np.ndarray, float] | None = None
        self._last_sample_time_sec = float("-inf")
        self._pending_sample: ObservationSample | None = None
        self._writer: OpenVLAEpisodeWriter | None = None
        self._recording = False
        self._home_ready = not bool(self.get_parameter("require_home_ready").value)
        self._instruction = str(self.get_parameter("instruction").value).strip()

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("tcp_pose_topic").value),
            self._pose_callback,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_callback,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("gripper_joint_state_topic").value),
            self._gripper_joint_state_callback,
            20,
        )
        wrist_topic = str(self.get_parameter("wrist_image_topic").value).strip()
        if wrist_topic:
            self.create_subscription(Image, wrist_topic, self._wrist_image_callback, 10)
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("recording_topic").value),
            self._recording_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("instruction_topic").value),
            self._instruction_callback,
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

        if self._home_ready and bool(self.get_parameter("auto_start").value):
            self._start_episode()
        self.get_logger().info(
            "OpenVLA recorder ready: publish Bool to "
            f"{self.get_parameter('recording_topic').value} to start/stop episodes"
        )

    def destroy_node(self) -> None:
        with self._lock:
            self._stop_episode()
        super().destroy_node()

    def _pose_callback(self, message: PoseStamped) -> None:
        pose = message.pose
        received = self._message_time_sec(message.header.stamp)
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        quaternion = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        with self._lock:
            self._latest_pose = position, quaternion, received

    def _joint_state_callback(self, message: JointState) -> None:
        by_name = {name: index for index, name in enumerate(message.name)}
        received = self._message_time_sec(message.header.stamp)
        with self._lock:
            if all(name in by_name for name in self._joint_names):
                joints = np.array(
                    [message.position[by_name[name]] for name in self._joint_names],
                    dtype=np.float32,
                )
                self._latest_joints = joints, received
            gripper_width = self._gripper_width_from_message(message, by_name)
            if gripper_width is not None:
                self._latest_gripper = gripper_width, received

    def _gripper_joint_state_callback(self, message: JointState) -> None:
        by_name = {name: index for index, name in enumerate(message.name)}
        gripper_width = self._gripper_width_from_message(message, by_name)
        if gripper_width is None:
            return
        received = self._message_time_sec(message.header.stamp)
        with self._lock:
            self._latest_gripper = gripper_width, received

    def _wrist_image_callback(self, message: Image) -> None:
        try:
            image = self._convert_image(message)
        except Exception as exception:
            self.get_logger().warn(f"Failed to convert wrist image: {exception}")
            return
        with self._lock:
            self._latest_wrist_image = image, self._message_time_sec(message.header.stamp)

    def _image_callback(self, message: Image) -> None:
        try:
            image = self._convert_image(message)
        except Exception as exception:
            self.get_logger().warn(f"Failed to convert primary image: {exception}")
            return
        timestamp_sec = self._message_time_sec(message.header.stamp)
        with self._lock:
            if not self._recording:
                return
            sample_period = 1.0 / max(
                0.1, float(self.get_parameter("sample_rate_hz").value))
            if timestamp_sec - self._last_sample_time_sec < sample_period:
                return
            sample = self._make_sample(image, timestamp_sec)
            if sample is None:
                return
            self._last_sample_time_sec = timestamp_sec
            if self._pending_sample is not None:
                try:
                    self._write_transition(self._pending_sample, sample)
                except Exception as exception:
                    self.get_logger().error(
                        f"Failed to write dataset transition: {exception}")
                    self._stop_episode()
                    return
            self._pending_sample = sample

    def _recording_callback(self, message: Bool) -> None:
        with self._lock:
            if message.data:
                self._start_episode()
            else:
                self._stop_episode()

    def _instruction_callback(self, message: String) -> None:
        instruction = message.data.strip()
        if not instruction:
            return
        with self._lock:
            if self._recording:
                self.get_logger().warn(
                    "Ignoring instruction change during an active episode")
                return
            self._instruction = instruction
            self.get_logger().info(f"Next episode instruction: {instruction}")

    def _home_ready_callback(self, message: Bool) -> None:
        with self._lock:
            self._home_ready = bool(message.data)
            if (
                self._home_ready
                and bool(self.get_parameter("auto_start").value)
                and not self._recording
            ):
                self._start_episode()
            elif not self._home_ready and self._recording:
                self._stop_episode()

    def _start_episode(self) -> None:
        if self._recording:
            return
        if bool(self.get_parameter("require_home_ready").value) and not self._home_ready:
            self.get_logger().warn("Cannot start dataset episode before Franka home-ready")
            return
        wrist_topic = str(self.get_parameter("wrist_image_topic").value).strip()
        self._writer = OpenVLAEpisodeWriter(
            str(self.get_parameter("dataset_root").value),
            dataset_name=str(self.get_parameter("dataset_name").value),
            instruction=self._instruction,
            image_size=int(self.get_parameter("image_size").value),
            has_wrist_image=bool(wrist_topic),
        )
        self._pending_sample = None
        self._last_sample_time_sec = float("-inf")
        self._recording = True
        self.get_logger().info(
            f"Recording OpenVLA episode {self._writer.episode_id}: {self._instruction}")

    def _stop_episode(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._pending_sample = None
        writer = self._writer
        self._writer = None
        if writer is None or writer.step_count == 0:
            if writer is not None:
                writer.abort()
            self.get_logger().warn("Discarded episode without complete transitions")
            return
        try:
            output_path = writer.finalize()
        except Exception as exception:
            self.get_logger().error(f"Failed to finalize OpenVLA episode: {exception}")
            return
        self.get_logger().info(
            f"Saved {writer.step_count} OpenVLA transitions to {output_path}")

    def _make_sample(
        self, image_rgb: np.ndarray, timestamp_sec: float
    ) -> ObservationSample | None:
        if (
            self._latest_pose is None
            or self._latest_joints is None
            or self._latest_gripper is None
        ):
            self.get_logger().warn(
                "Waiting for TCP pose, arm joints, and gripper state",
                throttle_duration_sec=2.0,
            )
            return None
        position, quaternion, pose_time = self._latest_pose
        joints, joint_time = self._latest_joints
        gripper_width, _ = self._latest_gripper
        max_age = max(0.0, float(self.get_parameter("max_data_age_sec").value))
        if max(
            abs(timestamp_sec - pose_time),
            abs(timestamp_sec - joint_time),
        ) > max_age:
            self.get_logger().warn(
                "Skipping image because robot state is stale",
                throttle_duration_sec=2.0,
            )
            return None

        wrist = None
        wrist_topic = str(self.get_parameter("wrist_image_topic").value).strip()
        if wrist_topic:
            if self._latest_wrist_image is None:
                return None
            wrist, wrist_time = self._latest_wrist_image
            if abs(timestamp_sec - wrist_time) > max_age:
                return None
        return ObservationSample(
            image_rgb=image_rgb.copy(),
            wrist_image_rgb=None if wrist is None else wrist.copy(),
            position=position.copy(),
            quaternion_xyzw=quaternion.copy(),
            joint_positions=joints.copy(),
            gripper_width=gripper_width,
            timestamp_sec=timestamp_sec,
        )

    def _gripper_width_from_message(
        self, message: JointState, by_name: dict[str, int]
    ) -> float | None:
        finger_positions = [
            float(message.position[by_name[name]])
            for name in self._gripper_joint_names
            if name in by_name and by_name[name] < len(message.position)
        ]
        if not finger_positions:
            return None
        return float(sum(finger_positions))

    def _write_transition(
        self, current: ObservationSample, next_sample: ObservationSample
    ) -> None:
        if self._writer is None:
            return
        threshold = float(self.get_parameter("gripper_open_threshold").value)
        state = openvla_state(
            current.position,
            current.quaternion_xyzw,
            current.gripper_width,
            gripper_open_threshold=threshold,
        )
        action = openvla_action(
            current.position,
            current.quaternion_xyzw,
            next_sample.position,
            next_sample.quaternion_xyzw,
            next_sample.gripper_width,
            gripper_open_threshold=threshold,
        )
        self._writer.append(
            image_rgb=current.image_rgb,
            wrist_image_rgb=current.wrist_image_rgb,
            state=state,
            joint_positions=current.joint_positions,
            action=action,
            timestamp_sec=current.timestamp_sec,
        )

    def _convert_image(self, message: Image) -> np.ndarray:
        rgb = ros_image_to_rgb(message)
        return center_crop_resize_rgb(
            rgb, int(self.get_parameter("image_size").value))

    def _message_time_sec(self, stamp) -> float:
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            return self.get_clock().now().nanoseconds * 1.0e-9
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def ros_image_to_rgb(message: Image) -> np.ndarray:
    """Decode common 8-bit ROS image encodings without cv_bridge/OpenCV."""
    encoding = message.encoding.lower()
    channel_counts = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    if encoding not in channel_counts:
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    height = int(message.height)
    width = int(message.width)
    channels = channel_counts[encoding]
    row_bytes = width * channels
    step = int(message.step)
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError(
            f"invalid image dimensions or step: {width}x{height}, step={step}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = height * step
    if raw.size < required_bytes:
        raise ValueError(
            f"image data has {raw.size} bytes, expected at least {required_bytes}")
    rows = raw[:required_bytes].reshape(height, step)
    pixels = rows[:, :row_bytes].reshape(height, width, channels)

    if encoding == "mono8":
        return np.repeat(pixels, 3, axis=2)
    if encoding in ("bgr8", "bgra8"):
        return pixels[:, :, [2, 1, 0]].copy()
    return pixels[:, :, :3].copy()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OpenVLADatasetRecorder()
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
