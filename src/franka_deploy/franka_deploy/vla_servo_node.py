import json
import threading
import time
from typing import Optional

from cv_bridge import CvBridge
import numpy as np
import rclpy
from franka_msgs.action import Grasp, Move
from geometry_msgs.msg import TwistStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from franka_deploy.action_mapping import (
    TwistLimits,
    action_to_twist,
    gripper_should_close,
)


class OpenVLAServoNode(Node):
    """OpenVLA → MoveIt Servo bridge.

    Architecture
    ------------
    - A **background thread** continuously runs VLA inference on the latest
      camera image (async, non-blocking).
    - The ROS **timer callback** (non-blocking) reads the most recent completed
      action, converts it to a twist velocity command using the **actual** time
      delta between actions, applies EMA smoothing, and publishes to
      MoveIt Servo.

    This avoids the "sudden jump" problem caused by:
      1. Synchronous HTTP blocking → variable / unknown inference latency
      2. Fixed ``control_frequency`` multiplier that ignored real dt
      3. Zero-decay during inference gaps creating bounce-back when a new
         action finally arrives
    """

    def __init__(self):
        super().__init__('openvla_servo')

        # ---- parameters ----
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/act')
        self.declare_parameter('instruction', 'move the object')
        self.declare_parameter('unnorm_key', 'bridge_orig')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('twist_topic', '/servo_node/delta_twist_cmds')
        self.declare_parameter('command_frame', 'fr3_link0')
        self.declare_parameter('control_frequency', 5.0)
        self.declare_parameter('request_timeout_sec', 10.0)
        self.declare_parameter('image_timeout_sec', 2.0)
        self.declare_parameter('max_linear_velocity', 0.05)
        self.declare_parameter('max_angular_velocity', 0.25)
        self.declare_parameter('max_linear_step', 0.002)
        self.declare_parameter('max_angular_step', 0.01)
        self.declare_parameter('smoothing_alpha', 0.3)
        self.declare_parameter('gripper_threshold', 0.5)
        self.declare_parameter('gripper_debounce_sec', 1.0)
        self.declare_parameter('gripper_open_width', 0.08)
        self.declare_parameter('gripper_speed', 0.05)
        self.declare_parameter('grasp_width', 0.03)
        self.declare_parameter('grasp_epsilon', 0.01)
        self.declare_parameter('grasp_speed', 0.03)
        self.declare_parameter('grasp_force', 30.0)
        self.declare_parameter('move_gripper_action', '/franka_gripper/move')
        self.declare_parameter('grasp_action', '/franka_gripper/grasp')

        self._instruction = str(self.get_parameter('instruction').value)

        self._bridge = CvBridge()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_time = self.get_clock().now()
        self._latest_image_lock = threading.Lock()

        # ---- async inference state ----
        self._latest_action: Optional[np.ndarray] = None
        self._latest_action_time: Optional[float] = None  # monotonic seconds
        self._action_lock = threading.Lock()
        self._inference_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name='vla-inference',
        )

        # ---- EMA smoothing ----
        self._smooth_linear: Optional[np.ndarray] = None
        self._smooth_angular: Optional[np.ndarray] = None

        # ---- last action time (for dt calculation) ----
        self._last_action_arrival: Optional[float] = None  # monotonic seconds

        # ---- gripper debounce ----
        self._last_gripper_command: Optional[bool] = None
        self._last_gripper_command_time: float = 0.0

        # ---- ROS interfaces ----
        self._twist_pub = self.create_publisher(
            TwistStamped,
            str(self.get_parameter('twist_topic').value),
            10,
        )
        self._raw_action_pub = self.create_publisher(
            Float64MultiArray, '~/raw_action', 10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self._image_cb,
            10,
        )
        self.create_subscription(
            String, '~/instruction', self._instruction_cb, 10,
        )

        self._move_client = ActionClient(
            self, Move, str(self.get_parameter('move_gripper_action').value),
        )
        self._grasp_client = ActionClient(
            self, Grasp, str(self.get_parameter('grasp_action').value),
        )

        # ---- servo start service ----
        self._servo_start_client = self.create_client(
            Trigger, '/servo_node/start_servo',
        )
        self._servo_start_timer = self.create_timer(2.0, self._start_servo)

        # ---- start inference thread ----
        self._inference_thread.start()

        # ---- control timer (non-blocking consumer) ----
        period = 1.0 / float(self.get_parameter('control_frequency').value)
        self.create_timer(period, self._tick)

        self.get_logger().info('OpenVLA Servo node started (async inference mode)')

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._inference_running = False
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        super().destroy_node()

    # ------------------------------------------------------------------
    #  Servo start
    # ------------------------------------------------------------------

    def _start_servo(self) -> None:
        self._servo_start_timer.cancel()
        if not self._servo_start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Servo start service not available, will retry ...',
            )
            self._servo_start_timer = self.create_timer(3.0, self._start_servo)
            return
        future = self._servo_start_client.call_async(Trigger.Request())
        future.add_done_callback(self._start_servo_cb)

    def _start_servo_cb(self, future) -> None:
        result = future.result()
        if result.success:
            self.get_logger().info('moveit_servo started successfully')
        else:
            self.get_logger().warn(
                f'Failed to start moveit_servo: {result.message}',
            )

    # ------------------------------------------------------------------
    #  Subscribers
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self._latest_image_lock:
            self._latest_image = np.asarray(image, dtype=np.uint8)
            self._latest_image_time = self.get_clock().now()

    def _instruction_cb(self, msg: String) -> None:
        self._instruction = msg.data.strip()
        self.get_logger().info(f'Updated OpenVLA instruction: {self._instruction}')

    # ------------------------------------------------------------------
    #  Async inference (background thread)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """Continuously run VLA inference on the latest camera frame.

        Runs in a daemon thread.  Grabs a copy of the latest image, sends it
        to the VLA server, and stores the action for the timer callback to
        consume.  If inference is slower than the image rate, only the most
        recent image is used each cycle — stale frames are skipped.
        """
        last_image_ns = 0  # track via stamp to skip already-processed frames

        while self._inference_running and rclpy.ok():
            # --- grab latest image ---
            with self._latest_image_lock:
                image = self._latest_image
                image_time = self._latest_image_time

            if image is None:
                time.sleep(0.05)
                continue

            image_ns = image_time.nanoseconds
            if image_ns == last_image_ns:
                # No new frame yet — brief sleep
                time.sleep(0.02)
                continue
            last_image_ns = image_ns

            # --- run inference ---
            try:
                action = self._request_action(image.copy())
                now_mono = time.monotonic()
                with self._action_lock:
                    self._latest_action = action
                    self._latest_action_time = now_mono
            except Exception as exc:
                self.get_logger().error(
                    f'VLA inference failed: {exc}', throttle_duration_sec=2.0,
                )
                # Brief back-off on error to avoid hot-looping
                time.sleep(0.1)

    # ------------------------------------------------------------------
    #  Control tick (non-blocking consumer — timer callback)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Read the latest completed action and publish a twist command.

        This is called by a ROS timer at ``control_frequency`` Hz.  It does
        **not** block on VLA inference — it just polls the most recent action
        from the background thread.
        """
        # --- read latest action ---
        with self._action_lock:
            action = self._latest_action
            action_time = self._latest_action_time

        if action is None:
            # No inference result yet — keep last twist (don't decay to zero)
            return

        # --- compute actual dt since last action ---
        now_mono = time.monotonic()
        if self._last_action_arrival is not None:
            dt = now_mono - self._last_action_arrival
        else:
            dt = 1.0 / float(self.get_parameter('control_frequency').value)

        # Only process if this action is newer than the last one we handled
        if action_time is not None and self._last_action_arrival is not None:
            if action_time <= self._last_action_arrival:
                # Already processed this action — keep publishing last twist
                if self._smooth_linear is not None:
                    self._publish_twist(self._smooth_linear, self._smooth_angular)
                return

        self._last_action_arrival = action_time if action_time is not None else now_mono

        # --- publish raw action for debugging ---
        self._publish_raw_action(action)

        # --- convert to twist & publish ---
        try:
            self._handle_motion(action, dt)
        except Exception as exc:
            self.get_logger().error(
                f'Motion handling failed: {exc}', throttle_duration_sec=2.0,
            )

        # --- gripper ---
        try:
            self._handle_gripper(action)
        except Exception as exc:
            self.get_logger().error(
                f'Gripper handling failed: {exc}', throttle_duration_sec=2.0,
            )

    # ------------------------------------------------------------------
    #  VLA HTTP client
    # ------------------------------------------------------------------

    def _request_action(self, image: np.ndarray) -> np.ndarray:
        import requests

        payload = {
            'image': image.tolist(),
            'instruction': self._instruction,
            'unnorm_key': str(self.get_parameter('unnorm_key').value),
        }
        response = requests.post(
            str(self.get_parameter('server_url').value),
            json=payload,
            timeout=float(self.get_parameter('request_timeout_sec').value),
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict) and 'action' in body:
            body = body['action']
        return np.asarray(body, dtype=float)

    # ------------------------------------------------------------------
    #  Motion (twist conversion + smoothing + publish)
    # ------------------------------------------------------------------

    def _handle_motion(self, action: np.ndarray, dt: float) -> None:
        limits = TwistLimits(
            max_linear_velocity=float(
                self.get_parameter('max_linear_velocity').value,
            ),
            max_angular_velocity=float(
                self.get_parameter('max_angular_velocity').value,
            ),
            max_linear_step=float(
                self.get_parameter('max_linear_step').value,
            ),
            max_angular_step=float(
                self.get_parameter('max_angular_step').value,
            ),
        )
        linear, angular = action_to_twist(action, dt, limits)

        # EMA low-pass filter with fixed alpha (smoothing)
        alpha = float(self.get_parameter('smoothing_alpha').value)
        alpha = max(0.0, min(1.0, alpha))
        if self._smooth_linear is None:
            self._smooth_linear = linear.copy()
            self._smooth_angular = angular.copy()
        else:
            self._smooth_linear = (
                alpha * linear + (1.0 - alpha) * self._smooth_linear
            )
            self._smooth_angular = (
                alpha * angular + (1.0 - alpha) * self._smooth_angular
            )

        self._publish_twist(self._smooth_linear, self._smooth_angular)

    # ------------------------------------------------------------------
    #  Gripper
    # ------------------------------------------------------------------

    def _handle_gripper(self, action: np.ndarray) -> None:
        should_close = gripper_should_close(
            action, float(self.get_parameter('gripper_threshold').value),
        )
        now = time.monotonic()
        if should_close == self._last_gripper_command:
            return
        if (
            now - self._last_gripper_command_time
            < float(self.get_parameter('gripper_debounce_sec').value)
        ):
            return

        self._last_gripper_command = should_close
        self._last_gripper_command_time = now
        if should_close:
            self._send_grasp()
        else:
            self._send_open()

    def _send_open(self) -> None:
        if not self._move_client.server_is_ready():
            self.get_logger().warn('Move gripper action server is not ready')
            return
        goal = Move.Goal()
        goal.width = float(self.get_parameter('gripper_open_width').value)
        goal.speed = float(self.get_parameter('gripper_speed').value)
        self._move_client.send_goal_async(goal)

    def _send_grasp(self) -> None:
        if not self._grasp_client.server_is_ready():
            self.get_logger().warn('Grasp action server is not ready')
            return
        goal = Grasp.Goal()
        goal.width = float(self.get_parameter('grasp_width').value)
        goal.epsilon.inner = float(self.get_parameter('grasp_epsilon').value)
        goal.epsilon.outer = float(self.get_parameter('grasp_epsilon').value)
        goal.speed = float(self.get_parameter('grasp_speed').value)
        goal.force = float(self.get_parameter('grasp_force').value)
        self._grasp_client.send_goal_async(goal)

    # ------------------------------------------------------------------
    #  Publishers
    # ------------------------------------------------------------------

    def _publish_raw_action(self, action: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = action.astype(float).tolist()
        self._raw_action_pub.publish(msg)

    def _publish_twist(
        self, linear: np.ndarray, angular: np.ndarray,
    ) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter('command_frame').value)
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self._twist_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OpenVLAServoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
