"""OpenVLA → MoveIt Planning bridge (non-servo).

Replaces the servo-based control loop with discrete planning:

1. Background thread runs continuous VLA inference on the latest camera image.
2. The control tick grabs the latest action, looks up the current TCP pose from TF,
   clips the delta, computes an absolute target pose (delta is in world/base frame,
   so position is added directly and orientation uses Euler-angle addition), and
   sends it to MoveIt for IK + motion planning + trajectory execution.
3. No manual smoothing — the trajectory executor handles timing.

Important: OpenVLA bridge_orig outputs delta actions in the **world/base frame**
(fr3_link0), not the TCP frame.  This is because relabel_bridge_actions()
computes actions as:
    movement_actions = state[1:, :6] - state[:-1, :6]
where state = [x, y, z, roll, pitch, yaw] in the robot's base frame.

Target pose computation:
    target.pos  = current.pos + [dx, dy, dz]          (world-frame addition)
    target.rpy  = current.rpy + [droll, dpitch, dyaw]  (Euler-angle addition)
"""

import json
import threading
import time
from typing import Optional

from cv_bridge import CvBridge
import numpy as np
import rclpy
from franka_msgs.action import Grasp, Move
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener

from franka_deploy.action_mapping import (
    DeltaLimits,
    clip_delta,
    compute_target_pose,
    gripper_should_close,
    pose_distance,
    validate_action,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _build_position_constraint(
    target: Pose,
    frame_id: str,
    link_name: str,
    tolerance_m: float = 0.01,
) -> PositionConstraint:
    """Build a box-shaped position constraint centered at *target*."""
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [tolerance_m * 2.0] * 3

    region = BoundingVolume()
    region.primitives.append(box)
    region.primitive_poses.append(target)

    pc = PositionConstraint()
    pc.header.frame_id = frame_id
    pc.link_name = link_name
    pc.constraint_region = region
    pc.weight = 1.0
    return pc


def _build_orientation_constraint(
    target: Pose,
    frame_id: str,
    link_name: str,
    tolerance_rad: float = 0.1,
) -> OrientationConstraint:
    """Build an orientation constraint matching *target.orientation*."""
    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation = target.orientation
    oc.absolute_x_axis_tolerance = tolerance_rad
    oc.absolute_y_axis_tolerance = tolerance_rad
    oc.absolute_z_axis_tolerance = tolerance_rad
    oc.weight = 1.0
    return oc


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class OpenVLAPlannerNode(Node):
    """OpenVLA → MoveIt discrete planner bridge.

    Synchronisation
    ---------------
    The VLA model expects the robot to be **at rest** when each image is captured.
    We enforce this via a simple state machine::

        settled ──[VLA action]──> planning ──[exec done]──> settled
           ▲                                                    │
           └────────────────────────────────────────────────────┘

    - **settled**: inference thread runs freely on the latest camera image.
    - **planning**: inference thread is **blocked** (paused); all images
      captured during motion are ignored.
    - When planning completes, any stale VLA action is discarded, and
      inference resumes with the next fresh image (robot at rest).

    Important: OpenVLA bridge_orig outputs delta actions in the **world/base
    frame**, not the TCP frame.  [dx,dy,dz] and [droll,dpitch,dyaw] are in
    the world/base frame.  Position deltas are added directly; orientation
    deltas are Euler-angle differences added to the current RPY parameters.
    """

    STATE_SETTLED = 'settled'
    STATE_PLANNING = 'planning'

    def __init__(self):
        super().__init__('openvla_planner')

        # ---- parameters ----
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/act')
        self.declare_parameter('instruction', 'move the object')
        self.declare_parameter('unnorm_key', 'bridge_orig')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('command_frame', 'fr3_link0')
        self.declare_parameter('tcp_frame', 'fr3_hand_tcp')
        self.declare_parameter('move_group_name', 'fr3_arm')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('request_timeout_sec', 10.0)
        self.declare_parameter('settle_delay_sec', 0.3)

        # safety / delta limits
        self.declare_parameter('max_linear_step', 0.05)
        self.declare_parameter('max_angular_step', 0.25)
        self.declare_parameter('max_workspace_radius', 0.85)

        # planning
        self.declare_parameter('planning_time', 2.0)
        self.declare_parameter('planning_attempts', 1)
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('acceleration_scaling', 0.3)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('orientation_tolerance', 0.1)

        # gripper
        self.declare_parameter('gripper_threshold', 0.0)
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

        # ---- CV bridge ----
        self._bridge = CvBridge()

        # ---- TF ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- state machine ----
        self._state = self.STATE_SETTLED
        self._state_lock = threading.Lock()
        self._plan_complete_time_ns: int = 0  # ROS time (ns) when last plan finished

        # ---- VLA inference state ----
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_time_ns: int = 0
        self._image_lock = threading.Lock()

        self._latest_action: Optional[np.ndarray] = None
        self._latest_action_time: Optional[float] = None
        self._action_lock = threading.Lock()

        self._inference_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name='vla-inference',
        )

        # ---- planning state ----
        self._last_target: Optional[Pose] = None

        # ---- gripper state ----
        self._last_gripper_command: Optional[bool] = None
        self._last_gripper_command_time: float = 0.0

        # ---- ROS interfaces ----
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

        # MoveIt MoveGroup action
        self._move_action_client = ActionClient(
            self, MoveGroup, '/move_action',
        )

        # Gripper actions
        self._move_client = ActionClient(
            self, Move, str(self.get_parameter('move_gripper_action').value),
        )
        self._grasp_client = ActionClient(
            self, Grasp, str(self.get_parameter('grasp_action').value),
        )

        # ---- start ----
        self._inference_thread.start()

        period = 1.0 / float(self.get_parameter('control_frequency').value)
        self.create_timer(period, self._tick)

        self.get_logger().info('OpenVLA Planner node started (settled → planning → settled)')

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._inference_running = False
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        super().destroy_node()

    # ------------------------------------------------------------------
    #  Subscribers
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}', throttle_duration_sec=5.0)
            return
        with self._image_lock:
            self._latest_image = np.asarray(image, dtype=np.uint8)
            self._latest_image_time_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _instruction_cb(self, msg: String) -> None:
        self._instruction = msg.data.strip()
        self.get_logger().info(f'Updated instruction: {self._instruction}')

    # ------------------------------------------------------------------
    #  TF helper
    # ------------------------------------------------------------------

    def _get_current_tcp_pose(self) -> Optional[Pose]:
        """Return the current TCP pose in *command_frame*, or None."""
        base = str(self.get_parameter('command_frame').value)
        tcp = str(self.get_parameter('tcp_frame').value)
        try:
            t = self._tf_buffer.lookup_transform(base, tcp, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(
                f'TF lookup {base}→{tcp} failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation
        return pose

    # ------------------------------------------------------------------
    #  Async inference (background thread)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """Run VLA inference only when robot is settled.

        While *planning*, the thread blocks — images captured mid-motion
        are stale and would produce actions inconsistent with the robot's
        true final state.
        """
        last_image_ns = 0
        while self._inference_running and rclpy.ok():
            # --- block while robot is in motion ---
            with self._state_lock:
                state = self._state
            if state == self.STATE_PLANNING:
                time.sleep(0.05)
                continue

            # --- grab latest image ---
            with self._image_lock:
                image = self._latest_image
                image_ns = self._latest_image_time_ns

            if image is None:
                time.sleep(0.05)
                continue

            # Only process images taken AFTER the last plan completed
            # (ensures robot is at rest in the image).
            if image_ns <= self._plan_complete_time_ns:
                time.sleep(0.02)
                continue

            if image_ns == last_image_ns:
                time.sleep(0.02)
                continue
            last_image_ns = image_ns

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
                time.sleep(0.1)

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
    #  Control tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Consume the latest VLA action and dispatch a planning request."""
        # --- don't overlap plans ---
        with self._state_lock:
            if self._state == self.STATE_PLANNING:
                return

        # --- read latest action ---
        with self._action_lock:
            action = self._latest_action
            action_time = self._latest_action_time

        if action is None:
            return

        # Mark this action as consumed
        with self._action_lock:
            self._latest_action = None

        # --- publish raw action ---
        self._publish_raw_action(action)

        # --- validate ---
        try:
            validate_action(action)
        except ValueError as exc:
            self.get_logger().warn(f'Invalid action: {exc}')
            return

        # --- get current TCP pose ---
        current_pose = self._get_current_tcp_pose()
        if current_pose is None:
            return

        # --- log raw action ---
        self.get_logger().info(
            f'VLA raw: pos=[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}] '
            f'rpy=[{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}] '
            f'grip={action[6]:.4f}'
        )

        # --- clip delta ---
        limits = DeltaLimits(
            max_linear_step=float(self.get_parameter('max_linear_step').value),
            max_angular_step=float(self.get_parameter('max_angular_step').value),
        )
        clipped = clip_delta(action, limits)

        # --- log current pose ---
        self.get_logger().info(
            f'TCP current: pos=[{current_pose.position.x:.4f}, '
            f'{current_pose.position.y:.4f}, {current_pose.position.z:.4f}]'
        )

        # --- compute target pose ---
        target = compute_target_pose(current_pose, clipped)

        # --- workspace safety ---
        max_radius = float(self.get_parameter('max_workspace_radius').value)
        dist = np.linalg.norm([
            target.position.x, target.position.y, target.position.z,
        ])
        if dist > max_radius:
            self.get_logger().warn(
                f'Target pose {dist:.3f}m exceeds workspace radius '
                f'{max_radius:.3f}m — skipping',
                throttle_duration_sec=1.0,
            )
            return

        # --- deduplicate target (avoid re-planning the same pose) ---
        if self._last_target is not None and pose_distance(target, self._last_target) < 0.002:
            return

        # --- plan + execute ---
        self._last_target = target
        with self._state_lock:
            self._state = self.STATE_PLANNING
        self._send_planning_goal(target)

        # --- gripper ---
        try:
            self._handle_gripper(action)
        except Exception as exc:
            self.get_logger().error(
                f'Gripper handling failed: {exc}', throttle_duration_sec=2.0,
            )

    # ------------------------------------------------------------------
    #  Planning via MoveGroup action
    # ------------------------------------------------------------------

    def _send_planning_goal(self, target: Pose) -> None:
        """Build a MotionPlanRequest and send it to MoveGroup."""
        if not self._move_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('MoveGroup action server not ready')
            self._transition_to_settled()
            return

        frame_id = str(self.get_parameter('command_frame').value)
        tcp_frame = str(self.get_parameter('tcp_frame').value)
        group_name = str(self.get_parameter('move_group_name').value)

        # --- constraints ---
        pc = _build_position_constraint(
            target, frame_id, tcp_frame,
            tolerance_m=float(self.get_parameter('position_tolerance').value),
        )
        oc = _build_orientation_constraint(
            target, frame_id, tcp_frame,
            tolerance_rad=float(self.get_parameter('orientation_tolerance').value),
        )
        constraints = Constraints()
        constraints.name = 'vla_target'
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)

        # --- workspace ---
        ws = WorkspaceParameters()
        ws.header.frame_id = frame_id
        ws.min_corner.x = -1.0
        ws.min_corner.y = -1.0
        ws.min_corner.z = -0.2
        ws.max_corner.x = 1.0
        ws.max_corner.y = 1.0
        ws.max_corner.z = 1.5

        # --- request ---
        request = MotionPlanRequest()
        request.workspace_parameters = ws
        request.group_name = group_name
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = int(self.get_parameter('planning_attempts').value)
        request.allowed_planning_time = float(self.get_parameter('planning_time').value)
        request.max_velocity_scaling_factor = float(self.get_parameter('velocity_scaling').value)
        request.max_acceleration_scaling_factor = float(self.get_parameter('acceleration_scaling').value)
        request.pipeline_id = 'move_group'
        request.planner_id = ''

        # --- planning options ---
        options = PlanningOptions()
        options.plan_only = False
        options.look_around = False
        options.replan = True
        options.replan_delay = 0.5
        options.replan_attempts = 1

        # --- goal ---
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = options

        self.get_logger().info(
            f'Sending plan goal: pos=[{target.position.x:.3f}, '
            f'{target.position.y:.3f}, {target.position.z:.3f}]'
        )
        send_goal_future = self._move_action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Planning goal rejected')
            self._transition_to_settled()
            return

        self.get_logger().info('Planning goal accepted — executing')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result()
        code = result.result.error_code
        if code.val == 1:  # SUCCESS
            self.get_logger().info('Motion plan executed successfully')
        else:
            self.get_logger().warn(
                f'Motion plan failed with error code {code.val}'
            )
        self._transition_to_settled()

    def _transition_to_settled(self) -> None:
        """Mark the robot as settled so inference can resume.

        Records the current time so the inference thread only processes
        images captured *after* this moment (robot at rest).
        """
        now = self.get_clock().now()
        settle_delay_ns = int(
            float(self.get_parameter('settle_delay_sec').value) * 1e9
        )
        self._plan_complete_time_ns = (
            now.seconds_nanoseconds()[0] * 1_000_000_000
            + now.seconds_nanoseconds()[1]
            + settle_delay_ns
        )
        self.get_logger().debug(
            f'Settled at t={self._plan_complete_time_ns / 1e9:.3f}s'
        )
        with self._state_lock:
            self._state = self.STATE_SETTLED

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
            self.get_logger().warn('Move gripper action server not ready',
                                   throttle_duration_sec=3.0)
            return
        goal = Move.Goal()
        goal.width = float(self.get_parameter('gripper_open_width').value)
        goal.speed = float(self.get_parameter('gripper_speed').value)
        self._move_client.send_goal_async(goal)

    def _send_grasp(self) -> None:
        if not self._grasp_client.server_is_ready():
            self.get_logger().warn('Grasp action server not ready',
                                   throttle_duration_sec=3.0)
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


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = OpenVLAPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
