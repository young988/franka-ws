"""
Pixel → Robot: Click a pixel on the camera image to define a 3D target,
then plan and execute with MoveIt.

Orientation is fixed: TCP Z axis perpendicular to base XY plane, pointing
opposite to base Z (i.e., RPY [180°, 0, 0]).

Usage:
    python3 pixel_to_robot.py

Requires:
    - Franka MoveIt running: ros2 launch franka_fr3_moveit_config moveit.launch.py
    - RealSense running: ros2 launch realsense2_camera rs_launch.py
"""
import threading
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp, Move
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped, Pose, Quaternion
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint,
    OrientationConstraint, WorkspaceParameters, RobotState)
from shape_msgs.msg import SolidPrimitive
from cv_bridge import CvBridge
from tf2_ros import (
    Buffer, TransformException, TransformListener)
from handeye_calibration.calibration_config import (
    IntrinsicsConfig, resolve_experiment_dir, resolve_intrinsics)
from handeye_calibration.sample_collector import make_latest_image_qos


def parse_bool_parameter(value):
    """Parse ROS launch bools that may arrive as strings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ('1', 'true', 'yes', 'on')


def camera_point_from_depth_image(depth_image, u, v, fx, fy, cx, cy, depth_window):
    """Convert a depth pixel into a camera-frame point with clipped bounds."""
    height, width = depth_image.shape[:2]
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))

    win = max(int(depth_window), 1) // 2
    x0 = max(0, u - win)
    x1 = min(width, u + win + 1)
    y0 = max(0, v - win)
    y1 = min(height, v + win + 1)

    window = depth_image[y0:y1, x0:x1].astype(np.float64) / 1000.0
    valid = window[window > 0]
    if len(valid) == 0:
        raise RuntimeError('Invalid depth at ({}, {})'.format(u, v))
    depth_m = float(np.median(valid))

    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    z_cam = depth_m
    return np.array([[x_cam], [y_cam], [z_cam]], dtype=np.float64)


def default_target_quaternion():
    """Fixed TCP target orientation."""
    return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


def build_move_goal(width, speed):
    goal = Move.Goal()
    goal.width = float(width)
    goal.speed = float(speed)
    return goal


def build_grasp_goal(speed, force, epsilon_width):
    goal = Grasp.Goal()
    goal.width = 0.0
    goal.speed = float(speed)
    goal.force = float(force)
    goal.epsilon.inner = float(epsilon_width)
    goal.epsilon.outer = float(epsilon_width)
    return goal


class PixelToRobot(Node):
    """Click a pixel → plan + execute with MoveIt."""

    def __init__(self):
        super().__init__('pixel_to_robot')
        cb_group = ReentrantCallbackGroup()

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('board_type', 'chessboard')
        self.declare_parameter('experiment_dir', '')
        self.declare_parameter('intrinsics_txt',
                               'camera_intrinsics_calibrated.txt')
        self.declare_parameter('intrinsics_file', '')
        self.declare_parameter('intrinsics_source', 'camera_info')
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('effector_frame', 'fr3_link8')
        self.declare_parameter('tcp_frame', 'fr3_hand_tcp')
        self.declare_parameter('tcp_offset', '0 0 0')
        self.declare_parameter('planning_group', 'fr3_arm')
        self.declare_parameter('planning_time', 1.0)
        self.declare_parameter('trajectory_action',
                               'fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_move_action', '/franka_gripper/move')
        self.declare_parameter('gripper_grasp_action', '/franka_gripper/grasp')
        self.declare_parameter('depth_window', 5)
        self.declare_parameter('enable_auto_grasp', True)
        self.declare_parameter('pregrasp_width', 0.08)
        self.declare_parameter('grasp_speed', 0.03)
        self.declare_parameter('grasp_force', 10.0)
        self.declare_parameter('min_grasp_width', 0.005)
        self.declare_parameter('target_point_topic',
                               '/pixel_to_robot/target_point')

        board_type = self.get_parameter('board_type').value
        experiment_dir = resolve_experiment_dir(
            self.get_parameter('experiment_dir').value, board_type)

        intrinsics_source = self.get_parameter('intrinsics_source').value
        self.K = None
        self.fx = self.fy = self.cx = self.cy = None
        if str(intrinsics_source).strip().lower() != 'camera_info':
            K, intrinsics_origin = resolve_intrinsics(IntrinsicsConfig(
                source=intrinsics_source,
                experiment_dir=experiment_dir,
                board_type=board_type,
                explicit_file=self.get_parameter('intrinsics_file').value,
                fx=606.25, fy=605.65, cx=321.501, cy=242.543))
            self._set_intrinsics(K, intrinsics_origin)

        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.effector_frame = self.get_parameter('effector_frame').value
        self.tcp_frame = self.get_parameter('tcp_frame').value
        tcp_offset_parts = self.get_parameter('tcp_offset').value.split()
        self.tcp_offset = np.array(
            [float(v) for v in tcp_offset_parts], dtype=np.float64).reshape(3, 1)

        self.planning_group = self.get_parameter('planning_group').value
        self.planning_time = float(self.get_parameter('planning_time').value)
        self.trajectory_action = self.get_parameter('trajectory_action').value
        self.gripper_move_action = self.get_parameter('gripper_move_action').value
        self.gripper_grasp_action = self.get_parameter('gripper_grasp_action').value
        self.depth_window = self.get_parameter('depth_window').value
        self.target_point_topic = (
            self.get_parameter('target_point_topic').value)

        self.bridge = CvBridge()
        self._color_img = None
        self._depth_img = None
        self._color_lock = threading.Lock()
        self._depth_lock = threading.Lock()

        image_qos = make_latest_image_qos()
        self.color_sub = self.create_subscription(
            Image,
            self.get_parameter('color_topic').value,
            self._color_cb, image_qos)
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self._depth_cb, image_qos)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, 10)
        self.target_point_pub = self.create_publisher(
            PointStamped, self.target_point_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.plan_client = self.create_client(
            GetMotionPlan, 'plan_kinematic_path', callback_group=cb_group)
        if not self.plan_client.wait_for_service(10.0):
            self.get_logger().warn(
                'MoveIt planning service not available yet: plan_kinematic_path')

        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            self.trajectory_action,
            callback_group=cb_group)
        if not self.traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn(
                'Trajectory action server not available yet: {}'.format(
                    self.trajectory_action))
        self.gripper_move_client = ActionClient(
            self, Move, self.gripper_move_action, callback_group=cb_group)
        self.gripper_grasp_client = ActionClient(
            self, Grasp, self.gripper_grasp_action, callback_group=cb_group)
        if not self.gripper_move_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                'Gripper move action server not available yet: {}'.format(
                    self.gripper_move_action))
        if not self.gripper_grasp_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                'Gripper grasp action server not available yet: {}'.format(
                    self.gripper_grasp_action))

        self.target_quat = default_target_quaternion()
        self._planning = False
        self._executor = None

    def _set_intrinsics(self, K, origin):
        self.K = K
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        self.get_logger().info(
            'Loaded intrinsics from {}: fx={:.2f} fy={:.2f} '
            'cx={:.2f} cy={:.2f}'.format(
                origin, self.fx, self.fy, self.cx, self.cy))

    def _color_cb(self, msg):
        with self._color_lock:
            self._color_img = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')

    def _depth_cb(self, msg):
        with self._depth_lock:
            self._depth_img = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='16UC1')

    def _camera_info_cb(self, msg):
        if self.K is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._set_intrinsics(K, self.get_parameter('camera_info_topic').value)

    def pixel_to_camera(self, u, v):
        """Convert pixel (u,v) to camera-frame 3D point using aligned depth."""
        if self.K is None:
            raise RuntimeError('No camera intrinsics received yet')
        with self._depth_lock:
            if self._depth_img is None:
                raise RuntimeError('No depth image received yet')
            depth_img = self._depth_img.copy()
        return camera_point_from_depth_image(
            depth_img, u, v, self.fx, self.fy, self.cx, self.cy,
            self.depth_window)

    def camera_to_base_tcp(self, p_cam):
        """Transform camera-frame point to base-frame TCP target through TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))
        except TransformException as exc:
            raise RuntimeError('TF lookup failed: {}'.format(exc)) from exc

        from scipy.spatial.transform import Rotation
        q = t.transform.rotation
        R_base_to_camera = Rotation.from_quat(
            [q.x, q.y, q.z, q.w]).as_matrix()
        T_base_to_camera = np.array([
            [t.transform.translation.x],
            [t.transform.translation.y],
            [t.transform.translation.z],
        ], dtype=np.float64)
        return R_base_to_camera @ p_cam + T_base_to_camera

    def plan_and_execute_async(self, x, y, z):
        """Plan a Cartesian motion and execute on the robot."""
        if self._planning:
            self.get_logger().warn('Already planning/executing')
            return

        self._planning = True
        target_point = PointStamped()
        target_point.header.stamp = self.get_clock().now().to_msg()
        target_point.header.frame_id = self.base_frame
        target_point.point.x = float(x)
        target_point.point.y = float(y)
        target_point.point.z = float(z)
        self.target_point_pub.publish(target_point)

        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation = self.target_quat
        self.get_logger().info(
            'Planning target in {}: [{:.3f}, {:.3f}, {:.3f}]'.format(
                self.base_frame, target.position.x, target.position.y,
                target.position.z))

        plan_req = MotionPlanRequest()
        plan_req.group_name = self.planning_group
        plan_req.allowed_planning_time = self.planning_time
        plan_req.max_velocity_scaling_factor = 0.2
        plan_req.max_acceleration_scaling_factor = 0.2

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_frame
        pos_constraint.link_name = self.tcp_frame
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        constraint_region = SolidPrimitive(type=2, dimensions=[0.01, 0.01, 0.01])
        pos_constraint.constraint_region.primitives.append(constraint_region)
        region_pose = Pose()
        region_pose.position = target.position
        region_pose.orientation.w = 1.0
        pos_constraint.constraint_region.primitive_poses.append(region_pose)
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.base_frame
        ori_constraint.link_name = self.tcp_frame
        ori_constraint.orientation = self.target_quat
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 3.14
        ori_constraint.weight = 0.9

        plan_req.goal_constraints.append(
            Constraints(position_constraints=[pos_constraint],
                        orientation_constraints=[ori_constraint]))

        ws = WorkspaceParameters()
        ws.header.frame_id = self.base_frame
        ws.min_corner.x = 0.05
        ws.min_corner.y = -1.0
        ws.min_corner.z = -0.4
        ws.max_corner.x = 1.2
        ws.max_corner.y = 1.0
        ws.max_corner.z = 1.6
        plan_req.workspace_parameters = ws

        rs = RobotState()
        rs.is_diff = True
        plan_req.start_state = rs

        service_req = GetMotionPlan.Request()
        service_req.motion_plan_request = plan_req
        future = self.plan_client.call_async(service_req)
        future.add_done_callback(
            lambda f: self._on_plan_done(f, target))

    def _on_plan_done(self, future, target):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error('Plan call failed: {}'.format(e))
            self._planning = False
            return

        mpr = response.motion_plan_response
        if mpr.error_code.val != 1:  # SUCCESS
            self.get_logger().error(
                'Plan failed: {}'.format(mpr.error_code.val))
            self._planning = False
            return

        traj = mpr.trajectory
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj.joint_trajectory
        self.get_logger().info(
            'Plan succeeded — sending to trajectory controller')

        send_future = self.traj_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._on_goal_sent(f))

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected')
            self._planning = False
            return
        self.get_logger().info('Trajectory goal accepted — executing')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_execution_done(f))

    def _on_execution_done(self, future):
        result = future.result()
        code = result.result.error_code
        final_width = getattr(result.result, 'final_width', 0.0)

        if code == 0:
            self.get_logger().info(
                'Execution succeeded (final_width={:.3f})'.format(final_width))
        else:
            self.get_logger().error(
                'Execution failed: error_code={}'.format(code))

        enable_auto_grasp = parse_bool_parameter(
            self.get_parameter('enable_auto_grasp').value)
        from handeye_calibration.grasp_logic import should_start_auto_grasp

        if should_start_auto_grasp(code, enable_auto_grasp):
            self._start_auto_grasp()
            return

        self._planning = False

    def _start_auto_grasp(self):
        if not self.gripper_move_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                'Gripper move action server unavailable: {}'.format(
                    self.gripper_move_action))
            self._planning = False
            return
        pregrasp_width = float(self.get_parameter('pregrasp_width').value)
        grasp_speed = float(self.get_parameter('grasp_speed').value)
        goal = build_move_goal(pregrasp_width, grasp_speed)
        self.get_logger().info(
            'Opening gripper: width={:.3f} speed={:.3f}'.format(
                goal.width, goal.speed))
        future = self.gripper_move_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_gripper_move_sent(f))

    def _on_gripper_move_sent(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error('Gripper move goal failed: {}'.format(exc))
            self._planning = False
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Gripper move goal rejected')
            self._planning = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_gripper_move_done(f))

    def _on_gripper_move_done(self, future):
        result = future.result()
        if result is None or not result.result.success:
            error = '' if result is None else result.result.error
            self.get_logger().error('Gripper move failed: {}'.format(error))
            self._planning = False
            return
        if not self.gripper_grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                'Gripper grasp action server unavailable: {}'.format(
                    self.gripper_grasp_action))
            self._planning = False
            return
        grasp_speed = float(self.get_parameter('grasp_speed').value)
        grasp_force = float(self.get_parameter('grasp_force').value)
        min_grasp_width = float(self.get_parameter('min_grasp_width').value)
        goal = build_grasp_goal(grasp_speed, grasp_force, min_grasp_width)
        self.get_logger().info(
            'Closing gripper: force={:.1f} speed={:.3f}'.format(
                goal.force, goal.speed))
        future = self.gripper_grasp_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_gripper_grasp_sent(f))

    def _on_gripper_grasp_sent(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error('Gripper grasp goal failed: {}'.format(exc))
            self._planning = False
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Gripper grasp goal rejected')
            self._planning = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_gripper_grasp_done(f))

    def _on_gripper_grasp_done(self, future):
        result = future.result()
        success = result is not None and bool(result.result.success)
        error = '' if result is None else result.result.error
        if success:
            self.get_logger().info('Grasp success')
        else:
            self.get_logger().error('Grasp failed: {}'.format(error))
        self._planning = False


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRobot()

    print('Waiting for images ...')
    while rclpy.ok():
        with node._color_lock:
            has_color = node._color_img is not None
        with node._depth_lock:
            has_depth = node._depth_img is not None
        has_intrinsics = node.K is not None
        if has_color and has_depth and has_intrinsics:
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    print('Images and camera_info received. Click on the image to set a target point.')
    print('  Left click  → plan + execute')
    print('  ESC / q     → quit')

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    node._executor = executor
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow('pixel_to_robot', cv2.WINDOW_AUTOSIZE)

    click_pt = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal click_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt = (x, y)

    cv2.setMouseCallback('pixel_to_robot', mouse_cb)

    try:
        while rclpy.ok():
            with node._color_lock:
                img = (node._color_img.copy()
                       if node._color_img is not None else None)
            if img is None:
                continue

            h, w = img.shape[:2]
            cx, cy = int(w / 2), int(h / 2)
            cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
            cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

            if click_pt is not None and not node._planning:
                u, v = click_pt
                cv2.drawMarker(img, (u, v), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, '({}, {})'.format(u, v),
                            (u + 15, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
                click_pt = None

                try:
                    p_cam = node.pixel_to_camera(u, v)
                    p_base = node.camera_to_base_tcp(p_cam)
                    node.get_logger().info(
                        'Clicked pixel ({}, {}) -> camera {} [{:.3f}, {:.3f}, {:.3f}] '
                        '-> base {} [{:.3f}, {:.3f}, {:.3f}]'.format(
                            u, v, node.camera_frame,
                            p_cam[0, 0], p_cam[1, 0], p_cam[2, 0],
                            node.base_frame,
                            p_base[0, 0], p_base[1, 0], p_base[2, 0]))
                    cv2.putText(img,
                                'Target (base): [{:.3f}, {:.3f}, {:.3f}]'
                                .format(p_base[0, 0],
                                        p_base[1, 0],
                                        p_base[2, 0]),
                                (10, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                    node.plan_and_execute_async(
                        p_base[0, 0], p_base[1, 0], p_base[2, 0])
                except Exception as e:
                    node.get_logger().error(str(e))

            if node._planning:
                cv2.putText(img, 'PLANNING / EXECUTING ...',
                            (10, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            cv2.imshow('pixel_to_robot', img)
            key = cv2.waitKey(20) & 255
            if key == 27 or key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
