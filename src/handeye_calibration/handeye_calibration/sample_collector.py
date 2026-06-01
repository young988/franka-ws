"""
Real-time sample collector for hand-eye calibration.

On pressing 's', saves the current camera image and records the robot
end-effector pose (from TF) to a CSV file.

Usage (launch):
    ros2 launch handeye_calibration collect_samples.launch.py \
        robot_ip:=172.16.0.2

Output:
    {sample_dir}/img/       — saved images (0000.png, 0001.png, ...)
    {sample_dir}/poses.csv  — tx,ty,tz,rx,ry,rz per row
"""
import os
import csv
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException
from handeye_calibration.board_detection import detect_calibration_points
from handeye_calibration.calibration_config import (
    BoardConfig, resolve_calibration_frames, resolve_sample_dir,
    write_intrinsics)


def resolve_preview_mode(preview_mode, board_type):
    """Resolve preview_mode=auto to the default backend for the selected board."""
    mode = str(preview_mode or 'auto').strip().lower()
    board = str(board_type or '').strip().lower().replace('-', '_')
    if mode == 'auto':
        if board in ('chessboard', 'single_aruco', 'charuco', 'aruco_grid'):
            return 'opencv'
        return 'none'
    if mode in ('none', 'opencv', 'aruco_ros'):
        return mode
    return 'opencv'


def resolve_preview_detection_interval(value, display_interval):
    """Normalize preview detection interval and keep it no faster than display refresh."""
    try:
        interval = float(value)
    except (TypeError, ValueError):
        interval = 0.25
    return max(interval, float(display_interval))


def make_latest_image_qos():
    """QoS for camera preview subscribers: drop old frames instead of building latency."""
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT)


def parse_bool_parameter(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


class SampleCollector(Node):
    """Collect hand-eye calibration samples: image + robot pose on keypress."""

    def __init__(self):
        super().__init__('sample_collector')
        self.declare_parameter('sample_dir', '')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        self.declare_parameter('robot_base_frame', 'fr3_link0')
        self.declare_parameter('robot_effector_frame', 'fr3_link8')
        self.declare_parameter('calibration_setup', 'eye_in_hand')
        self.declare_parameter('tracking_base_frame', '')
        self.declare_parameter('tracking_marker_frame', '')
        self.declare_parameter('intrinsics_output_name',
                               'camera_intrinsics_camera_info.txt')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0)
        self.declare_parameter('dictionary', 'auto')
        self.declare_parameter('board_type', 'chessboard')
        self.declare_parameter('squares_x', 0)
        self.declare_parameter('squares_y', 0)
        self.declare_parameter('square_size', 0)
        self.declare_parameter('preview_mode', 'none')
        self.declare_parameter('aruco_result_topic', '')
        self.declare_parameter('display_interval', 0.05)
        self.declare_parameter('preview_detection_interval', 0.25)
        self.declare_parameter('headless', False)

        frames = resolve_calibration_frames(
            calibration_setup=self.get_parameter('calibration_setup').value,
            robot_base_frame=self.get_parameter('robot_base_frame').value,
            robot_effector_frame=self.get_parameter('robot_effector_frame').value,
            tracking_base_frame=self.get_parameter('tracking_base_frame').value,
            tracking_marker_frame=self.get_parameter(
                'tracking_marker_frame').value,
        )
        self.calibration_setup = frames.calibration_setup
        self.base_frame = frames.robot_base_frame
        self.effector_frame = frames.robot_effector_frame
        self.tracking_base_frame = frames.tracking_base_frame
        self.tracking_marker_frame = frames.tracking_marker_frame

        self.board_config = BoardConfig.from_values(
            board_type=self.get_parameter('board_type').value,
            dictionary=self.get_parameter('dictionary').value,
            marker_id=self.get_parameter('marker_id').value,
            marker_size=self.get_parameter('marker_size').value,
            squares_x=self.get_parameter('squares_x').value,
            squares_y=self.get_parameter('squares_y').value,
            square_size=self.get_parameter('square_size').value,
        )

        self.preview_mode = resolve_preview_mode(
            self.get_parameter('preview_mode').value,
            self.board_config.board_type)
        self.display_interval = max(
            float(self.get_parameter('display_interval').value), 0.01)
        self.preview_detection_interval = resolve_preview_detection_interval(
            self.get_parameter('preview_detection_interval').value,
            self.display_interval)
        self.headless = parse_bool_parameter(
            self.get_parameter('headless').value)

        sample_dir = resolve_sample_dir(
            self.get_parameter('sample_dir').value,
            self.board_config.board_type,
            calibration_setup=self.calibration_setup)
        self.sample_dir = sample_dir
        self.img_dir = os.path.join(sample_dir, 'img')
        self.pose_path = os.path.join(sample_dir, 'poses.csv')
        self.intrinsics_path = os.path.join(
            sample_dir,
            self.get_parameter('intrinsics_output_name').value)
        self.legacy_intrinsics_path = os.path.join(sample_dir,
                                                   'camera_intrinsics.txt')
        os.makedirs(self.img_dir, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_img_msg = None
        image_qos = make_latest_image_qos()
        self.sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._image_cb, image_qos)

        self.latest_aruco_img = None
        self._last_preview_detection_time = 0.0
        self._preview_points = None
        self._preview_detected = False
        aruco_result_topic = self.get_parameter('aruco_result_topic').value
        self.aruco_sub = None
        if self.preview_mode == 'aruco_ros' and aruco_result_topic:
            self.aruco_sub = self.create_subscription(
                Image, aruco_result_topic, self._aruco_image_cb, image_qos)

        self.K = None
        self.load_or_subscribe_intrinsics()

        self.counter = self._load_existing_count()

        if not self.headless:
            try:
                cv2.namedWindow('sample_collector', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('sample_collector', 960, 540)
            except cv2.error as exc:
                self.headless = True
                self.get_logger().warn(
                    'OpenCV window unavailable; continuing in headless mode: {}'
                    .format(exc))
        self.get_logger().info(
            "Sample collector ready. Press 's' to save, 'q' to quit.")
        self.get_logger().info(
            'Saving to: {} / {}'.format(self.img_dir, self.pose_path))
        self.get_logger().info('Saved pose semantics: base -> tool')
        self.get_logger().info(
            'Calibration setup: {} (tracking {} -> {})'.format(
                self.calibration_setup,
                self.tracking_base_frame,
                self.tracking_marker_frame))
        self.get_logger().info(
            'Preview mode: {} ({})'.format(self.preview_mode,
                                           self.board_config.board_type))
        self.get_logger().info(
            'Display interval: {:.3f}s, OpenCV detection interval: {:.3f}s'
            .format(self.display_interval, self.preview_detection_interval))

        self.timer = self.create_timer(self.display_interval, self._loop)

    def load_or_subscribe_intrinsics(self):
        """Load intrinsics from saved file, or subscribe to camera_info."""
        existing_path = None
        for path in (self.intrinsics_path, self.legacy_intrinsics_path):
            if os.path.exists(path):
                existing_path = path
                break

        if existing_path:
            K = np.loadtxt(existing_path)
            self.K = K.reshape(3, 3).astype(np.float64)
            self.get_logger().info(
                'Loaded intrinsics from {}: '
                'fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}'.format(
                    existing_path,
                    self.K[0, 0], self.K[1, 1],
                    self.K[0, 2], self.K[1, 2]))
            return

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, 10)
        self.get_logger().info('Waiting for camera_info...')

    def _camera_info_cb(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.k).reshape(3, 3).astype(np.float64)
        write_intrinsics(self.intrinsics_path, self.K,
                         header='RealSense CameraInfo.K saved by '
                                'sample_collector')
        if not os.path.exists(self.legacy_intrinsics_path):
            write_intrinsics(self.legacy_intrinsics_path, self.K,
                             header='Legacy copy of RealSense CameraInfo.K')
        self.get_logger().info(
            'Saved camera_info intrinsics: '
            'fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}'.format(
                self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]))

    def _load_existing_count(self):
        if os.path.exists(self.pose_path):
            with open(self.pose_path, 'r') as f:
                return sum(1 for _ in f)
        return 0

    def _image_cb(self, msg):
        self.latest_img_msg = msg
        self.latest_img = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

    def _aruco_image_cb(self, msg):
        self.latest_aruco_img = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

    def _draw_preview_points(self, display):
        if self._preview_points is None:
            return
        points = self._preview_points
        for point in points:
            cv2.circle(display, tuple(point), 4, (0, 255, 0), -1)
        if self.board_config.board_type == 'chessboard':
            pattern = (self.board_config.squares_x,
                       self.board_config.squares_y)
            cv2.drawChessboardCorners(
                display, pattern,
                points.astype(np.float32).reshape(-1, 1, 2), True)
            return
        if len(points) >= 4:
            for idx in range(0, len(points) - 3, 4):
                cv2.polylines(
                    display,
                    [points[idx:idx + 4].reshape(-1, 1, 2)],
                    True, (0, 255, 0), 2)

    def _opencv_preview(self, image, run_detection=True):
        display = image.copy()
        if run_detection:
            try:
                obj_pts, img_pts = detect_calibration_points(
                    display, self.board_config)
            except (RuntimeError, ValueError, cv2.error) as exc:
                self.get_logger().debug(
                    'Preview detection failed: {}'.format(exc))
                obj_pts, img_pts = None, None

        self._preview_detected = (obj_pts is not None
                                  and img_pts is not None)
        self._preview_points = (
            np.asarray(img_pts, dtype=np.int32).reshape(-1, 2)
            if self._preview_detected else None)

        self._draw_preview_points(display)
        return (display, self._preview_detected)

    def _get_robot_pose(self):
        """Look up tracked transform from TF, return (tx,ty,tz,rx,ry,rz)."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.tracking_base_frame,
                self.tracking_marker_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0))
        except LookupException as e:
            self.get_logger().warn('TF lookup failed: {}'.format(e))
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        tz = t.transform.translation.z
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w

        from scipy.spatial.transform import Rotation
        base_to_tool_rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        base_to_tool_translation = np.array([[tx], [ty], [tz]], dtype=np.float64)

        rpy = Rotation.from_matrix(base_to_tool_rotation).as_euler(
            'xyz', degrees=True)

        return (float(base_to_tool_translation[0, 0]),
                float(base_to_tool_translation[1, 0]),
                float(base_to_tool_translation[2, 0]),
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]))

    def _save_sample(self):
        if self.latest_img is None:
            self.get_logger().warn('No image received yet')
            return

        pose = self._get_robot_pose()
        if pose is None:
            return

        img_name = '{:04d}.png'.format(self.counter)
        img_path = os.path.join(self.img_dir, img_name)
        cv2.imwrite(img_path, self.latest_img)

        with open(self.pose_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(pose)

        self.get_logger().info(
            'Sample {} saved | pose: tx={:.4f} ty={:.4f} tz={:.4f} '
            'rx={:.2f} ry={:.2f} rz={:.2f}'.format(self.counter, *pose))
        self.counter += 1

    def _loop(self):
        if self.latest_img is None:
            return

        if self.headless:
            return

        detected = False
        if (self.preview_mode == 'aruco_ros'
                and self.latest_aruco_img is not None):
            display = self.latest_aruco_img.copy()
            detected = True
        elif self.preview_mode == 'opencv':
            now = time.monotonic()
            run_detection = (now - self._last_preview_detection_time
                             >= self.preview_detection_interval)
            if run_detection:
                self._last_preview_detection_time = now
            display, detected = self._opencv_preview(
                self.latest_img, run_detection)
        else:
            display = self.latest_img.copy()

        has_intrinsics = self.K is not None
        if self.preview_mode == 'none':
            status = 'Preview: raw'
        else:
            status = 'Preview: {} {}'.format(
                self.preview_mode, 'OK' if detected else '--')
        status += ' | K: OK' if has_intrinsics else ' | K: --'

        cv2.putText(display,
                    'Samples: {} | {}'.format(self.counter, status),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, '[s] save sample  [q] quit',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        cv2.imshow('sample_collector', display)
        key = cv2.waitKey(1) & 255
        if key == ord('s'):
            self._save_sample()
            return
        if key == ord('q'):
            self.get_logger().info(
                'Quit — collected {} samples'.format(self.counter))
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return


def main(args=None):
    rclpy.init(args=args)
    node = SampleCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
