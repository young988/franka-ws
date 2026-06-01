"""Publish the hand-eye calibration result as a static TF.

eye_in_hand:
  1. Camera driver TF:   camera_link -> optical_frame   (looked up from TF)
  2. Calibration CSV:    optical_frame -> link8         (from handeye_results.csv)
  Published:  fr3_link8 -> camera_link

eye_to_hand:
  1. Camera driver TF:   camera_link -> optical_frame   (looked up from TF)
  2. Calibration CSV:    optical_frame -> base          (from handeye_results.csv)
  Published:  fr3_link0 -> camera_link

Composition (same formula for both modes):
  parent -> child = (CSV: optical -> parent) @ (TF: child -> optical)
"""

import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from handeye_calibration.calibration_config import (
    normalize_calibration_setup, sample_paths)


# ---------------------------------------------------------------------------
#  Math helpers
# ---------------------------------------------------------------------------

def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[0, 1] = 2.0 * (qx * qy - qz * qw)
    R[0, 2] = 2.0 * (qx * qz + qy * qw)
    R[1, 0] = 2.0 * (qx * qy + qz * qw)
    R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[1, 2] = 2.0 * (qy * qz - qx * qw)
    R[2, 0] = 2.0 * (qx * qz - qy * qw)
    R[2, 1] = 2.0 * (qy * qz + qx * qw)
    R[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return R


def _matrix_from_tf(msg: TransformStamped) -> np.ndarray:
    """Convert a TransformStamped to a 4x4 homogeneous transform."""
    q = msg.transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = msg.transform.translation.x
    T[1, 3] = msg.transform.translation.y
    T[2, 3] = msg.transform.translation.z
    T[:3, :3] = _quat_to_matrix(q.x, q.y, q.z, q.w)
    return T


def _tf_from_matrix(matrix: np.ndarray, stamp, parent_frame: str,
                    child_frame: str) -> TransformStamped:
    """Build a TransformStamped from a 4x4 homogeneous matrix."""
    R = np.asarray(matrix[:3, :3], dtype=np.float64)
    t = np.asarray(matrix[:3, 3], dtype=np.float64)

    # Rotation matrix -> quaternion  [x, y, z, w]
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
        qw = 0.25 * s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
            qw = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
            qw = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            qw = (R[1, 0] - R[0, 1]) / s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)

    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(t[0])
    msg.transform.translation.y = float(t[1])
    msg.transform.translation.z = float(t[2])
    msg.transform.rotation.x = float(quat[0])
    msg.transform.rotation.y = float(quat[1])
    msg.transform.rotation.z = float(quat[2])
    msg.transform.rotation.w = float(quat[3])
    return msg


# ---------------------------------------------------------------------------
#  CSV reader
# ---------------------------------------------------------------------------

def _read_calibration(result_file: str, method: str = 'best'):
    """Read one row from handeye_results.csv as a 4x4 homogeneous matrix.

    eye_in_hand:  optical_frame → link8   (camera optical in robot effector frame)
    eye_to_hand:  optical_frame → base    (camera optical in robot base frame)

    Returns (method_name, 4x4_matrix).
    """
    with open(result_file, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f'No calibration data in {result_file}')

    wanted = str(method or 'best').strip().lower()
    row = rows[0]  # default: first row
    if wanted not in ('', 'best', 'auto'):
        for r in rows:
            if r.get('method', '').strip().lower() == wanted:
                row = r
                break
        else:
            raise RuntimeError(
                f"Method '{method}' not found in {result_file}")

    values = [float(row[f'm{r}{c}']) for r in range(4) for c in range(4)]
    return row.get('method', 'unknown'), np.array(values, dtype=np.float64).reshape(4, 4)





# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class HandeyeTfPublisher(Node):
    """Publish the hand-eye calibration result as a static TF.

    Composition (same for both modes):
      parent -> child = calibration_matrix @ (child -> optical)

    eye_in_hand:  parent=fr3_link8, child=camera_link
    eye_to_hand:  parent=fr3_link0, child=camera_link
    """

    def __init__(self):
        super().__init__('handeye_tf_publisher')

        # --- parameters ---
        self.declare_parameter('result_file', '')
        self.declare_parameter('sample_dir', '')
        self.declare_parameter('board_type', 'chessboard')
        self.declare_parameter('calibration_setup', 'eye_in_hand')
        self.declare_parameter('method', 'best')
        self.declare_parameter('parent_frame', '')
        self.declare_parameter('child_frame', 'camera_link')
        self.declare_parameter('optical_frame', 'camera_color_optical_frame')

        calibration_setup = normalize_calibration_setup(
            self.get_parameter('calibration_setup').value)

        # --- auto-default parent_frame ---
        parent_frame = self.get_parameter('parent_frame').value
        if not parent_frame:
            if calibration_setup == 'eye_to_hand':
                parent_frame = 'fr3_link0'
            else:
                parent_frame = 'fr3_link8'
        self._parent_frame = parent_frame
        self._child_frame = self.get_parameter('child_frame').value
        self._optical_frame = self.get_parameter('optical_frame').value

        # --- resolve result file ---
        result_file = self.get_parameter('result_file').value
        if not result_file:
            paths = sample_paths(
                sample_dir=self.get_parameter('sample_dir').value,
                board_type=self.get_parameter('board_type').value,
                calibration_setup=calibration_setup)
            result_file = os.path.join(paths.output_dir, 'handeye_results.csv')

        if not os.path.exists(result_file):
            raise FileNotFoundError(f'Hand-eye result file not found: {result_file}')

        method = self.get_parameter('method').value
        self._method_name, self._calibration_matrix = _read_calibration(result_file, method)

        self.get_logger().info(
            f'Loaded hand-eye result: {result_file}  method={self._method_name}')
        self.get_logger().info(
            f'Calibration setup: {calibration_setup}, '
            f'publishing {self._parent_frame} -> {self._child_frame}')

        # --- TF infrastructure ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._broadcaster = StaticTransformBroadcaster(self)

        # Try every second until the camera driver TF appears
        self._retry_timer = self.create_timer(1.0, self._try_publish)
        self._published = False

    # ------------------------------------------------------------------
    def _try_publish(self):
        """Look up child -> optical from camera driver TF, compose, publish."""
        if self._published:
            return

        if self._child_frame == self._optical_frame:
            cam_to_optical = np.eye(4, dtype=np.float64)
        else:
            try:
                cam_to_optical_msg = self._tf_buffer.lookup_transform(
                    self._optical_frame,          # target
                    self._child_frame,            # source  →  child -> optical
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2))
            except Exception as exc:
                self.get_logger().info(
                    f'Waiting for TF {self._child_frame} -> {self._optical_frame} '
                    f'from camera driver … ({exc})', throttle_duration_sec=5)
                return

            # child -> optical
            cam_to_optical = _matrix_from_tf(cam_to_optical_msg)

        # parent -> child = (optical -> parent) @ (child -> optical)
        #   eye_in_hand:  link8 -> camera_link
        #   eye_to_hand:  base  -> camera_link
        parent_to_child = self._calibration_matrix @ cam_to_optical

        stamp = self.get_clock().now().to_msg()

        self._broadcaster.sendTransform(_tf_from_matrix(
            parent_to_child, stamp,
            self._parent_frame, self._child_frame))

        self._retry_timer.cancel()
        self._published = True

        self.get_logger().info(
            f'Published: {self._parent_frame} -> {self._child_frame}')


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = HandeyeTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
