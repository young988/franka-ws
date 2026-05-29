# Source Generated with Decompyle++
# File: aruco_camera_calibrator.cpython-310.pyc (Python 3.10)

'''
Offline camera intrinsic calibration using configurable boards.

Usage:
  ros2 run handeye_calibration aruco_camera_calibrator --ros-args -p board_type:=chessboard

Output:
  camera_intrinsics_calibrated.txt
  camera_intrinsics_comparison.txt
'''
import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from handeye_calibration.board_detection import detect_calibration_points
from handeye_calibration.calibration_config import BoardConfig, sample_paths, write_intrinsics

def resolve_calibrator_image_dir(image_dir, board_type, sample_dir = ('',)):
    paths = sample_paths(sample_dir=sample_dir, board_type=board_type, image_dir=image_dir)
    return paths.image_dir


def camera_reprojection_errors(obj_points_list, img_points_list, rvecs, tvecs, K, dist):
    per_image_errors = []
    for i in range(len(obj_points_list)):
        (proj, _) = cv2.projectPoints(obj_points_list[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_points_list[i], axis=1)
        per_image_errors.append((float(np.mean(err)), float(np.max(err))))
    return per_image_errors


def calibrate_camera_with_outlier_rejection(obj_points_list, img_points_list,
                                            captured_names, img_size,
                                            threshold_px=0.0,
                                            max_iterations=0,
                                            min_frames=5,
                                            logger=None):
    threshold_px = float(threshold_px or 0.0)
    max_iterations = int(max_iterations or 0)
    active = list(range(len(obj_points_list)))
    removed = []

    for iteration in range(max_iterations + 1):
        active_obj = [obj_points_list[i] for i in active]
        active_img = [img_points_list[i] for i in active]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            active_obj, active_img, img_size, None, None)
        errors = camera_reprojection_errors(active_obj, active_img, rvecs, tvecs, K, dist)

        if threshold_px <= 0.0 or iteration >= max_iterations:
            return ret, K, dist, rvecs, tvecs, active, removed, errors

        worst_local_idx, worst_error = max(
            enumerate(errors), key=lambda item: item[1][0])
        if worst_error[0] <= threshold_px or len(active) <= min_frames:
            return ret, K, dist, rvecs, tvecs, active, removed, errors

        removed_idx = active.pop(worst_local_idx)
        removed.append((removed_idx, worst_error[0], worst_error[1]))
        if logger is not None:
            logger.warn(
                'Reject intrinsic outlier {}: mean={:.3f}px max={:.3f}px'
                .format(captured_names[removed_idx], worst_error[0], worst_error[1]))


class ArucoCameraCalibrator(Node):
    
    def __init__(self = None):
        super().__init__('aruco_camera_calibrator')
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0)
        self.declare_parameter('dictionary', 'auto')
        self.declare_parameter('board_type', 'chessboard')
        self.declare_parameter('squares_x', 0)
        self.declare_parameter('squares_y', 0)
        self.declare_parameter('square_size', 0)
        self.declare_parameter('sample_dir', '')
        self.declare_parameter('image_dir', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('fx', 606.25)
        self.declare_parameter('fy', 605.65)
        self.declare_parameter('cx', 321.501)
        self.declare_parameter('cy', 242.543)
        self.declare_parameter('intrinsics_outlier_threshold_px', 0.0)
        self.declare_parameter('intrinsics_outlier_iterations', 0)
        self.board_config = BoardConfig.from_values(board_type=self.get_parameter('board_type').value, dictionary=self.get_parameter('dictionary').value, marker_id=self.get_parameter('marker_id').value, marker_size=self.get_parameter('marker_size').value, squares_x=self.get_parameter('squares_x').value, squares_y=self.get_parameter('squares_y').value, square_size=self.get_parameter('square_size').value)
        paths = sample_paths(sample_dir=self.get_parameter('sample_dir').value, board_type=self.board_config.board_type, image_dir=self.get_parameter('image_dir').value, output_dir=self.get_parameter('output_dir').value)
        self.image_dir = paths.image_dir
        self.output_dir = paths.output_dir
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        existing = os.path.join(self.output_dir, 'camera_intrinsics.txt')
        if os.path.exists(existing):
            K = np.loadtxt(existing)
            if K.ndim == 1:
                K = K.reshape(3, 3)
            self.fx = float(K[(0, 0)])
            self.fy = float(K[(1, 1)])
            self.cx = float(K[(0, 2)])
            self.cy = float(K[(1, 2)])
            self.get_logger().info('Loaded intrinsics: fx={:.2f} fy={:.2f}'.format(self.fx, self.fy))
        self.obj_points_list = []
        self.img_points_list = []
        self.captured_names = []
        self.img_size = None
        self.captured_count = 0
        self.get_logger().info('Offline mode: reading from {}'.format(self.image_dir))
        self._image_list = []
        for root, _dirs, files in os.walk(self.image_dir):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    self._image_list.append(os.path.join(root, f))
        self.get_logger().info('Found {} images'.format(len(self._image_list)))
        self._offline_idx = 0
        self._offline_timer = self.create_timer(0.3, self._offline_step)

    
    def _detect(self, image_path):
        '''Return (obj_pts, img_pts) or (None, None) for the configured board.'''
        image = cv2.imread(image_path)
        if image is None:
            return (None, None)
        return detect_calibration_points(image, self.board_config)

    
    def _offline_step(self):
        if self._offline_idx >= len(self._image_list):
            self._offline_timer.cancel()
            self.get_logger().info('Offline collection done ({} images), running calibration ...'.format(self.captured_count))
            if self.captured_count >= 5:
                self.calibrate()
            else:
                self.get_logger().error('Not enough detections')
            self.get_logger().info('Exiting')
            raise SystemExit(0)
        path = self._image_list[self._offline_idx]
        self._offline_idx += 1
        (obj_pts, img_pts) = self._detect(path)
        if obj_pts is None:
            self.get_logger().warn('[{}/{}] Skip {}: marker not found'.format(self._offline_idx, len(self._image_list), os.path.basename(path)))
            return None
        self.obj_points_list.append(obj_pts)
        self.img_points_list.append(img_pts)
        self.captured_names.append(os.path.basename(path))
        self.img_size = (cv2.imread(path).shape[1], cv2.imread(path).shape[0])
        self.captured_count += 1
        self.get_logger().info('[{}/{}] OK: {}'.format(self._offline_idx, len(self._image_list), os.path.basename(path)))

    
    def calibrate(self):
        if len(self.obj_points_list) < 5:
            self.get_logger().error('Need at least 5 captures, got {}'.format(len(self.obj_points_list)))
            return None
        self.get_logger().info('Running calibration with {} frames ...'.format(len(self.obj_points_list)))
        threshold_px = self.get_parameter('intrinsics_outlier_threshold_px').value
        outlier_iterations = self.get_parameter('intrinsics_outlier_iterations').value
        (ret, K, dist, rvecs, tvecs, active_indices, removed, active_errors) = (
            calibrate_camera_with_outlier_rejection(
                self.obj_points_list,
                self.img_points_list,
                self.captured_names,
                self.img_size,
                threshold_px=threshold_px,
                max_iterations=outlier_iterations,
                logger=self.get_logger()))
        fx = K[(0, 0)]
        fy = K[(1, 1)]
        cx = K[(0, 2)]
        cy = K[(1, 2)]
        per_image_errors = [
            (self.captured_names[active_indices[i]], mean_e, max_e)
            for i, (mean_e, max_e) in enumerate(active_errors)]
        self.get_logger().info('')
        self.get_logger().info('Per-image reprojection error (px):')
        self.get_logger().info('  {:<20s} {:>8s} {:>8s}'.format('image', 'mean', 'max'))
        for name, mean_e, max_e in per_image_errors:
            self.get_logger().info('  {:<20s} {:>8.3f} {:>8.3f}'.format(name, mean_e, max_e))
        if removed:
            self.get_logger().info('Rejected {} intrinsic outlier frame(s)'.format(len(removed)))
        self.get_logger().info('')
        self.get_logger().info('Overall RMS reprojection error: {:.3f} px'.format(ret))
        self.get_logger().info('fx={:.4f}  fy={:.4f}  cx={:.4f}  cy={:.4f}'.format(fx, fy, cx, cy))
        if dist is not None:
            self.get_logger().info('dist: {}'.format(dist.ravel()))
        calib_path = os.path.join(self.output_dir, 'camera_intrinsics_calibrated.txt')
        write_intrinsics(calib_path, K, header='{} calibrated intrinsics'.format(self.board_config.board_type))
        self.get_logger().info('Saved to {}'.format(calib_path))
        existing_path = os.path.join(self.output_dir, 'camera_intrinsics.txt')
        if os.path.exists(existing_path):
            K_old = np.loadtxt(existing_path)
            if K_old.ndim == 1:
                K_old = K_old.reshape(3, 3)
            fx0 = K_old[(0, 0)]
            fy0 = K_old[(1, 1)]
            cx0 = K_old[(0, 2)]
            cy0 = K_old[(1, 2)]
            self.get_logger().info('')
            self.get_logger().info('  old: fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}'.format(fx0, fy0, cx0, cy0))
            self.get_logger().info('  new: fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}'.format(fx, fy, cx, cy))
            self.get_logger().info('  diff: Δfx={:+.2f} Δfy={:+.2f} Δcx={:+.2f} Δcy={:+.2f}'.format(fx - fx0, fy - fy0, cx - cx0, cy - cy0))
            comp_path = os.path.join(self.output_dir, 'camera_intrinsics_comparison.txt')
            with open(comp_path, 'w') as f:
                f.write('Calibration comparison\n')
                f.write('======================\n')
                f.write('RMS reprojection error: {:.4f} px\n\n'.format(ret))
                f.write('{:<10} {:>10} {:>10} {:>10} {:>10}\n'.format('', 'fx', 'fy', 'cx', 'cy'))
                f.write('{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n'.format('old', fx0, fy0, cx0, cy0))
                f.write('{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n'.format('new', fx, fy, cx, cy))
                f.write('{:<10} {:>+10.4f} {:>+10.4f} {:>+10.4f} {:>+10.4f}\n'.format('diff', fx - fx0, fy - fy0, cx - cx0, cy - cy0))
            self.get_logger().info('Saved comparison to {}'.format(comp_path))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoCameraCalibrator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
