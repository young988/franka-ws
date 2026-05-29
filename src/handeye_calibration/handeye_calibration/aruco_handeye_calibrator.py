# Source Generated with Decompyle++
# File: aruco_handeye_calibrator.cpython-310.pyc (Python 3.10)

'''
Hand-eye calibration node using ArUco markers.

Replaces chessboard-based calibration with ArUco marker detection.
Computes the camera-to-gripper (hand-eye) transform AX = XB.

Two modes:
  offline — read images from disk + robot poses from Excel
  online  — subscribe to camera topics, take samples via service (TODO)
'''
import os
import csv
from math import cos, sin, pi
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from handeye_calibration.board_detection import estimate_board_pose
from handeye_calibration.calibration_config import BoardConfig, IntrinsicsConfig, normalize_calibration_setup, resolve_intrinsics, sample_paths

def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def angle2rotation(x, y, z):
    '''RPY (radians) -> 3x3 rotation matrix.'''
    Rx = np.array([
        [
            1,
            0,
            0],
        [
            0,
            cos(x),
            -sin(x)],
        [
            0,
            sin(x),
            cos(x)]])
    Ry = np.array([
        [
            cos(y),
            0,
            sin(y)],
        [
            0,
            1,
            0],
        [
            -sin(y),
            0,
            cos(y)]])
    Rz = np.array([
        [
            cos(z),
            -sin(z),
            0],
        [
            sin(z),
            cos(z),
            0],
        [
            0,
            0,
            1]])
    return Rz @ Ry @ Rx


def gripper2base(rx_deg, ry_deg, rz_deg, tx, ty, tz):
    '''RPY (degrees) + translation -> (R, T) gripper in base frame.'''
    rx = (rx_deg / 180) * pi
    ry = (ry_deg / 180) * pi
    rz = (rz_deg / 180) * pi
    R = angle2rotation(rx, ry, rz)
    T = np.array([
        [
            tx],
        [
            ty],
        [
            tz]], dtype=np.float64)
    return (R, T)


def detect_marker(img, K, distortion, marker_size, marker_id, dictionary):
    '''Detect a single ArUco marker and return object/image points for PnP.

    Returns (obj_pts_3d, img_pts_2d) or (None, None).
    '''
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    params = cv2.aruco.DetectorParameters_create()
    params.minMarkerPerimeterRate = 0.005
    params.polygonalApproxAccuracyRate = 0.1
    (corners, ids, _) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
    if ids is None or marker_id not in ids.flatten():
        return (None, None)
    idx = list(ids.flatten()).index(marker_id)
    corner = corners[idx][0].astype(np.float32)
    corner_input = corner.reshape(-1, 1, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_refined = cv2.cornerSubPix(gray, corner_input, (5, 5), (-1, -1), criteria)
    corner = corner_refined.reshape(-1, 2)
    half = marker_size / 2
    obj_pts = np.array([
        [
            -half,
            half,
            0],
        [
            half,
            half,
            0],
        [
            half,
            -half,
            0],
        [
            -half,
            -half,
            0]], dtype=np.float64)
    return (obj_pts, corner)


def marker_to_camera(img, K, distortion, marker_size, marker_id, dictionary):
    '''Compute marker-to-camera transform via PnP.

    Returns (R, T) or raises RuntimeError.
    '''
    (obj_pts, img_pts) = detect_marker(img, K, distortion, marker_size, marker_id, dictionary)
    if obj_pts is None:
        raise RuntimeError('ArUco marker id={} not found in image'.format(marker_id))
    (retval, rvec, tvec) = cv2.solvePnP(obj_pts, img_pts, K, distortion)
    (R, _) = cv2.Rodrigues(rvec)
    return (R, tvec.reshape(3, 1))


def compute_calibration_outputs(calibration_setup, R_base_to_tracking, T_base_to_tracking, R_target_to_camera, T_target_to_camera, R_camera_to_tracking, T_camera_to_tracking):
    normalized_setup = normalize_calibration_setup(calibration_setup)
    if normalized_setup == 'eye_in_hand':
        R_target_to_base = R_base_to_tracking[0] @ R_camera_to_tracking @ R_target_to_camera[0]
        T_target_to_base = T_base_to_tracking[0] + (R_base_to_tracking[0] @ T_camera_to_tracking) + (R_base_to_tracking[0] @ R_camera_to_tracking @ T_target_to_camera[0])
    elif normalized_setup == 'eye_to_hand':
        R_tracking_to_camera = R_camera_to_tracking.T
        T_tracking_to_camera = -R_tracking_to_camera @ T_camera_to_tracking
        R_target_to_base = R_base_to_tracking[0] @ R_tracking_to_camera @ R_target_to_camera[0]
        T_target_to_base = T_base_to_tracking[0] + (R_base_to_tracking[0] @ T_tracking_to_camera) + (R_base_to_tracking[0] @ R_tracking_to_camera @ T_target_to_camera[0])
    else:
        raise ValueError("Unsupported calibration setup '{}'. Use eye_in_hand or eye_to_hand.".format(calibration_setup))
    return {
        'R_target_to_base': R_target_to_base,
        'T_target_to_base': T_target_to_base }


def make_homogeneous(R, T):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    transform[:3, 3:4] = np.asarray(T, dtype=np.float64).reshape(3, 1)
    return transform


def read_robot_poses(pose_path):
    R_list = []
    T_list = []
    with open(pose_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() in ('tx', 'x'):
                continue
            if len(row) < 6:
                raise ValueError('Pose row must contain tx,ty,tz,rx,ry,rz: {}'.format(row))
            tx, ty, tz, rx, ry, rz = [float(value) for value in row[:6]]
            R, T = gripper2base(rx, ry, rz, tx, ty, tz)
            R_list.append(R)
            T_list.append(T)
    return R_list, T_list


def collect_target_to_camera(image_list, board_config, K, distortion, logger=None,
                             use_pnp_ransac=False,
                             pnp_ransac_iterations=100,
                             pnp_ransac_reprojection_error=3.0,
                             pnp_ransac_confidence=0.99):
    R_list = []
    T_list = []
    detected_indices = []
    for idx, image_path in enumerate(image_list):
        try:
            R, T = estimate_board_pose(
                image_path, board_config, K, distortion,
                use_ransac=use_pnp_ransac,
                ransac_iterations=pnp_ransac_iterations,
                ransac_reprojection_error=pnp_ransac_reprojection_error,
                ransac_confidence=pnp_ransac_confidence)
        except Exception as exc:
            if logger is not None:
                logger.warn('Skip {}: {}'.format(os.path.basename(image_path), exc))
            continue
        R_list.append(R)
        T_list.append(T)
        detected_indices.append(idx)
        if logger is not None:
            logger.info('Detected board in {}'.format(os.path.basename(image_path)))
    return R_list, T_list, detected_indices


def verification_stats(calibration_setup, R_base_to_tracking, T_base_to_tracking,
                       R_target_to_camera, T_target_to_camera,
                       R_camera_to_tracking, T_camera_to_tracking):
    translations = []
    for i in range(len(R_base_to_tracking)):
        outputs = compute_calibration_outputs(
            calibration_setup=calibration_setup,
            R_base_to_tracking=[R_base_to_tracking[i]],
            T_base_to_tracking=[T_base_to_tracking[i]],
            R_target_to_camera=[R_target_to_camera[i]],
            T_target_to_camera=[T_target_to_camera[i]],
            R_camera_to_tracking=R_camera_to_tracking,
            T_camera_to_tracking=T_camera_to_tracking)
        translations.append(outputs['T_target_to_base'].ravel())
    trans = np.asarray(translations, dtype=np.float64)
    mean_t = trans.mean(axis=0)
    devs = np.linalg.norm(trans - mean_t, axis=1)
    return float(np.sqrt(np.mean(devs ** 2))), float(np.max(devs)), mean_t, devs


def target_translation_deviations(calibration_setup, R_base_to_tracking,
                                  T_base_to_tracking, R_target_to_camera,
                                  T_target_to_camera, R_camera_to_tracking,
                                  T_camera_to_tracking):
    translations = []
    for i in range(len(R_base_to_tracking)):
        outputs = compute_calibration_outputs(
            calibration_setup=calibration_setup,
            R_base_to_tracking=[R_base_to_tracking[i]],
            T_base_to_tracking=[T_base_to_tracking[i]],
            R_target_to_camera=[R_target_to_camera[i]],
            T_target_to_camera=[T_target_to_camera[i]],
            R_camera_to_tracking=R_camera_to_tracking,
            T_camera_to_tracking=T_camera_to_tracking)
        translations.append(outputs['T_target_to_base'].ravel())
    trans = np.asarray(translations, dtype=np.float64)
    center = np.median(trans, axis=0)
    return np.linalg.norm(trans - center, axis=1)


def average_eye_to_hand_camera_to_base(R_base_to_target, T_base_to_target,
                                       R_target_to_camera, T_target_to_camera):
    camera_to_base_transforms = []
    for R_bt, T_bt, R_tc, T_tc in zip(R_base_to_target, T_base_to_target,
                                     R_target_to_camera, T_target_to_camera):
        base_to_target = make_homogeneous(R_bt, T_bt)
        target_to_camera = make_homogeneous(R_tc, T_tc)
        camera_to_base_transforms.append(base_to_target @ np.linalg.inv(target_to_camera))

    translations = np.asarray([tf[:3, 3] for tf in camera_to_base_transforms])
    quaternions = []
    from scipy.spatial.transform import Rotation
    for tf in camera_to_base_transforms:
        quat = Rotation.from_matrix(tf[:3, :3]).as_quat()
        if quaternions and np.dot(quaternions[0], quat) < 0:
            quat = -quat
        quaternions.append(quat)
    quat_mean = np.mean(np.asarray(quaternions), axis=0)
    quat_mean = quat_mean / np.linalg.norm(quat_mean)
    return Rotation.from_quat(quat_mean).as_matrix(), translations.mean(axis=0).reshape(3, 1)


def solve_handeye_calibration(calibration_setup, R_base_to_tracking,
                              T_base_to_tracking, R_target_to_camera,
                              T_target_to_camera):
    normalized_setup = normalize_calibration_setup(calibration_setup)
    if len(R_base_to_tracking) < 3:
        raise RuntimeError('Need at least 3 valid pose/image pairs, got {}'.format(
            len(R_base_to_tracking)))

    if normalized_setup == 'eye_to_hand':
        R_cb, T_cb = average_eye_to_hand_camera_to_base(
            R_base_to_tracking, T_base_to_tracking,
            R_target_to_camera, T_target_to_camera)
        rmse, max_dev, _mean_t, _devs = verification_stats(
            normalized_setup, R_base_to_tracking, T_base_to_tracking,
            R_target_to_camera, T_target_to_camera, R_cb, T_cb)
        return [{'method': 'DIRECT_AVERAGE', 'rmse': rmse, 'max_dev': max_dev,
                 'R': R_cb, 'T': T_cb}]

    methods = [
        ('TSAI', cv2.CALIB_HAND_EYE_TSAI),
        ('PARK', cv2.CALIB_HAND_EYE_PARK),
        ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD),
        ('ANDREFF', cv2.CALIB_HAND_EYE_ANDREFF),
        ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]
    results = []
    for name, method in methods:
        try:
            R_cb, T_cb = cv2.calibrateHandEye(
                R_base_to_tracking, T_base_to_tracking,
                R_target_to_camera, T_target_to_camera,
                method=method)
        except cv2.error:
            continue
        T_cb = np.asarray(T_cb, dtype=np.float64).reshape(3, 1)
        rmse, max_dev, _mean_t, _devs = verification_stats(
            normalized_setup, R_base_to_tracking, T_base_to_tracking,
            R_target_to_camera, T_target_to_camera, R_cb, T_cb)
        results.append({'method': name, 'rmse': rmse, 'max_dev': max_dev,
                        'R': R_cb, 'T': T_cb})
    if not results:
        raise RuntimeError('cv2.calibrateHandEye did not return any valid result')
    return sorted(results, key=lambda result: result['rmse'])


def subset(values, indices):
    return [values[i] for i in indices]


def ransac_handeye_indices(calibration_setup, R_base_to_tracking,
                           T_base_to_tracking, R_target_to_camera,
                           T_target_to_camera, threshold=0.005,
                           iterations=100, min_inliers=6,
                           sample_size=3, random_seed=7,
                           logger=None):
    total = len(R_base_to_tracking)
    effective_min_samples = max(int(sample_size), 3)
    if total < effective_min_samples:
        if logger is not None:
            logger.warn(
                'Hand-eye RANSAC skipped: only {} samples, need at least {}'
                .format(total, effective_min_samples))
        return list(range(total))

    effective_min_inliers = min(int(min_inliers), total)
    if effective_min_inliers < int(min_inliers):
        if logger is not None:
            logger.warn(
                'Hand-eye RANSAC min_inliers clamped from {} to {} '
                '(total samples)'.format(int(min_inliers), total))

    rng = np.random.default_rng(int(random_seed))
    best_indices = []
    best_error = float('inf')
    all_indices = np.arange(total)

    for _ in range(int(iterations)):
        sample_indices = sorted(rng.choice(all_indices, size=int(sample_size),
                                           replace=False).tolist())
        try:
            candidates = solve_handeye_calibration(
                calibration_setup,
                subset(R_base_to_tracking, sample_indices),
                subset(T_base_to_tracking, sample_indices),
                subset(R_target_to_camera, sample_indices),
                subset(T_target_to_camera, sample_indices))
        except Exception:
            continue

        for candidate in candidates[:1]:
            devs = target_translation_deviations(
                calibration_setup, R_base_to_tracking, T_base_to_tracking,
                R_target_to_camera, T_target_to_camera,
                candidate['R'], candidate['T'])
            inliers = np.where(devs <= float(threshold))[0].tolist()
            if len(inliers) < effective_min_inliers:
                continue
            mean_error = float(np.mean(devs[inliers]))
            if (len(inliers) > len(best_indices)
                    or (len(inliers) == len(best_indices)
                        and mean_error < best_error)):
                best_indices = inliers
                best_error = mean_error

    if not best_indices:
        if logger is not None:
            logger.warn(
                'Hand-eye RANSAC found no consensus '
                '(min_inliers={}, threshold={:.3f}m); '
                'using all {} samples'.format(
                    effective_min_inliers, float(threshold), total))
        return list(range(total))

    if logger is not None:
        logger.info(
            'Hand-eye RANSAC kept {}/{} samples, '
            'rejected {}, mean inlier deviation {:.6f}m'
            .format(len(best_indices), total,
                    total - len(best_indices), best_error))
    return best_indices


def run_handeye_calibration(calibration_setup, R_base_to_tracking, T_base_to_tracking,
                            R_target_to_camera, T_target_to_camera,
                            use_ransac=False, ransac_threshold=0.005,
                            ransac_iterations=100, ransac_min_inliers=6,
                            ransac_sample_size=3, logger=None):
    if not parse_bool(use_ransac):
        return solve_handeye_calibration(
            calibration_setup, R_base_to_tracking, T_base_to_tracking,
            R_target_to_camera, T_target_to_camera)

    inliers = ransac_handeye_indices(
        calibration_setup, R_base_to_tracking, T_base_to_tracking,
        R_target_to_camera, T_target_to_camera,
        threshold=ransac_threshold,
        iterations=ransac_iterations,
        min_inliers=ransac_min_inliers,
        sample_size=ransac_sample_size,
        logger=logger)
    results = solve_handeye_calibration(
        calibration_setup,
        subset(R_base_to_tracking, inliers),
        subset(T_base_to_tracking, inliers),
        subset(R_target_to_camera, inliers),
        subset(T_target_to_camera, inliers))
    for result in results:
        result['inlier_count'] = len(inliers)
        result['sample_count'] = len(R_base_to_tracking)
    return results


def write_handeye_results(path, results):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    header = ['method', 'rmse', 'max_dev']
    header.extend(['m{}{}'.format(r, c) for r in range(4) for c in range(4)])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in sorted(results, key=lambda item: item['rmse']):
            RT = make_homogeneous(result['R'], result['T'])
            writer.writerow([
                result['method'],
                '{:.9f}'.format(float(result['rmse'])),
                '{:.9f}'.format(float(result.get('max_dev', 0.0))),
            ] + ['{:.12g}'.format(float(value)) for value in RT.reshape(-1)])


class HandeyeCalibrationNode(Node):
    '''ROS 2 node for hand-eye calibration with ArUco markers.'''
    
    def __init__(self = None):
        super().__init__('handeye_calibration')
        self.declare_parameter('fx', 0)
        self.declare_parameter('fy', 0)
        self.declare_parameter('cx', 0)
        self.declare_parameter('cy', 0)
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0)
        self.declare_parameter('dictionary', 'auto')
        self.declare_parameter('board_type', 'chessboard')
        self.declare_parameter('squares_x', 0)
        self.declare_parameter('squares_y', 0)
        self.declare_parameter('square_size', 0)
        self.declare_parameter('sample_dir', '')
        self.declare_parameter('image_dir', '')
        self.declare_parameter('pose_file', '')
        self.declare_parameter('intrinsics_file', '')
        self.declare_parameter('intrinsics_source', 'auto')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('calibration_setup', 'eye_in_hand')
        self.declare_parameter('use_ransac', True)
        self.declare_parameter('pnp_ransac_reprojection_error', 3.0)
        self.declare_parameter('pnp_ransac_iterations', 100)
        self.declare_parameter('pnp_ransac_confidence', 0.99)
        self.declare_parameter('handeye_ransac_threshold', 0.005)
        self.declare_parameter('handeye_ransac_iterations', 100)
        self.declare_parameter('handeye_ransac_min_inliers', 6)
        self.declare_parameter('handeye_ransac_sample_size', 3)
        self.board_config = BoardConfig.from_values(board_type=self.get_parameter('board_type').value, dictionary=self.get_parameter('dictionary').value, marker_id=self.get_parameter('marker_id').value, marker_size=self.get_parameter('marker_size').value, squares_x=self.get_parameter('squares_x').value, squares_y=self.get_parameter('squares_y').value, square_size=self.get_parameter('square_size').value)
        paths = sample_paths(sample_dir=self.get_parameter('sample_dir').value, board_type=self.board_config.board_type, image_dir=self.get_parameter('image_dir').value, pose_file=self.get_parameter('pose_file').value, output_dir=self.get_parameter('output_dir').value, calibration_setup=self.get_parameter('calibration_setup').value)
        self.sample_dir = paths.sample_dir
        self.image_dir = paths.image_dir
        self.pose_file = paths.pose_file
        self.output_dir = paths.output_dir
        self.calibration_setup = normalize_calibration_setup(self.get_parameter('calibration_setup').value)
        intrinsics_file = self.get_parameter('intrinsics_file').value
        intrinsics_source = self.get_parameter('intrinsics_source').value
        if intrinsics_file and intrinsics_source == 'auto':
            intrinsics_source = 'file'
        if not self.get_parameter('fx').value:
            pass
        if not self.get_parameter('fy').value:
            pass
        if not self.get_parameter('cx').value:
            pass
        if not self.get_parameter('cy').value:
            pass
        intr_cfg = IntrinsicsConfig(source=intrinsics_source, explicit_file=intrinsics_file, experiment_dir=self.output_dir, fx=606.25, fy=605.65, cx=321.501, cy=242.543)
        (self.K, intrinsics_origin) = resolve_intrinsics(intr_cfg)
        self.get_logger().info('Using intrinsics from: {}'.format(intrinsics_origin))
        self.distortion = np.zeros((1, 5), dtype=np.float64)
        self.bridge = CvBridge()
        self.get_logger().info('Handeye calibration node started')
        self.get_logger().info('K = [{:.2f}, {:.2f}, {:.2f}, {:.2f}], board_type={}, marker_id={}, size={}m'.format(self.K[(0, 0)], self.K[(1, 1)], self.K[(0, 2)], self.K[(1, 2)], self.board_config.board_type, self.board_config.marker_id, self.board_config.marker_size))
        self.get_logger().info('Calibration setup: {}'.format(self.calibration_setup))
        self.get_logger().info('RANSAC enabled: {}'.format(
            self.get_parameter('use_ransac').value))
        self.get_logger().info('Sample paths: image_dir={}, pose_file={}, output_dir={}'.format(self.image_dir, self.pose_file, self.output_dir))
        self.get_logger().info('Running in offline mode')
        self._offline_timer = self.create_timer(0.1, self._run_offline_once)

    
    def _run_offline_once(self):
        self._offline_timer.cancel()
        self._run_offline(self.image_dir, self.pose_file)
        self.get_logger().info('Offline calibration finished, exiting')
        raise SystemExit(0)

    
    def _run_offline(self, img_dir, pose_path):
        '''Offline calibration from disk (same flow as reference).'''
        image_list = []
        for root, _dirs, files in os.walk(img_dir):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    image_list.append(os.path.join(root, f))
        if len(image_list) < 2:
            self.get_logger().error('Need at least 2 images, found {}'.format(len(image_list)))
            return None
        R_gb_all, T_gb_all = read_robot_poses(pose_path)
        if len(R_gb_all) != len(image_list):
            raise RuntimeError('Image/pose count mismatch: {} images vs {} poses'.format(
                len(image_list), len(R_gb_all)))
        R_tc_list, T_tc_list, detected_indices = collect_target_to_camera(
            image_list, self.board_config, self.K, self.distortion,
            logger=self.get_logger(),
            use_pnp_ransac=self.get_parameter('use_ransac').value,
            pnp_ransac_iterations=self.get_parameter(
                'pnp_ransac_iterations').value,
            pnp_ransac_reprojection_error=self.get_parameter(
                'pnp_ransac_reprojection_error').value,
            pnp_ransac_confidence=self.get_parameter(
                'pnp_ransac_confidence').value)
        if len(R_tc_list) < 3:
            raise RuntimeError('Need at least 3 valid detections, got {}'.format(len(R_tc_list)))
        R_gb = [R_gb_all[i] for i in detected_indices]
        T_gb = [T_gb_all[i] for i in detected_indices]
        total_samples = len(R_tc_list)
        self.get_logger().info(
            'Running hand-eye calibration on {} samples (setup: {})'
            .format(total_samples, self.calibration_setup))

        # 1) Non-RANSAC — all samples, always computed
        results_no_ransac = run_handeye_calibration(
            self.calibration_setup, R_gb, T_gb, R_tc_list, T_tc_list,
            use_ransac=False, logger=self.get_logger())
        no_ransac_path = os.path.join(self.output_dir,
                                      'handeye_results_no_ransac.csv')
        write_handeye_results(no_ransac_path, results_no_ransac)
        self.get_logger().info(
            'Saved non-RANSAC results to {}'.format(no_ransac_path))
        for result in results_no_ransac:
            self.get_logger().info(
                '[no-RANSAC] {}: rmse={:.6f}m max_dev={:.6f}m'.format(
                    result['method'], result['rmse'], result['max_dev']))

        # 2) RANSAC — always enabled for the primary output
        ransac_kwargs = dict(
            ransac_threshold=self.get_parameter('handeye_ransac_threshold').value,
            ransac_iterations=self.get_parameter('handeye_ransac_iterations').value,
            ransac_min_inliers=self.get_parameter('handeye_ransac_min_inliers').value,
            ransac_sample_size=self.get_parameter('handeye_ransac_sample_size').value,
            logger=self.get_logger())
        results = run_handeye_calibration(
            self.calibration_setup, R_gb, T_gb, R_tc_list, T_tc_list,
            use_ransac=True, **ransac_kwargs)
        result_path = os.path.join(self.output_dir, 'handeye_results.csv')
        write_handeye_results(result_path, results)
        self.get_logger().info('Saved RANSAC results to {}'.format(result_path))
        for result in results:
            self.get_logger().info(
                '[RANSAC] {}: rmse={:.6f}m max_dev={:.6f}m '
                'inliers={}/{}'.format(
                    result['method'], result['rmse'], result['max_dev'],
                    result.get('inlier_count', total_samples),
                    result.get('sample_count', total_samples)))

        # Print matrix for best RANSAC result
        if results:
            self._print_result(results[0]['R'], results[0]['T'],
                               R_gb, T_gb, R_tc_list, T_tc_list,
                               self.calibration_setup)

        # Compare best RANSAC vs best non-RANSAC
        best_ransac = results[0] if results else None
        best_no_ransac = results_no_ransac[0] if results_no_ransac else None
        if best_ransac and best_no_ransac:
            R_r = np.asarray(best_ransac['R'], dtype=np.float64)
            T_r = np.asarray(best_ransac['T'], dtype=np.float64).ravel()
            R_n = np.asarray(best_no_ransac['R'], dtype=np.float64)
            T_n = np.asarray(best_no_ransac['T'], dtype=np.float64).ravel()
            trans_diff = np.linalg.norm(T_r - T_n)
            # Rotation difference as an angle (degrees) via relative rotation
            R_rel = R_r.T @ R_n
            trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
            rot_angle_rad = np.arccos((trace_val - 1.0) / 2.0)
            rot_angle_deg = float(np.degrees(rot_angle_rad))
            self.get_logger().info(
                '==================================================')
            self.get_logger().info(
                'RANSAC vs no-RANSAC comparison (best method each):')
            self.get_logger().info(
                '  translation diff = {:.3f} mm'.format(trans_diff * 1000))
            self.get_logger().info(
                '  rotation diff    = {:.4f} deg'.format(rot_angle_deg))
            if trans_diff < 1e-9 and rot_angle_deg < 1e-6:
                self.get_logger().warn(
                    '  -> Results are IDENTICAL. '
                    'RANSAC did not reject any samples '
                    '(try tighter --handeye_ransac_threshold).')
            self.get_logger().info(
                '==================================================')

        return results

    
    def _print_result(self, R_cb, T_cb, R_gb, T_gb, R_tc, T_tc, calibration_setup):
        '''Print hand-eye matrix and per-sample verification.'''
        RT_handeye = make_homogeneous(R_cb, T_cb)
        label = 'camera -> gripper' if calibration_setup == 'eye_in_hand' else 'camera -> base'
        print('\nCalibration matrix ({}):\n{}\n'.format(label, RT_handeye))
        rmse, max_dev, mean_t, devs = verification_stats(
            calibration_setup, R_gb, T_gb, R_tc, T_tc, R_cb, T_cb)
        print('Verification (target->base should be constant):')
        print('  mean translation: [{:.6f}, {:.6f}, {:.6f}] m'.format(*mean_t))
        print('  translation rmse: {:.6f} m, max: {:.6f} m'.format(rmse, max_dev))
        for idx, dev in enumerate(devs):
            print('  sample {:04d}: dev={:.6f} m'.format(idx, dev))


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeCalibrationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
