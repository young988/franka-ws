"""Shared configuration helpers for calibration experiments."""
from dataclasses import dataclass
import os
import numpy as np

DEFAULT_SAMPLE_ROOT = '/home/young/ros2_ws/src/handeye_calibration/samples'
DEFAULT_SAMPLE_DIR = DEFAULT_SAMPLE_ROOT

BOARD_PRESETS = {
    'chessboard': {
        'dictionary': 'DICT_ARUCO_ORIGINAL',
        'marker_id': 582,
        'marker_size': 0.1,
        'squares_x': 11,
        'squares_y': 8,
        'square_size': 0.015,
    },
    'single_aruco': {
        'dictionary': 'DICT_ARUCO_ORIGINAL',
        'marker_id': 582,
        'marker_size': 0.1,
        'squares_x': 1,
        'squares_y': 1,
        'square_size': 0.1,
    },
    'charuco': {
        'dictionary': 'DICT_ARUCO_ORIGINAL',
        'marker_id': 582,
        'marker_size': 0.02,
        'squares_x': 5,
        'squares_y': 7,
        'square_size': 0.04,
    },
    'aruco_grid': {
        'dictionary': 'DICT_ARUCO_ORIGINAL',
        'marker_id': 582,
        'marker_size': 0.03,
        'squares_x': 5,
        'squares_y': 7,
        'square_size': 0.04,
    },
}

VALID_CALIBRATION_SETUPS = ('eye_in_hand', 'eye_to_hand')


def _auto_string(value):
    text = str(value or '').strip()
    return text.lower() in ('', 'auto')


def _auto_int(value):
    return value is None or int(value) <= 0


def _auto_float(value):
    return value is None or float(value) <= 0.0


@dataclass(frozen=True)
class BoardConfig:
    board_type: str
    dictionary: str
    marker_id: int
    marker_size: float
    squares_x: int
    squares_y: int
    square_size: float

    @classmethod
    def from_values(cls, board_type, dictionary, marker_id, marker_size,
                    squares_x, squares_y, square_size):
        normalized_type = normalize_board_type(board_type)
        preset = BOARD_PRESETS.get(normalized_type,
                                   BOARD_PRESETS['single_aruco'])

        normalized_dict = str(dictionary or 'DICT_ARUCO_ORIGINAL').strip()
        if _auto_string(dictionary):
            normalized_dict = preset['dictionary']
        if not normalized_dict.startswith('DICT_'):
            normalized_dict = 'DICT_' + normalized_dict

        resolved_square_size = (preset['square_size']
                                if _auto_float(square_size)
                                else float(square_size))
        resolved_marker_size = (preset['marker_size']
                                if _auto_float(marker_size)
                                else float(marker_size))
        if (normalized_type in ('charuco', 'aruco_grid')
                and resolved_marker_size >= resolved_square_size):
            resolved_marker_size = preset['marker_size']

        return cls(
            normalized_type,
            normalized_dict,
            preset['marker_id'] if _auto_int(marker_id) else int(marker_id),
            resolved_marker_size,
            preset['squares_x'] if _auto_int(squares_x) else int(squares_x),
            preset['squares_y'] if _auto_int(squares_y) else int(squares_y),
            resolved_square_size,
        )


@dataclass(frozen=True)
class CalibrationFrames:
    calibration_setup: str
    robot_base_frame: str
    robot_effector_frame: str
    tracking_base_frame: str
    tracking_marker_frame: str


def normalize_board_type(board_type):
    normalized_type = str(board_type or 'single_aruco').strip().lower()
    normalized_type = normalized_type.replace('-', '_')
    if normalized_type in ('aruco', 'single'):
        normalized_type = 'single_aruco'
    return normalized_type


def normalize_calibration_setup(calibration_setup):
    normalized_setup = str(calibration_setup or 'eye_in_hand').strip().lower()
    normalized_setup = normalized_setup.replace('-', '_')
    if normalized_setup not in VALID_CALIBRATION_SETUPS:
        raise ValueError(
            "Unsupported calibration setup '{}'. "
            "Use eye_in_hand or eye_to_hand.".format(calibration_setup))
    return normalized_setup


def resolve_calibration_frames(calibration_setup, robot_base_frame,
                               robot_effector_frame, tracking_base_frame='',
                               tracking_marker_frame=''):
    normalized_setup = normalize_calibration_setup(calibration_setup)
    resolved_tracking_base_frame = tracking_base_frame or robot_base_frame
    resolved_tracking_marker_frame = (tracking_marker_frame
                                      or robot_effector_frame)
    return CalibrationFrames(
        calibration_setup=normalized_setup,
        robot_base_frame=robot_base_frame,
        robot_effector_frame=robot_effector_frame,
        tracking_base_frame=resolved_tracking_base_frame,
        tracking_marker_frame=resolved_tracking_marker_frame,
    )


def sample_dir_for_board(board_type,
                         sample_root=DEFAULT_SAMPLE_ROOT,
                         calibration_setup='eye_in_hand'):
    return os.path.join(sample_root,
                        normalize_calibration_setup(calibration_setup),
                        normalize_board_type(board_type))


def resolve_sample_dir(value, board_type,
                       sample_root=DEFAULT_SAMPLE_ROOT,
                       calibration_setup='eye_in_hand'):
    if value:
        return value
    return sample_dir_for_board(board_type,
                                sample_root=sample_root,
                                calibration_setup=calibration_setup)


@dataclass(frozen=True)
class SamplePaths:
    sample_dir: str
    image_dir: str
    pose_file: str
    output_dir: str


def sample_paths(sample_dir='', board_type='', image_dir='',
                 pose_file='', output_dir='', calibration_setup='eye_in_hand'):
    resolved_sample_dir = resolve_sample_dir(
        sample_dir, board_type, calibration_setup=calibration_setup)
    if not image_dir:
        image_dir = os.path.join(resolved_sample_dir, 'img')
    if not pose_file:
        pose_file = os.path.join(resolved_sample_dir, 'poses.csv')
    if not output_dir:
        output_dir = resolved_sample_dir
    return SamplePaths(resolved_sample_dir, image_dir, pose_file, output_dir)


@dataclass(frozen=True)
class IntrinsicsConfig:
    source: str = 'camera_info'
    explicit_file: str = ''
    experiment_dir: str = DEFAULT_SAMPLE_DIR
    board_type: str = 'single_aruco'
    fx: float = 606.25
    fy: float = 605.65
    cx: float = 321.501
    cy: float = 242.543

    @property
    def normalized_source(self):
        return str(self.source or 'camera_info').strip().lower()


def resolve_experiment_dir(value=None, board_type=None,
                           calibration_setup='eye_in_hand'):
    if value:
        return value
    if board_type is None:
        return DEFAULT_SAMPLE_DIR
    return sample_dir_for_board(board_type,
                                calibration_setup=calibration_setup)


def intrinsics_candidates(config):
    source = config.normalized_source
    experiment_dir = resolve_experiment_dir(config.experiment_dir,
                                            config.board_type)
    if source == 'file':
        if config.explicit_file:
            return [config.explicit_file]
        return []
    if source == 'calibrated':
        return [os.path.join(experiment_dir,
                             'camera_intrinsics_calibrated.txt')]
    if source == 'camera_info':
        return [os.path.join(experiment_dir,
                             'camera_intrinsics_camera_info.txt'),
                os.path.join(experiment_dir, 'camera_intrinsics.txt')]
    if source == 'auto':
        paths = []
        if config.explicit_file:
            paths.append(config.explicit_file)
        paths.extend([
            os.path.join(experiment_dir, 'camera_intrinsics_calibrated.txt'),
            os.path.join(experiment_dir, 'camera_intrinsics_camera_info.txt'),
            os.path.join(experiment_dir, 'camera_intrinsics.txt'),
        ])
        return paths
    if source == 'manual':
        return []
    raise ValueError(
        "Unsupported intrinsics_source '{}'. "
        "Use camera_info, calibrated, file, auto, or manual.".format(
            config.source))


def load_intrinsics_matrix(path):
    K = np.loadtxt(path)
    if K.ndim == 1:
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError('Intrinsics file must contain a 3x3 matrix: {}'
                         .format(path))
    return K.astype(np.float64)


def manual_intrinsics_matrix(config):
    return np.array([
        [float(config.fx), 0, float(config.cx)],
        [0, float(config.fy), float(config.cy)],
        [0, 0, 1],
    ], dtype=np.float64)


def resolve_intrinsics(config):
    if config.normalized_source == 'manual':
        return (manual_intrinsics_matrix(config), 'manual')
    for path in intrinsics_candidates(config):
        if path and os.path.exists(path):
            return (load_intrinsics_matrix(path), path)
    if config.normalized_source == 'auto':
        return (manual_intrinsics_matrix(config), 'manual')
    raise FileNotFoundError(
        "No intrinsics file found for source '{}'. Checked: {}".format(
            config.source,
            ', '.join(intrinsics_candidates(config)) or '<none>'))


def write_intrinsics(path, K, header='fx 0 cx / 0 fy cy / 0 0 1'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.asarray(K, dtype=np.float64).reshape(3, 3),
               fmt='%.6f', header=header)
