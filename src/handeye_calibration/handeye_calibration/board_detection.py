"""Board detection helpers shared by camera and hand-eye calibration."""
import cv2
import numpy as np


def aruco_dictionary(name):
    dict_name = name if str(name).startswith('DICT_') else 'DICT_' + str(name)
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dict_name, cv2.aruco.DICT_ARUCO_ORIGINAL))


def detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def chessboard_object_points(corners_x, corners_y, square_size):
    obj = np.zeros((int(corners_x) * int(corners_y), 3), np.float32)
    grid = np.mgrid[0:int(corners_x), 0:int(corners_y)].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size)
    return obj


def _make_charuco_board(config):
    aruco_dict = aruco_dictionary(config.dictionary)
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        return cv2.aruco.CharucoBoard_create(
            config.squares_x, config.squares_y,
            config.square_size, config.marker_size, aruco_dict)
    return cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_size, config.marker_size, aruco_dict)


def _make_grid_board(config):
    aruco_dict = aruco_dictionary(config.dictionary)
    if hasattr(cv2.aruco, 'GridBoard_create'):
        return cv2.aruco.GridBoard_create(
            config.squares_x, config.squares_y,
            config.marker_size, config.square_size - config.marker_size,
            aruco_dict)
    return cv2.aruco.GridBoard(
        (config.squares_x, config.squares_y),
        config.marker_size, config.square_size - config.marker_size,
        aruco_dict)


def _board_ids(board):
    if hasattr(board, 'getIds'):
        return board.getIds()
    return board.ids


def _board_object_points(board):
    if hasattr(board, 'getObjPoints'):
        return board.getObjPoints()
    return board.objPoints


def _charuco_corners(board):
    if hasattr(board, 'getChessboardCorners'):
        return board.getChessboardCorners()
    return board.chessboardCorners


def detect_calibration_points(image, config):
    """Return (object_points, image_points) for intrinsic calibration."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if config.board_type == 'chessboard':
        pattern = (config.squares_x, config.squares_y)
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            return (None, None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return (
            chessboard_object_points(config.squares_x, config.squares_y,
                                     config.square_size),
            refined.reshape(-1, 2))

    aruco_dict = aruco_dictionary(config.dictionary)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=detector_parameters())

    if ids is None:
        return (None, None)

    if config.board_type == 'charuco':
        board = _make_charuco_board(config)
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if count is None or count < 4:
            return (None, None)
        board_corners = _charuco_corners(board)
        obj = np.asarray(
            [board_corners[int(i)] for i in charuco_ids.flatten()],
            dtype=np.float32)
        return (obj, charuco_corners.reshape(-1, 2))

    if config.board_type == 'aruco_grid':
        board = _make_grid_board(config)
        obj_points = []
        img_points = []
        ids_flat = np.asarray(_board_ids(board)).flatten()
        obj_points_by_marker = _board_object_points(board)
        id_to_index = {int(marker_id): idx
                       for idx, marker_id in enumerate(ids_flat)}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            idx = id_to_index.get(int(marker_id))
            if idx is None:
                continue
            obj_points.extend(
                np.asarray(obj_points_by_marker[idx], dtype=np.float32)
                .reshape(4, 3))
            img_points.extend(marker_corners.reshape(4, 2))
        if len(obj_points) < 4:
            return (None, None)
        return (np.asarray(obj_points, dtype=np.float32),
                np.asarray(img_points, dtype=np.float32))

    if config.board_type == 'single_aruco':
        target = int(config.marker_id)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != target:
                continue
            half = config.marker_size / 2.0
            obj = np.array([
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)
            return (obj, marker_corners.reshape(4, 2).astype(np.float32))

    raise ValueError('Unsupported board_type: {}'.format(config.board_type))


def estimate_board_pose(image_path, config, K, distortion=None,
                        use_ransac=False, ransac_iterations=100,
                        ransac_reprojection_error=3.0,
                        ransac_confidence=0.99):
    """Return board-to-camera pose (R, T) for any supported board type."""
    if distortion is None:
        distortion = np.zeros((1, 5), dtype=np.float64)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError('Failed to read image: {}'.format(image_path))

    if config.board_type in ('single_aruco', 'chessboard', 'aruco_grid'):
        obj_pts, img_pts = detect_calibration_points(image, config)
        if obj_pts is None:
            raise RuntimeError('Board not found in image')
        obj_pts = np.asarray(obj_pts, dtype=np.float64)
        img_pts = np.asarray(img_pts, dtype=np.float64)
        if parse_bool(use_ransac) and len(obj_pts) >= 6:
            ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, K, distortion,
                iterationsCount=int(ransac_iterations),
                reprojectionError=float(ransac_reprojection_error),
                confidence=float(ransac_confidence))
        else:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, distortion)
        if not ok:
            raise RuntimeError('solvePnP failed')
        R, _ = cv2.Rodrigues(rvec)
        return (R, tvec.reshape(3, 1))

    if config.board_type == 'charuco':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco_dictionary(config.dictionary)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_parameters())
        if ids is None:
            raise RuntimeError('ChArUco markers not found')
        board = _make_charuco_board(config)
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if count is None or count < 4:
            raise RuntimeError('Not enough ChArUco corners')
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, distortion, None, None)
        if not ok:
            raise RuntimeError('estimatePoseCharucoBoard failed')
        R, _ = cv2.Rodrigues(rvec)
        return (R, tvec.reshape(3, 1))

    raise RuntimeError('Unsupported board_type: {}'.format(config.board_type))
