import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from handeye_calibration.aruco_handeye_calibrator import (
    compute_calibration_outputs,
    write_handeye_results,
)
from handeye_calibration.pixel_to_robot import (
    build_grasp_goal,
    build_move_goal,
    camera_point_from_depth_image,
    default_target_quaternion,
    parse_bool_parameter,
)


def matrix(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3:4] = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return transform


def split(transform):
    return transform[:3, :3], transform[:3, 3:4]


def test_eye_in_hand_output_keeps_target_fixed_in_base():
    base_to_gripper = matrix(
        Rotation.from_euler("z", 45, degrees=True).as_matrix(),
        [0.3, 0.1, 0.4],
    )
    camera_to_gripper = matrix(
        Rotation.from_euler("x", 90, degrees=True).as_matrix(),
        [0.02, -0.03, 0.08],
    )
    target_to_base = matrix(np.eye(3), [0.6, -0.2, 0.25])

    target_to_camera = np.linalg.inv(camera_to_gripper) @ np.linalg.inv(base_to_gripper) @ target_to_base

    R_target_to_base, T_target_to_base = split(target_to_base)
    outputs = compute_calibration_outputs(
        calibration_setup="eye_in_hand",
        R_base_to_tracking=[base_to_gripper[:3, :3]],
        T_base_to_tracking=[base_to_gripper[:3, 3:4]],
        R_target_to_camera=[target_to_camera[:3, :3]],
        T_target_to_camera=[target_to_camera[:3, 3:4]],
        R_camera_to_tracking=camera_to_gripper[:3, :3],
        T_camera_to_tracking=camera_to_gripper[:3, 3:4],
    )

    np.testing.assert_allclose(outputs["R_target_to_base"], R_target_to_base, atol=1e-9)
    np.testing.assert_allclose(outputs["T_target_to_base"], T_target_to_base, atol=1e-9)


def test_write_handeye_results_creates_csv_with_flattened_matrix(tmp_path):
    result_path = tmp_path / "handeye_results.csv"
    R = np.eye(3)
    T = np.array([[0.1], [0.2], [0.3]])

    write_handeye_results(result_path, [{"method": "TSAI", "rmse": 0.004, "max_dev": 0.007, "R": R, "T": T}])

    with result_path.open() as f:
        rows = list(csv.reader(f))

    assert rows[0][:3] == ["method", "rmse", "max_dev"]
    assert rows[1][0] == "TSAI"
    assert float(rows[1][1]) == 0.004
    values = [float(value) for value in rows[1][3:19]]
    np.testing.assert_allclose(np.asarray(values).reshape(4, 4)[:3, 3], [0.1, 0.2, 0.3])


def test_parse_bool_parameter_treats_false_string_as_false():
    assert parse_bool_parameter("false") is False
    assert parse_bool_parameter("0") is False
    assert parse_bool_parameter(False) is False
    assert parse_bool_parameter("true") is True


def test_camera_point_from_depth_image_clamps_median_window_at_image_edges():
    depth = np.array([
        [1000, 1000, 0],
        [1000, 2000, 2000],
        [0, 2000, 2000],
    ], dtype=np.uint16)

    point = camera_point_from_depth_image(
        depth_image=depth,
        u=0,
        v=0,
        fx=100.0,
        fy=100.0,
        cx=0.0,
        cy=0.0,
        depth_window=5,
    )

    np.testing.assert_allclose(point.ravel(), [0.0, 0.0, 2.0])


def test_default_target_quaternion_points_tcp_down():
    quat = default_target_quaternion()
    assert (quat.x, quat.y, quat.z, quat.w) == (1.0, 0.0, 0.0, 0.0)


def test_build_move_goal_opens_to_requested_width():
    goal = build_move_goal(width=0.08, speed=0.03)
    assert goal.width == 0.08
    assert goal.speed == 0.03


def test_build_grasp_goal_closes_with_force_and_epsilon():
    goal = build_grasp_goal(speed=0.03, force=10.0, epsilon_width=0.005)
    assert goal.width == 0.0
    assert goal.speed == 0.03
    assert goal.force == 10.0
    assert goal.epsilon.inner == 0.005
    assert goal.epsilon.outer == 0.005
