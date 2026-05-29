"""
Launch handeye_calibration in offline mode (images + pose file).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

SAMPLE_ROOT = '/home/young/ros2_ws/src/handeye_calibration/samples'


def generate_launch_description():
    sample_dir = PathJoinSubstitution([
        SAMPLE_ROOT,
        LaunchConfiguration('calibration_setup'),
        LaunchConfiguration('board_type')])

    return LaunchDescription([
        DeclareLaunchArgument(
            'board_type', default_value='chessboard',
            description='single_aruco, charuco, aruco_grid, or chessboard'),
        DeclareLaunchArgument(
            'intrinsics_source', default_value='auto',
            description='auto, camera_info, calibrated, file, or manual'),
        DeclareLaunchArgument(
            'calibration_setup', default_value='eye_in_hand',
            description='eye_in_hand or eye_to_hand'),
        DeclareLaunchArgument(
            'use_ransac', default_value='true',
            description='Use RANSAC for PnP and hand-eye sample rejection'),
        DeclareLaunchArgument(
            'pnp_ransac_reprojection_error', default_value='3.0',
            description='solvePnPRansac reprojection threshold in pixels'),
        DeclareLaunchArgument(
            'pnp_ransac_iterations', default_value='100',
            description='solvePnPRansac iteration count'),
        DeclareLaunchArgument(
            'pnp_ransac_confidence', default_value='0.99',
            description='solvePnPRansac confidence'),
        DeclareLaunchArgument(
            'handeye_ransac_threshold', default_value='0.005',
            description='Hand-eye RANSAC target consistency threshold in meters'),
        DeclareLaunchArgument(
            'handeye_ransac_iterations', default_value='100',
            description='Hand-eye RANSAC iteration count'),
        DeclareLaunchArgument(
            'handeye_ransac_min_inliers', default_value='6',
            description='Minimum inlier samples required by hand-eye RANSAC'),
        DeclareLaunchArgument(
            'handeye_ransac_sample_size', default_value='3',
            description='Sample count used for each hand-eye RANSAC hypothesis'),
        Node(
            package='handeye_calibration',
            executable='aruco_handeye_calibrator',
            name='handeye_calibration',
            output='screen',
            parameters=[{
                'sample_dir': sample_dir,
                'image_dir': PathJoinSubstitution([sample_dir, 'img']),
                'pose_file': PathJoinSubstitution([sample_dir, 'poses.csv']),
                'board_type': LaunchConfiguration('board_type'),
                'output_dir': sample_dir,
                'intrinsics_source': LaunchConfiguration('intrinsics_source'),
                'calibration_setup': LaunchConfiguration('calibration_setup'),
                'use_ransac': LaunchConfiguration('use_ransac'),
                'pnp_ransac_reprojection_error': LaunchConfiguration(
                    'pnp_ransac_reprojection_error'),
                'pnp_ransac_iterations': LaunchConfiguration(
                    'pnp_ransac_iterations'),
                'pnp_ransac_confidence': LaunchConfiguration(
                    'pnp_ransac_confidence'),
                'handeye_ransac_threshold': LaunchConfiguration(
                    'handeye_ransac_threshold'),
                'handeye_ransac_iterations': LaunchConfiguration(
                    'handeye_ransac_iterations'),
                'handeye_ransac_min_inliers': LaunchConfiguration(
                    'handeye_ransac_min_inliers'),
                'handeye_ransac_sample_size': LaunchConfiguration(
                    'handeye_ransac_sample_size'),
            }]),
    ])
