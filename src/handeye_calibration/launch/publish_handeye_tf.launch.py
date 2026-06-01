"""Publish camera_link TF from saved hand-eye calibration results."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'result_file',
            default_value='',
            description='Path to handeye_results.csv; auto-resolved when empty'),
        DeclareLaunchArgument(
            'sample_dir',
            default_value='',
            description='Calibration sample directory; used when result_file is empty'),
        DeclareLaunchArgument(
            'board_type',
            default_value='chessboard',
            description='Board type used to resolve the default result path'),
        DeclareLaunchArgument(
            'calibration_setup',
            default_value='eye_in_hand',
            description='eye_in_hand or eye_to_hand'),
        DeclareLaunchArgument(
            'method',
            default_value='best',
            description='Calibration row to publish: best or method name such as TSAI'),
        DeclareLaunchArgument(
            'parent_frame',
            default_value='',
            description='Parent frame (auto: fr3_link8 for eye_in_hand, '
                        'fr3_link0 for eye_to_hand)'),
        DeclareLaunchArgument(
            'child_frame',
            default_value='camera_link',
            description='Target frame to publish, e.g. camera_link'),
        DeclareLaunchArgument(
            'optical_frame',
            default_value='camera_color_optical_frame',
            description='Optical frame used during calibration (bridge to child_frame)'),
        Node(
            package='handeye_calibration',
            executable='handeye_tf_publisher',
            name='handeye_tf_publisher',
            output='screen',
            parameters=[{
                'result_file': LaunchConfiguration('result_file'),
                'sample_dir': LaunchConfiguration('sample_dir'),
                'board_type': LaunchConfiguration('board_type'),
                'calibration_setup': LaunchConfiguration('calibration_setup'),
                'method': LaunchConfiguration('method'),
                'parent_frame': LaunchConfiguration('parent_frame'),
                'child_frame': LaunchConfiguration('child_frame'),
                'optical_frame': LaunchConfiguration('optical_frame'),
            }]),
    ])
