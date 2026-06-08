from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("franka_telep"))
    robot_description = (package_share / "urdf" / "fr3_teleop_preview.urdf").read_text()
    config_file = str(package_share / "config" / "franka_telep.yaml")
    rviz_config = str(package_share / "rviz" / "fr3_teleop_preview.rviz")

    start_servo_reader = LaunchConfiguration("start_servo_reader")
    start_rviz = LaunchConfiguration("start_rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_servo_reader",
            default_value="false",
            description="Start the serial reader; leave false when /servo_angles already exists.",
        ),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        Node(
            package="franka_telep",
            executable="zhonglin_servo_reader",
            name="zhonglin_servo_reader",
            output="screen",
            parameters=[config_file],
            condition=IfCondition(start_servo_reader),
        ),
        Node(
            package="franka_telep",
            executable="urdf_joint_state",
            name="urdf_joint_state",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace="teleop_preview",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
            }],
            remappings=[
                ("joint_states", "/teleop_preview/joint_states"),
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_teleop_preview_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(start_rviz),
        ),
    ])
