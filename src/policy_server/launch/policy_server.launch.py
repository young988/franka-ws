"""Launch policy_server as a standalone inference process."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=PathJoinSubstitution([
                FindPackageShare("policy_server"),
                "config",
                "policy_server.yaml",
            ]),
        ),
        DeclareLaunchArgument("backend", default_value="openvla"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="8000"),
        ExecuteProcess(
            cmd=[
                "python3",
                "-m",
                "policy_server.server",
                "--config",
                LaunchConfiguration("config"),
                "--backend",
                LaunchConfiguration("backend"),
                "--host",
                LaunchConfiguration("host"),
                "--port",
                LaunchConfiguration("port"),
            ],
            output="screen",
        ),
    ])
