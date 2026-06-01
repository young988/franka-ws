"""Launch policy_server as a standalone inference process."""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _default_python_executable():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(conda_python):
            return conda_python
    return sys.executable


def generate_launch_description():
    return LaunchDescription(
        [
        DeclareLaunchArgument(
            "config",
            default_value=PathJoinSubstitution([
                FindPackageShare("policy_server"),
                "config",
                "policy_server.yaml",
            ]),
            description="Path to the policy server YAML config file.",
        ),
        DeclareLaunchArgument("backend", default_value="openvla", description="Policy backend type: openvla, bc_isaaclab_stack, dummy, or python_plugin class path."),
        DeclareLaunchArgument("host", default_value="127.0.0.1", description="Server bind address."),
        DeclareLaunchArgument("port", default_value="8000", description="Server port."),
        DeclareLaunchArgument(
            "python_executable",
            default_value=_default_python_executable(),
            description="Python executable used for the standalone policy server process.",
        ),
        ExecuteProcess(
            cmd=[
                LaunchConfiguration("python_executable"),
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
    ],)
