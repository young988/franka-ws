"""Axis test launch — robot base + axis_test backend + axis_test runtime.

No camera, no handeye TF, no RViz. Micro-motion in one DOF at 2 Hz.

Usage:
    ros2 launch franka_policy_runtime axis_test.launch.py dimension:=x max_steps:=20
    ros2 launch franka_policy_runtime axis_test.launch.py dimension:=rz max_steps:=10
    ros2 launch franka_policy_runtime axis_test.launch.py dimension:=y action_scale:=1.0
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _default_python_executable():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(conda_python):
            return conda_python
    return sys.executable


def generate_launch_description():
    # ── launch arguments ──────────────────────────────────────────
    args = [
        # robot base (passed through)
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2",
                              description="FR3 robot IP address."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false",
                              description="Run mock hardware interfaces instead of a real FR3."),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Include Franka gripper."),
        # axis test backend
        DeclareLaunchArgument("dimension", default_value="x",
                              description="Axis to test: x, y, z, rx, ry, rz."),
        DeclareLaunchArgument("max_steps", default_value="20",
                              description="Maximum micro-motion steps before auto-stop."),
        # policy server
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1",
                              description="Policy server host address."),
        DeclareLaunchArgument("policy_port", default_value="8000",
                              description="Policy server port."),
        DeclareLaunchArgument(
            "server_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("policy_server"),
                "config",
                "policy_server.yaml",
            ]),
            description="Path to the policy server YAML config file.",
        ),
        # runtime
        DeclareLaunchArgument("action_scale", default_value="1.0",
                              description="Action scaling factor (1.0 = 1cm/1° effective per step)."),
        DeclareLaunchArgument("start_policy_server", default_value="false",
                              description="Launch the policy server subprocess."),
        DeclareLaunchArgument("start_policy_runtime", default_value="true",
                              description="Launch the axis test runtime node."),
        DeclareLaunchArgument(
            "python_executable",
            default_value=_default_python_executable(),
            description="Python executable for the policy server subprocess.",
        ),
    ]

    # ── robot base (no camera, no handeye TF, no RViz) ────────────
    robot_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "launch",
                "robot_base.launch.py",
            ])
        ]),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "load_gripper": LaunchConfiguration("load_gripper"),
        }.items(),
    )

    # ── policy server (axis_test backend) ─────────────────────────
    policy_server = ExecuteProcess(
        cmd=[
            LaunchConfiguration("python_executable"),
            "-m", "policy_server.server",
            "--config", LaunchConfiguration("server_config"),
            "--backend", "axis_test",
            "--host", LaunchConfiguration("policy_host"),
            "--port", LaunchConfiguration("policy_port"),
        ],
        env={
            "AXIS_TEST_DIMENSION": LaunchConfiguration("dimension"),
            "AXIS_TEST_MAX_STEPS": LaunchConfiguration("max_steps"),
        },
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_policy_server")),
    )

    # ── axis test runtime node ────────────────────────────────────
    axis_test_runtime = Node(
        package="franka_policy_runtime",
        executable="axis_test_runtime",
        name="axis_test_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "franka_policy_runtime.yaml",
            ]),
            {
                "mode": "single_step",
                "control_period_sec": 0.5,       # 2 Hz
                "policy_url": PythonExpression([
                    "'http://",
                    LaunchConfiguration("policy_host"),
                    ":",
                    LaunchConfiguration("policy_port"),
                    "/act'",
                ]),
                "action_scale": LaunchConfiguration("action_scale"),
            },
        ],
        condition=IfCondition(LaunchConfiguration("start_policy_runtime")),
    )

    return LaunchDescription(args + [robot_base, policy_server, axis_test_runtime])
