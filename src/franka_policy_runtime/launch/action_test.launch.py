"""Action dimension test launch.

Robot base + action_test_runtime node.  No sensors, no policy server,
no handeye — pure robot stack with the test runtime injecting hard-coded
actions.

Usage:
    ros2 launch franka_policy_runtime action_test.launch.py
    ros2 launch franka_policy_runtime action_test.launch.py \
        load_gripper:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2",
                              description="FR3 robot IP address (use 192.168.0.100 for fake hardware)."),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Include Franka gripper in robot description and launch driver."),
        DeclareLaunchArgument("step_interval_sec", default_value="2.0",
                              description="Wait time between test steps (seconds)."),
        DeclareLaunchArgument("action_scale", default_value="0.5",
                              description="Multiplier applied to action delta before IK."),
        DeclareLaunchArgument("tolerance_pos_m", default_value="0.01",
                              description="Max acceptable position error for OK flag."),
        DeclareLaunchArgument("csv_output_path", default_value="",
                              description="If set, write results CSV to this path."),
    ]

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
            "load_gripper": LaunchConfiguration("load_gripper"),
        }.items(),
    )

    action_test = Node(
        package="franka_policy_runtime",
        executable="action_test",
        name="action_test_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "action_test.yaml",
            ]),
            {
                "step_interval_sec": LaunchConfiguration("step_interval_sec"),
                "action_scale": LaunchConfiguration("action_scale"),
                "tolerance_pos_m": LaunchConfiguration("tolerance_pos_m"),
                "csv_output_path": LaunchConfiguration("csv_output_path"),
            },
        ],
    )

    return LaunchDescription(args + [robot_base, action_test])
