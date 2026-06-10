"""Preview episode replay in RViz (no real robot, no hardware).

Usage:
    ros2 launch franka_telep episode_replay_preview.launch.py episode_id:=0
    ros2 launch franka_telep episode_replay_preview.launch.py episode_id:=0 speed:=0.5
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("franka_telep"))
    robot_description = (pkg_share / "urdf" / "fr3_teleop_preview.urdf").read_text()
    config_file = str(pkg_share / "config" / "franka_telep.yaml")
    rviz_config = str(pkg_share / "rviz" / "fr3_teleop_preview.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("episode_id", default_value="0"),
        DeclareLaunchArgument("episode_path", default_value=""),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="true"),

        # ── replay node → /teleop_preview/joint_states ──────────
        Node(
            package="franka_telep",
            executable="episode_replay",
            name="episode_replay",
            output="screen",
            parameters=[config_file, {
                "episode_id": LaunchConfiguration("episode_id"),
                "episode_path": LaunchConfiguration("episode_path"),
                "speed": LaunchConfiguration("speed"),
                "loop": LaunchConfiguration("loop"),
                "mode": "impedance",
                "publish_rate_hz": 50.0,
                "leader_joint_state_topic": "/teleop_preview/joint_states",
            }],
        ),

        # ── robot_state_publisher ← /teleop_preview/joint_states ─
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace="teleop_preview",
            output="screen",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "/teleop_preview/joint_states")],
        ),

        # ── RViz ────────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_replay_preview_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
        ),
    ])
