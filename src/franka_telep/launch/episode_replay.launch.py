"""Replay a recorded teleop episode on FR3.

Flow:
    1. Read episode start position → send via JTC (fr3_arm_controller)
    2. Wait for convergence → switch to follower_controller
    3. Replay the rest via impedance tracking

Usage:
    ros2 launch franka_telep episode_replay.launch.py episode_id:=0 speed:=1.0
    ros2 launch franka_telep episode_replay.launch.py episode_id:=0 loop:=true
"""

import json
import os
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _episode_dir(context):
    path_str = LaunchConfiguration("episode_path").perform(context).strip()
    if path_str:
        return Path(path_str).expanduser()
    root = Path(LaunchConfiguration("data_root").perform(context)).expanduser()
    ep_id = int(LaunchConfiguration("episode_id").perform(context))
    return root / f"episode_{ep_id:06d}"


def _episode_start_position(context) -> list[float]:
    ep_dir = _episode_dir(context)
    data = np.load(str(ep_dir / "steps.npz"))
    return data["joint_positions"][0].tolist()


def launch_setup(context):
    robot_ip = LaunchConfiguration("robot_ip")
    namespace = LaunchConfiguration("namespace")
    load_gripper = LaunchConfiguration("load_gripper")
    namespace_val = namespace.perform(context).strip("/")
    switch_svc = f"/{namespace_val}/controller_manager/switch_controller" if namespace_val else "/controller_manager/switch_controller"

    start_joints = _episode_start_position(context)
    ep_dir = _episode_dir(context)
    ep_id = LaunchConfiguration("episode_id").perform(context)
    total_steps = len(np.load(str(ep_dir / "steps.npz"))["joint_positions"])
    meta = json.loads((ep_dir / "episode.json").read_text())

    print(f"[episode_replay.launch] episode_id={ep_id}  steps={total_steps}  "
          f"instruction={meta.get('instruction', 'N/A')}")
    print(f"[episode_replay.launch] start_position (rad): "
          f"{[round(v, 4) for v in start_joints]}")

    # ── robot description ─────────────────────────────────────────
    franka_xacro = os.path.join(
        get_package_share_directory("franka_description"), "robots", "fr3", "fr3.urdf.xacro")
    robot_description = {
        "robot_description": ParameterValue(
            Command([
                FindExecutable(name="xacro"), " ", franka_xacro,
                " hand:=", load_gripper,
                " robot_ip:=", robot_ip,
                " use_fake_hardware:=false",
                " fake_sensor_commands:=false",
                " ros2_control:=true",
            ]), value_type=str)
    }

    controllers_yaml = os.path.join(
        get_package_share_directory("franka_telep"), "config", "uarm_teleop_controllers.yaml")

    # ── robot core ─────────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        namespace=namespace, output="screen", parameters=[robot_description])

    ros2_control = Node(
        package="controller_manager", executable="ros2_control_node",
        namespace=namespace,
        parameters=[robot_description, controllers_yaml],
        remappings=[("joint_states", "franka/joint_states")],
        output={"stdout": "screen", "stderr": "screen"},
        on_exit=Shutdown())

    joint_state_broadcaster = Node(
        package="controller_manager", executable="spawner",
        namespace=namespace,
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "30"],
        output="screen")

    franka_robot_state_broadcaster = Node(
        package="controller_manager", executable="spawner",
        namespace=namespace,
        arguments=["franka_robot_state_broadcaster", "--controller-manager-timeout", "30"],
        output="screen")

    # Phase 1: fr3_arm_controller (active) — move to start
    fr3_arm_spawner = Node(
        package="controller_manager", executable="spawner",
        namespace=namespace,
        arguments=["fr3_arm_controller", "--controller-manager-timeout", "30"],
        output="screen")

    # Phase 2: follower_controller (inactive, activated after homing)
    follower_spawner = Node(
        package="controller_manager", executable="spawner",
        namespace=namespace,
        arguments=["follower_controller", "--controller-manager-timeout", "30", "--inactive"],
        output="screen")

    joint_state_publisher = Node(
        package="joint_state_publisher", executable="joint_state_publisher",
        namespace=namespace,
        parameters=[{"source_list": ["franka/joint_states", "franka_gripper/joint_states"], "rate": 30}],
        output="screen")

    gripper = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("franka_gripper"), "launch", "gripper.launch.py"])]),
        launch_arguments={"robot_ip": robot_ip, "use_fake_hardware": "false", "namespace": namespace}.items())

    # ── move-to-start (JTC) ────────────────────────────────────────
    home_duration = 8.0  # generous: slow move to start
    move_to_start = Node(
        package="franka_telep", executable="franka_home_initializer",
        name="franka_home_initializer", output="screen",
        parameters=[{
            "joint_state_topic": "/franka/joint_states",
            "trajectory_action": "/fr3_arm_controller/follow_joint_trajectory",
            "ready_topic": "/franka_teleop/home_ready",
            "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
            "home_joint_positions": start_joints,
            "trajectory_duration_sec": home_duration,
            "goal_tolerance_rad": 0.05,
            "ready_delay_sec": 0.5,
        }])

    # ── controller switch after homing ─────────────────────────────
    switch_cmd = TimerAction(
        period=home_duration + 2.0,
        actions=[ExecuteProcess(
            cmd=[[
                FindExecutable(name="ros2"), " service call ", switch_svc,
                " controller_manager_msgs/srv/SwitchController ",
                '"{activate_controllers: [follower_controller], ',
                'deactivate_controllers: [fr3_arm_controller], ',
                'strictness: 2, activate_asap: true, ',
                'timeout: {sec: 3, nanosec: 0}}"',
            ]],
            shell=True, name="switch_to_follower", output="both")])

    # ── replay (impedance mode) ────────────────────────────────────
    replay = Node(
        package="franka_telep", executable="episode_replay",
        name="episode_replay", output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("franka_telep"), "config", "franka_telep.yaml"]),
            {
                "episode_id": LaunchConfiguration("episode_id"),
                "speed": LaunchConfiguration("speed"),
                "loop": LaunchConfiguration("loop"),
                "mode": LaunchConfiguration("mode"),
                "require_home_ready": True,
            },
        ])

    return [
        robot_state_publisher, ros2_control,
        joint_state_broadcaster, franka_robot_state_broadcaster,
        fr3_arm_spawner, follower_spawner,
        joint_state_publisher, gripper,
        move_to_start, switch_cmd, replay,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        DeclareLaunchArgument("data_root", default_value="~/franka_openvla_data/franka_teleop/raw"),
        DeclareLaunchArgument("episode_id", default_value="0"),
        DeclareLaunchArgument("episode_path", default_value=""),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("mode", default_value="impedance"),
        DeclareLaunchArgument("loop", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
