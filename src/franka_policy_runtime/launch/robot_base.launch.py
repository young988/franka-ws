"""Launch the robot base stack (no policy, no sensors, no RViz).

This launch owns the minimal graph needed to control the FR3 arm:
robot_state_publisher, ros2_control with the Cartesian pose controller,
Franka gripper, and joint state aggregation.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    namespace = LaunchConfiguration("namespace")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    load_gripper = LaunchConfiguration("load_gripper")

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.urdf.xacro",
    )
    robot_description_config = Command([
        FindExecutable(name="xacro"),
        " ",
        franka_xacro_file,
        " hand:=",
        load_gripper,
        " robot_ip:=",
        robot_ip,
        " use_fake_hardware:=",
        use_fake_hardware,
        " fake_sensor_commands:=",
        fake_sensor_commands,
        " ros2_control:=true",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
        output="screen",
        parameters=[robot_description],
    )

    controllers_yaml = os.path.join(
        get_package_share_directory("franka_policy_controller"),
        "config",
        "franka_bringup_cartesian_pose_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        parameters=[
            controllers_yaml,
            robot_description,
        ],
        remappings=[("joint_states", "franka/joint_states")],
        output={"stdout": "screen", "stderr": "screen"},
        on_exit=Shutdown(),
    )

    controller_spawners = [
        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                controller,
                "--controller-manager-timeout",
                "60",
                "--controller-manager",
                PathJoinSubstitution([namespace, "controller_manager"]),
            ],
            output="screen",
        )
        for controller in ["joint_state_broadcaster", "franka_cartesian_pose_controller"]
    ]

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[{
            "source_list": ["franka/joint_states", "fr3_gripper/joint_states"],
            "rate": 30,
        }],
        output="screen",
    )

    franka_robot_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["franka_robot_state_broadcaster"],
        output="screen",
        condition=UnlessCondition(use_fake_hardware),
    )

    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_gripper"),
                "launch",
                "gripper.launch.py",
            ])
        ]),
        launch_arguments={
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "namespace": namespace,
        }.items(),
        condition=IfCondition(load_gripper),
    )

    return [
        robot_state_publisher,
        ros2_control_node,
        joint_state_publisher,
        franka_robot_state_broadcaster,
        gripper_launch,
    ] + controller_spawners


def generate_launch_description():
    return LaunchDescription(
        [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2", description="FR3 robot IP address (use 192.168.0.100 for fake hardware)."),
        DeclareLaunchArgument("namespace", default_value="", description="ROS namespace to launch the robot stack in (empty = no namespace)."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false", description="Run mock hardware interfaces instead of connecting to a real FR3."),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false", description="Publish simulated joint state / sensor data from fake hardware."),
        DeclareLaunchArgument("load_gripper", default_value="true", description="Include the Franka gripper in the robot description and launch its driver."),
        OpaqueFunction(function=launch_setup),
    ],)
