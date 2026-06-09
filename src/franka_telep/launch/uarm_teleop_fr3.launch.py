"""Launch uArm to FR3 teleoperation using joint-impedance tracking.

Architecture
------------
  uArm (Zhonglin servos)          FR3
  zhonglin_servo_reader      ->   robot_state_publisher
  /servo_absolute_angles     ->   ros2_control_node
  uarm_leader_publisher      ->   follower_controller
  /uarm_leader/joint_states  ->   effort command

Phase 0 (optional): franka_home_initializer sends FR3 to home via
                    fr3_arm_controller, then triggers a controller
                    switch to follower_controller.

Phase 1: follower_controller tracks /uarm_leader/joint_states
         with joint-impedance (PD) at 1000 Hz controller update
         rate. There is no trajectory interpolation or action overhead.
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
    TimerAction,
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
    robot_ip = LaunchConfiguration("robot_ip")
    namespace = LaunchConfiguration("namespace")
    load_gripper = LaunchConfiguration("load_gripper")
    config_file = LaunchConfiguration("config_file")
    use_home_init = LaunchConfiguration("use_home_init")
    servo_port = LaunchConfiguration("servo_port")
    namespace_value = namespace.perform(context).strip("/")
    switch_controller_service = (
        f"/{namespace_value}/controller_manager/switch_controller"
        if namespace_value
        else "/controller_manager/switch_controller"
    )

    # FR3 robot description.
    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.urdf.xacro",
    )
    robot_description_config = Command([
        FindExecutable(name="xacro"), " ", franka_xacro_file,
        " hand:=", load_gripper,
        " robot_ip:=", robot_ip,
        " use_fake_hardware:=false",
        " fake_sensor_commands:=false",
        " ros2_control:=true",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    controllers_yaml_path = os.path.join(
        get_package_share_directory("franka_telep"),
        "config",
        "uarm_teleop_controllers.yaml",
    )

    # Nodes.
    zhonglin_servo_reader = Node(
        package="franka_telep",
        executable="zhonglin_servo_reader",
        name="zhonglin_servo_reader",
        output="screen",
        parameters=[config_file, {"port": servo_port}],
    )

    uarm_leader_publisher = Node(
        package="franka_telep",
        executable="uarm_leader_publisher",
        name="uarm_leader_publisher",
        output="screen",
        parameters=[config_file],
    )

    franka_home_initializer = Node(
        package="franka_telep",
        executable="franka_home_initializer",
        name="franka_home_initializer",
        output="screen",
        parameters=[config_file],
        condition=IfCondition(use_home_init),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
        output="screen",
        parameters=[robot_description],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        parameters=[robot_description, controllers_yaml_path],
        remappings=[("joint_states", "franka/joint_states")],
        output={"stdout": "screen", "stderr": "screen"},
        on_exit=Shutdown(),
    )

    # Joint / franka state broadcasters are always active.
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "30"],
        output="screen",
    )
    franka_robot_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["franka_robot_state_broadcaster"],
        output="screen",
    )

    # Homing phase: start fr3_arm_controller first
    fr3_arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["fr3_arm_controller", "--controller-manager-timeout", "30"],
        output="screen",
        condition=IfCondition(use_home_init),
    )

    # With homing enabled, follower_controller is loaded inactive and the
    # switch command below activates it after fr3_arm_controller finishes home.
    follower_spawner_inactive = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=[
            "follower_controller",
            "--controller-manager-timeout", "30",
            "--inactive",
        ],
        output="screen",
        condition=IfCondition(use_home_init),
    )

    # Without homing, start follower_controller immediately.
    follower_spawner_active = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=[
            "follower_controller",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
        condition=UnlessCondition(use_home_init),
    )

    # After the home trajectory has finished (~ home_duration + margin),
    # switch to the follower controller by deactivating fr3_arm_controller
    # and activating follower_controller.
    switch_cmd = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[[
                    FindExecutable(name="ros2"),
                    " service call ",
                    switch_controller_service,
                    " ",
                    "controller_manager_msgs/srv/SwitchController ",
                    "\"{activate_controllers: [follower_controller], ",
                    "deactivate_controllers: [fr3_arm_controller], ",
                    "strictness: 2, activate_asap: true, ",
                    'timeout: {sec: 3, nanosec: 0}}"',
                ]],
                shell=True,
                name="switch_to_follower_controller",
                output="both",
                condition=IfCondition(use_home_init),
            ),
        ],
        condition=IfCondition(use_home_init),
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[{
            "source_list": ["franka/joint_states", "franka_gripper/joint_states"],
            "rate": 30,
        }],
        output="screen",
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
            "use_fake_hardware": "false",
            "namespace": namespace,
        }.items(),
        condition=IfCondition(load_gripper),
    )

    return [
        zhonglin_servo_reader,
        uarm_leader_publisher,
        robot_state_publisher,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        franka_robot_state_spawner,
        fr3_arm_spawner,
        follower_spawner_inactive,
        follower_spawner_active,
        franka_home_initializer,
        switch_cmd,
        joint_state_publisher,
        gripper_launch,
    ]


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("franka_telep"),
        "config",
        "franka_telep.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip", default_value="172.16.0.2",
            description="FR3 robot IP address",
        ),
        DeclareLaunchArgument(
            "namespace", default_value="",
            description="ROS namespace for the robot stack",
        ),
        DeclareLaunchArgument(
            "load_gripper", default_value="true",
            description="Include Franka gripper driver",
        ),
        DeclareLaunchArgument(
            "config_file", default_value=default_config,
            description="Path to franka_telep.yaml (servo + mapping params)",
        ),
        DeclareLaunchArgument(
            "servo_port", default_value="/dev/ttyUSB0",
            description="Serial port for the Zhonglin/uArm servo bus",
        ),
        DeclareLaunchArgument(
            "use_home_init", default_value="true",
            description="Run franka_home_initializer and switch controllers after homing",
        ),
        OpaqueFunction(function=launch_setup),
    ])
