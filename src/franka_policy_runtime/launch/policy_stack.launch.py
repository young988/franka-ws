"""Launch the full Franka policy stack.

This launch owns the full graph needed by the policy runtime:
robot_state_publisher, ros2_control with franka_policy_controller, MoveIt
move_group for IK, Franka gripper, RealSense, policy server, and the policy
runtime node.

Defaults are set for real hardware (FR3). Use the launch arguments to toggle
individual components off (e.g. start_policy_server:=false).
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, relative_path):
    path = os.path.join(get_package_share_directory(package_name), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fr3_ompl_config():
    config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/ResolveConstraintFrames "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_yaml = load_yaml("franka_fr3_moveit_config", "config/ompl_planning.yaml")
    if "fr3_arm" not in ompl_yaml and "panda_arm" in ompl_yaml:
        ompl_yaml["fr3_arm"] = ompl_yaml["panda_arm"]
    config["move_group"].update(ompl_yaml)
    return config


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

    srdf_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.srdf.xacro",
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command([FindExecutable(name="xacro"), " ", srdf_xacro_file, " hand:=", load_gripper]),
            value_type=str,
        )
    }

    kinematics_yaml = load_yaml("franka_fr3_moveit_config", "config/kinematics.yaml")
    planning_config = fr3_ompl_config()
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_config,
            planning_scene_monitor_parameters,
        ],
    )

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
        "franka_bringup_policy_controllers.yaml",
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
        for controller in ["joint_state_broadcaster", "franka_policy_controller"]
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

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("franka_fr3_moveit_config"),
                "rviz",
                "moveit.rviz",
            ),
        ],
        parameters=[robot_description, robot_description_semantic, planning_config, kinematics_yaml],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return [
        robot_state_publisher,
        ros2_control_node,
        joint_state_publisher,
        franka_robot_state_broadcaster,
        gripper_launch,
        move_group_node,
        rviz_node,
    ] + controller_spawners


def generate_launch_description():
    policy_runtime = Node(
        package="franka_policy_runtime",
        executable="policy_runtime_node",
        name="franka_policy_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "franka_policy_runtime.yaml",
            ]),
            {
                "policy_url": PythonExpression([
                    "'http://",
                    LaunchConfiguration("policy_host"),
                    ":",
                    LaunchConfiguration("policy_port"),
                    "/act'",
                ]),
                "mode": LaunchConfiguration("policy_mode"),
                "image_topic": LaunchConfiguration("image_topic"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "gripper_move_action": LaunchConfiguration("gripper_move_action"),
            },
        ],
        condition=IfCondition(LaunchConfiguration("start_policy_runtime")),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ]),
        launch_arguments={
            "camera_name": "camera",
            "camera_namespace": "camera",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "publish_tf": "true",
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_sensor")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("launch_sensor", default_value="true"),
        DeclareLaunchArgument("start_policy_runtime", default_value="true"),
        DeclareLaunchArgument("start_policy_server", default_value="true"),
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("policy_port", default_value="8000"),
        DeclareLaunchArgument("policy_mode", default_value="single_step"),
        DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument("gripper_move_action", default_value="/franka_gripper/move"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("policy_server"),
                    "launch",
                    "policy_server.launch.py",
                ])
            ]),
            launch_arguments={
                "backend": "openvla",
                "host": LaunchConfiguration("policy_host"),
                "port": LaunchConfiguration("policy_port"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("start_policy_server")),
        ),
        realsense_launch,
        OpaqueFunction(function=launch_setup),
        policy_runtime,
    ])
