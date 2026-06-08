"""Launch the robot base stack (no policy, no sensors, no RViz).

This launch owns the minimal graph needed to control the FR3 arm:
robot_state_publisher, ros2_control with fr3_arm_controller, MoveIt
move_group for IK, Franka gripper, joint state aggregation.

Design: other launches (bc_cube_stack, vla_policy) include this via
IncludeLaunchDescription and append their own sensors + inference
nodes. This file has NO knowledge of cameras, handeye calibration,
observers, policy backends, or visualisation tools.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition
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
    load_gripper = LaunchConfiguration("load_gripper")
    kinematics_solver_timeout = LaunchConfiguration("kinematics_solver_timeout")

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
        " use_fake_hardware:=false"
        " fake_sensor_commands:=false"
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
    if "fr3_arm" in kinematics_yaml:
        kinematics_yaml["fr3_arm"]["kinematics_solver_timeout"] = float(
            kinematics_solver_timeout.perform(context)
        )
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
        get_package_share_directory("franka_policy_runtime"),
        "config",
        "franka_bringup_policy_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        parameters=[
            robot_description,
            controllers_yaml,
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
        for controller in ["fr3_arm_controller", "joint_state_broadcaster"]
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
        robot_state_publisher,
        ros2_control_node,
        joint_state_publisher,
        franka_robot_state_broadcaster,
        gripper_launch,
        move_group_node,
    ] + controller_spawners


def generate_launch_description():
    return LaunchDescription(
        [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2", description="FR3 robot IP address (use 192.168.0.100 for fake hardware)."),
        DeclareLaunchArgument("namespace", default_value="", description="ROS namespace to launch the robot stack in (empty = no namespace)."),
        DeclareLaunchArgument("load_gripper", default_value="true", description="Include the Franka gripper in the robot description and launch its driver."),
        DeclareLaunchArgument("kinematics_solver_timeout", default_value="0.1", description="MoveIt IK solver timeout override in seconds."),
        OpaqueFunction(function=launch_setup),
    ],)
