"""Launch AnyGrasp RGB-D inference and the FR3 grasp runtime."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        DeclareLaunchArgument("launch_sensor", default_value="true"),
        DeclareLaunchArgument("publish_handeye_tf", default_value="true"),
        DeclareLaunchArgument("handeye_method", default_value="best"),
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("policy_port", default_value="8000"),
        DeclareLaunchArgument(
            "policy_python_executable",
            default_value="/home/young/miniconda3/envs/openvla/bin/python",
            description="Python executable containing AnyGrasp and its CUDA dependencies.",
        ),
        DeclareLaunchArgument("start_policy_server", default_value="true"),
        DeclareLaunchArgument("start_policy_runtime", default_value="true"),
        DeclareLaunchArgument(
            "execute_grasp",
            default_value="false",
            description="Actually move the robot. False performs perception and TF dry run only.",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/eye_to_hand_camera/eye_to_hand_camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/eye_to_hand_camera/eye_to_hand_camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/eye_to_hand_camera/eye_to_hand_camera/color/camera_info",
        ),
        DeclareLaunchArgument("enable_depth", default_value="true",
                              description="Enable depth stream."),
        DeclareLaunchArgument("align_depth", default_value="true",
                              description="Enable aligned depth to color."),
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
            "controller_mode": "trajectory",
        }.items(),
    )

    camera_providers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "launch",
                "eye_to_hand_realsense.launch.py",
            ])
        ]),
        launch_arguments={
            "enable_depth": LaunchConfiguration("enable_depth"),
            "align_depth": LaunchConfiguration("align_depth"),
            "publish_handeye_tf": LaunchConfiguration("publish_handeye_tf"),
            "handeye_method": LaunchConfiguration("handeye_method"),
        }.items(),
    )

    policy_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("policy_server"),
                "launch",
                "policy_server.launch.py",
            ])
        ]),
        launch_arguments={
            "backend": "anygrasp",
            "host": LaunchConfiguration("policy_host"),
            "port": LaunchConfiguration("policy_port"),
            "python_executable": LaunchConfiguration("policy_python_executable"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_policy_server")),
    )

    runtime = Node(
        package="franka_policy_runtime",
        executable="anygrasp_runtime",
        name="anygrasp_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "anygrasp_runtime.yaml",
            ]),
            {
                "policy_url": PythonExpression([
                    "'http://",
                    LaunchConfiguration("policy_host"),
                    ":",
                    LaunchConfiguration("policy_port"),
                    "/act'",
                ]),
                "image_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "execute_grasp": LaunchConfiguration("execute_grasp"),
            },
        ],
        condition=IfCondition(LaunchConfiguration("start_policy_runtime")),
    )

    return LaunchDescription(args + [
        robot_base,
        camera_providers,
        policy_server,
        runtime,
    ])
