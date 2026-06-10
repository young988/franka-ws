"""Launch FR3 uArm teleoperation with OpenVLA demonstration collection."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


EYE_TO_HAND_CAMERA = "eye_to_hand_camera"
EYE_TO_HAND_SERIAL = "420122071571"


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        DeclareLaunchArgument("servo_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("use_home_init", default_value="true"),
        DeclareLaunchArgument("launch_sensor", default_value="true"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("dataset_root", default_value="~/franka_openvla_data"),
        DeclareLaunchArgument("dataset_name", default_value="franka_teleop"),
        DeclareLaunchArgument("instruction", default_value="pick up the object"),
        DeclareLaunchArgument("sample_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("image_size", default_value="256"),
        DeclareLaunchArgument(
            "image_topic",
            default_value=f"/{EYE_TO_HAND_CAMERA}/{EYE_TO_HAND_CAMERA}/color/image_raw",
        ),
        DeclareLaunchArgument("wrist_image_topic", default_value=""),
    ]

    teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_telep"),
                "launch",
                "uarm_teleop_fr3.launch.py",
            ])
        ]),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "servo_port": LaunchConfiguration("servo_port"),
            "use_home_init": LaunchConfiguration("use_home_init"),
        }.items(),
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ]),
        launch_arguments={
            "camera_name": EYE_TO_HAND_CAMERA,
            "camera_namespace": EYE_TO_HAND_CAMERA,
            "serial_no": f"'{EYE_TO_HAND_SERIAL}'",
            "enable_color": "true",
            "enable_depth": "false",
            "enable_infra": "false",
            "enable_sync": "true",
            "rgb_camera.color_profile": "640,480,15",
            "rgb_camera.power_line_frequency": "2",
            "pointcloud.enable": "false",
            "publish_tf": "true",
            "base_frame_id": f"{EYE_TO_HAND_CAMERA}_link",
            "tf_prefix": "",
        }.items(),
    )

    recorder = Node(
        package="franka_telep",
        executable="openvla_dataset_recorder",
        name="openvla_dataset_recorder",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_telep"),
                "config",
                "franka_telep.yaml",
            ]),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "wrist_image_topic": LaunchConfiguration("wrist_image_topic"),
                "dataset_root": LaunchConfiguration("dataset_root"),
                "dataset_name": LaunchConfiguration("dataset_name"),
                "instruction": LaunchConfiguration("instruction"),
                "sample_rate_hz": ParameterValue(
                    LaunchConfiguration("sample_rate_hz"), value_type=float
                ),
                "image_size": ParameterValue(
                    LaunchConfiguration("image_size"), value_type=int
                ),
                "auto_start": ParameterValue(
                    LaunchConfiguration("auto_start"), value_type=bool
                ),
                "gripper_command_topic": "/uarm_leader/gripper_command",
            },
        ],
    )

    return LaunchDescription(
        arguments
        + [
            teleop,
            GroupAction(
                actions=[camera],
                condition=IfCondition(LaunchConfiguration("launch_sensor")),
                scoped=True,
                forwarding=False,
            ),
            recorder,
        ]
    )
