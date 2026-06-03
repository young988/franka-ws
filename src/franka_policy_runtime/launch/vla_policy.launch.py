"""VLA (OpenVLA) policy launch.

Robot base + eye-to-hand RealSense + handeye TF + policy_server (OpenVLA
backend) + policy_runtime_node (OpenVLAObserver).

Usage:
    ros2 launch franka_policy_runtime vla_policy.launch.py
    ros2 launch franka_policy_runtime vla_policy.launch.py \
        instruction:="put the cube in the bin"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

EYE_TO_HAND_CAMERA = "eye_to_hand_camera"
EYE_TO_HAND_SERIAL = "420122071571"


def generate_launch_description():
    # ── launch arguments ──────────────────────────────────────────
    args = [
        # robot base (passed through)
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2", description="FR3 robot IP address (use 192.168.0.100 for fake hardware)."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false", description="Run mock hardware interfaces instead of a real FR3."),
        DeclareLaunchArgument("load_gripper", default_value="true", description="Include Franka gripper in robot description and launch driver."),
        DeclareLaunchArgument("launch_sensor", default_value="true", description="Launch the eye-to-hand RealSense D435i camera."),
        # VLA observer
        DeclareLaunchArgument("instruction", default_value="pick up the cube", description="Text instruction for the OpenVLA model (e.g. 'pick up the cube', 'put the cube in the bin')."),
        DeclareLaunchArgument("unnorm_key", default_value="bridge_orig", description="Action unnormalization key for the OpenVLA model (dataset-specific, e.g. 'bridge_orig', 'fractal')."),
        # policy runtime
        DeclareLaunchArgument("policy_mode", default_value="single_step", description="Action scheduling mode: single_step (wait per action), chunk_async (overlap fuse), or streaming (replace all)."),
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1", description="Policy server host address."),
        DeclareLaunchArgument("policy_port", default_value="8000", description="Policy server port."),
        DeclareLaunchArgument("start_policy_server", default_value="true", description="Launch the policy server subprocess from this launch file."),
        DeclareLaunchArgument("start_policy_runtime", default_value="true", description="Launch the policy runtime node from this launch file."),
        # handeye
        DeclareLaunchArgument("publish_handeye_tf", default_value="true", description="Publish the hand-eye transform from pre-calibrated parameters."),
        DeclareLaunchArgument("handeye_method", default_value="best", description="Hand-eye solver method: tsai, park, dornaika, hoda, daniilidis, or best (auto-select)."),
        # topics
        DeclareLaunchArgument(
            "image_topic",
            default_value=f"/{EYE_TO_HAND_CAMERA}/{EYE_TO_HAND_CAMERA}/color/image_raw",
            description="Color image topic for the VLA observer.",
        ),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states", description="JointState topic for robot state observation."),
        DeclareLaunchArgument("gripper_move_action", default_value="/franka_gripper/move", description="Franka gripper Move action topic."),
    ]

    # ── robot base ─────────────────────────────────────────────────
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

    # ── eye-to-hand RealSense camera ───────────────────────────────
    realsense_launch = IncludeLaunchDescription(
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
            "align_depth.enable": "false",
            "rgb_camera.color_profile": "1280,720,30",
            "pointcloud.enable": "false",
            "publish_tf": "true",
            "base_frame_id": f"{EYE_TO_HAND_CAMERA}_link",
            "tf_prefix": "",
        }.items(),
    )

    # ── handeye TF publisher ──────────────────────────────────────
    handeye_tf = Node(
        package="handeye_calibration",
        executable="handeye_tf_publisher",
        name="eye_to_hand_handeye_tf_publisher",
        output="screen",
        parameters=[{
            "calibration_setup": "eye_to_hand",
            "method": LaunchConfiguration("handeye_method"),
            "child_frame": f"{EYE_TO_HAND_CAMERA}_{EYE_TO_HAND_CAMERA}_link",
            "optical_frame": f"{EYE_TO_HAND_CAMERA}_color_optical_frame",
        }],
        condition=IfCondition(LaunchConfiguration("publish_handeye_tf")),
    )

    # ── policy server ─────────────────────────────────────────────
    policy_server = IncludeLaunchDescription(
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
    )

    # ── policy runtime node ───────────────────────────────────────
    policy_runtime = Node(
        package="franka_policy_runtime",
        executable="vla_policy_runtime",
        name="vla_policy_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "franka_policy_runtime.yaml",
            ]),
            {
                "instruction": LaunchConfiguration("instruction"),
                "unnorm_key": LaunchConfiguration("unnorm_key"),
                "mode": LaunchConfiguration("policy_mode"),
                "policy_url": PythonExpression([
                    "'http://",
                    LaunchConfiguration("policy_host"),
                    ":",
                    LaunchConfiguration("policy_port"),
                    "/act'",
                ]),
                "image_topic": LaunchConfiguration("image_topic"),
                "object_pose_provider": "none",
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "gripper_move_action": LaunchConfiguration("gripper_move_action"),
            },
        ],
        condition=IfCondition(LaunchConfiguration("start_policy_runtime")),
    )

    return LaunchDescription(
        args
        + [
            robot_base,
            GroupAction(
                actions=[realsense_launch],
                condition=IfCondition(LaunchConfiguration("launch_sensor")),
                scoped=True,
                forwarding=False,
            ),
            handeye_tf,
            policy_server,
            policy_runtime,
        ],
    )
