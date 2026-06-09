"""BC Cube Stack policy launch.

Robot base + eye-to-hand RealSense + handeye TF + policy_server (BC
backend) + policy_runtime_node (IsaacLabStackBCObserver with
ColorCubeStackObjectProvider).

Usage:
    ros2 launch franka_policy_runtime bc_cube_stack.launch.py
    ros2 launch franka_policy_runtime bc_cube_stack.launch.py \
        load_gripper:=false
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
        DeclareLaunchArgument("load_gripper", default_value="true", description="Include Franka gripper in robot description and launch driver."),
        DeclareLaunchArgument("launch_sensor", default_value="true", description="Launch the eye-to-hand RealSense D435i camera (color + depth)."),
        # BC observer / perception
        DeclareLaunchArgument("object_pose_provider", default_value="color_cube", description="Object pose provider type: color_cube (color-based detection) or none."),
        DeclareLaunchArgument("object_target_color", default_value="red", description="Target cube color for color-based object detection (red, green, blue, yellow)."),
        DeclareLaunchArgument("object_camera_frame", default_value="eye_to_hand_camera_color_optical_frame", description="Camera optical frame for pixel-to-3D object pose estimation."),
        DeclareLaunchArgument("object_min_pixels", default_value="30", description="Minimum pixel area for a valid object detection (filters noise)."),
        # policy runtime
        DeclareLaunchArgument("policy_host", default_value="127.0.0.1", description="Policy server host address."),
        DeclareLaunchArgument("policy_port", default_value="8000", description="Policy server port."),
        DeclareLaunchArgument("start_policy_server", default_value="true", description="Launch the policy server subprocess from this launch file."),
        DeclareLaunchArgument("start_policy_runtime", default_value="true", description="Launch the policy runtime node from this launch file."),
        DeclareLaunchArgument("control_mode", default_value="cartesian_delta", description="Policy output mode: cartesian_delta or joint_position."),
        # handeye
        DeclareLaunchArgument("publish_handeye_tf", default_value="true", description="Publish the hand-eye transform from pre-calibrated parameters."),
        DeclareLaunchArgument("handeye_method", default_value="best", description="Hand-eye solver method: tsai, park, dornaika, hoda, daniilidis, or best (auto-select)."),
        # topic overrides (derived from camera name, but user-overridable)
        DeclareLaunchArgument(
            "image_topic",
            default_value=f"/{EYE_TO_HAND_CAMERA}/{EYE_TO_HAND_CAMERA}/color/image_raw",
            description="Color image topic for the BC observer.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value=f"/{EYE_TO_HAND_CAMERA}/{EYE_TO_HAND_CAMERA}/aligned_depth_to_color/image_raw",
            description="Depth image topic (aligned to color) for object pose estimation.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value=f"/{EYE_TO_HAND_CAMERA}/{EYE_TO_HAND_CAMERA}/color/camera_info",
            description="CameraInfo topic (intrinsics) for pixel-to-3D projection.",
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
            "load_gripper": LaunchConfiguration("load_gripper"),
            "controller_mode": LaunchConfiguration("control_mode"),
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
            "enable_depth": "true",
            "enable_infra": "false",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "rgb_camera.color_profile": "1280,720,30",
            "depth_module.depth_profile": "1280,720,30",
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
            "backend": "bc_isaaclab_stack",
            "host": LaunchConfiguration("policy_host"),
            "port": LaunchConfiguration("policy_port"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_policy_server")),
    )

    # ── policy runtime node ───────────────────────────────────────
    policy_runtime = Node(
        package="franka_policy_runtime",
        executable="bc_cube_stack_runtime",
        name="bc_cube_stack_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "franka_policy_runtime.yaml",
            ]),
            {
                "object_pose_provider": LaunchConfiguration("object_pose_provider"),
                "object_target_color": LaunchConfiguration("object_target_color"),
                "object_camera_frame": LaunchConfiguration("object_camera_frame"),
                "object_min_pixels": LaunchConfiguration("object_min_pixels"),
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
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "gripper_move_action": LaunchConfiguration("gripper_move_action"),
                "control_mode": LaunchConfiguration("control_mode"),
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
