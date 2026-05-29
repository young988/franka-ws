"""
One-click launch: Franka hardware + MoveIt + RealSense + pixel_to_robot.

Brings up:
  - Franka hardware interface + state broadcasters
  - MoveIt MoveGroup (planning) + RViz
  - RealSense D435i (aligned color + depth)
  - pixel_to_robot (click image -> plan + execute)

No OctoMap mapping — use manual_mapping.launch.py for that.

Usage:
    ros2 launch handeye_calibration pixel_to_robot.launch.py \
        robot_ip:=172.16.0.2 \
        board_type:=chessboard
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SAMPLE_ROOT = '/home/young/ros2_ws/src/handeye_calibration/samples'


def generate_launch_description():
    arg_robot_ip = DeclareLaunchArgument(
        'robot_ip', default_value='172.16.0.2',
        description='Franka robot IP / hostname')
    arg_use_fake_hardware = DeclareLaunchArgument(
        'use_fake_hardware', default_value='false',
        description='Use fake hardware (simulation)')
    arg_fake_sensor_commands = DeclareLaunchArgument(
        'fake_sensor_commands', default_value='false',
        description='Use fake sensor commands with fake hardware')
    arg_board_type = DeclareLaunchArgument(
        'board_type', default_value='chessboard',
        description='single_aruco, charuco, aruco_grid, or chessboard')
    arg_intrinsics_source = DeclareLaunchArgument(
        'intrinsics_source', default_value='camera_info',
        description='auto, camera_info, calibrated, file, or manual')
    arg_calibration_setup = DeclareLaunchArgument(
        'calibration_setup', default_value='eye_in_hand',
        description='eye_in_hand or eye_to_hand')
    arg_publish_handeye_tf = DeclareLaunchArgument(
        'publish_handeye_tf', default_value='true',
        description='Publish calibrated franka link8 to camera_link TF')
    arg_handeye_method = DeclareLaunchArgument(
        'handeye_method', default_value='best',
        description='Calibration row to publish: best or method name')
    arg_parent_frame = DeclareLaunchArgument(
        'parent_frame', default_value='fr3_link8',
        description='Parent frame for hand-eye TF, e.g. fr3_link8 or franka_link8')
    arg_child_frame = DeclareLaunchArgument(
        'child_frame', default_value='camera_link',
        description='Target frame for hand-eye TF')
    arg_optical_frame = DeclareLaunchArgument(
        'optical_frame', default_value='camera_color_optical_frame',
        description='Optical frame used during calibration (bridge to child_frame)')
    arg_enable_auto_grasp = DeclareLaunchArgument(
        'enable_auto_grasp', default_value='true',
        description='Automatically attempt a cautious grasp after '
                    'successful motion')
    arg_pregrasp_width = DeclareLaunchArgument(
        'pregrasp_width', default_value='0.08',
        description='Open gripper width before grasping')
    arg_grasp_speed = DeclareLaunchArgument(
        'grasp_speed', default_value='0.03',
        description='Gripper speed for pre-open and grasp')
    arg_grasp_force = DeclareLaunchArgument(
        'grasp_force', default_value='10.0',
        description='Gripper force for cautious grasp')
    arg_min_grasp_width = DeclareLaunchArgument(
        'min_grasp_width', default_value='0.005',
        description='Threshold used to classify empty grasp vs '
                    'object contact')
    arg_trajectory_action = DeclareLaunchArgument(
        'trajectory_action',
        default_value='fr3_arm_controller/follow_joint_trajectory',
        description='FollowJointTrajectory action exposed by the active arm controller')
    arg_gripper_move_action = DeclareLaunchArgument(
        'gripper_move_action',
        default_value='/franka_gripper/move',
        description='Franka gripper Move action')
    arg_gripper_grasp_action = DeclareLaunchArgument(
        'gripper_grasp_action',
        default_value='/franka_gripper/grasp',
        description='Franka gripper Grasp action')

    experiment_dir = PathJoinSubstitution([
        SAMPLE_ROOT,
        LaunchConfiguration('calibration_setup'),
        LaunchConfiguration('board_type')])

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_fr3_moveit_config'),
                'launch',
                'moveit.launch.py'])]),
        launch_arguments=[
            ('robot_ip', LaunchConfiguration('robot_ip')),
            ('use_fake_hardware', LaunchConfiguration('use_fake_hardware')),
            ('fake_sensor_commands', LaunchConfiguration('fake_sensor_commands')),
            ('namespace', '')])

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'])]),
        launch_arguments=[
            ('camera_name', 'camera'),
            ('camera_namespace', 'camera'),
            ('serial_no', "''"),
            ('enable_color', 'true'),
            ('enable_depth', 'true'),
            ('enable_infra', 'false'),
            ('enable_sync', 'true'),
            ('align_depth.enable', 'true'),
            ('color_width', '1280'),
            ('color_height', '720'),
            ('depth_width', '1280'),
            ('depth_height', '720'),
            ('pointcloud.enable', 'false'),
            ('publish_tf', 'true'),
            ('base_frame_id', 'link'),
            ('tf_prefix', '')])

    handeye_tf_node = Node(
        package='handeye_calibration',
        executable='handeye_tf_publisher',
        name='handeye_tf_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('publish_handeye_tf')),
        parameters=[{
            'sample_dir': experiment_dir,
            'board_type': LaunchConfiguration('board_type'),
            'calibration_setup': LaunchConfiguration('calibration_setup'),
            'method': LaunchConfiguration('handeye_method'),
            'parent_frame': LaunchConfiguration('parent_frame'),
            'child_frame': LaunchConfiguration('child_frame'),
            'optical_frame': LaunchConfiguration('optical_frame'),
        }])

    pixel_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='handeye_calibration',
                executable='pixel_to_robot',
                name='pixel_to_robot',
                output='screen',
                parameters=[{
                    'color_topic': TextSubstitution(
                        text='/camera/camera/color/image_raw'),
                    'depth_topic': TextSubstitution(
                        text='/camera/camera/aligned_depth_to_color/image_raw'),
                    'camera_info_topic': TextSubstitution(
                        text='/camera/camera/color/camera_info'),
                    'camera_frame': TextSubstitution(
                        text='camera_color_optical_frame'),
                    'board_type': LaunchConfiguration('board_type'),
                    'experiment_dir': experiment_dir,
                    'intrinsics_source': LaunchConfiguration('intrinsics_source'),
                    'planning_time': 1.0,
                    'trajectory_action': LaunchConfiguration('trajectory_action'),
                    'gripper_move_action': LaunchConfiguration('gripper_move_action'),
                    'gripper_grasp_action': LaunchConfiguration('gripper_grasp_action'),
                    'enable_auto_grasp': LaunchConfiguration('enable_auto_grasp'),
                    'pregrasp_width': LaunchConfiguration('pregrasp_width'),
                    'grasp_speed': LaunchConfiguration('grasp_speed'),
                    'grasp_force': LaunchConfiguration('grasp_force'),
                    'min_grasp_width': LaunchConfiguration('min_grasp_width'),
                    'target_point_topic': TextSubstitution(
                        text='/pixel_to_robot/target_point'),
                }]),
        ])

    return LaunchDescription([
        arg_robot_ip,
        arg_use_fake_hardware,
        arg_fake_sensor_commands,
        arg_board_type,
        arg_intrinsics_source,
        arg_calibration_setup,
        arg_publish_handeye_tf,
        arg_handeye_method,
        arg_parent_frame,
        arg_child_frame,
        arg_optical_frame,
        arg_enable_auto_grasp,
        arg_pregrasp_width,
        arg_grasp_speed,
        arg_grasp_force,
        arg_min_grasp_width,
        arg_trajectory_action,
        arg_gripper_move_action,
        arg_gripper_grasp_action,
        moveit_launch,
        realsense_launch,
        handeye_tf_node,
        pixel_node,
    ])
