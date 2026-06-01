"""
Launch hand-eye calibration sample collection.

Brings up:
  - Franka bringup (hardware interface + state broadcasters)
  - Gravity compensation controller (robot holds pose freely)
  - RealSense D435i camera
  - Sample collector (press 's' to save image + robot pose)

Usage:
    ros2 launch handeye_calibration collect_samples.launch.py \
        robot_ip:=172.16.0.2 \
        board_type:=chessboard
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SAMPLE_ROOT = '/home/young/ros2_ws/src/handeye_calibration/samples'


def generate_launch_description():
    arg_robot_ip = DeclareLaunchArgument(
        'robot_ip', default_value='172.16.0.2',
        description='Franka robot IP / hostname')
    arg_use_fake_hardware = DeclareLaunchArgument(
        'use_fake_hardware', default_value='false',
        description='Use Franka fake hardware')
    arg_fake_sensor_commands = DeclareLaunchArgument(
        'fake_sensor_commands', default_value='false',
        description='Use fake sensor commands with fake hardware')
    arg_board_type = DeclareLaunchArgument(
        'board_type', default_value='chessboard',
        description='single_aruco, charuco, aruco_grid, or chessboard')
    arg_calibration_setup = DeclareLaunchArgument(
        'calibration_setup', default_value='eye_in_hand',
        description='eye_in_hand or eye_to_hand')
    arg_headless = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run without OpenCV display window')

    franka_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_bringup'),
                'launch',
                'franka.launch.py'])]),
        launch_arguments=[
            ('arm_id', 'fr3'),
            ('robot_ip', LaunchConfiguration('robot_ip')),
            ('use_fake_hardware', LaunchConfiguration('use_fake_hardware')),
            ('fake_sensor_commands', LaunchConfiguration('fake_sensor_commands')),
            ('load_gripper', 'false')])

    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gravity_compensation_example_controller',
                   '--controller-manager-timeout', '30'],
        output='screen')

    realsense_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'])]),
        launch_arguments=[
            ('camera_name', 'camera'),
            ('camera_namespace', 'camera'),
            ('serial_no', "'044322073013'"),
            ('enable_color', 'true'),
            ('enable_depth', 'true'),
            ('enable_infra', 'false'),
            ('rgb_camera.color_profile', '1280,720,30'),
            ('depth_module.depth_profile', '1280,720,30'),
            ('publish_tf', 'true'),
            ('base_frame_id', 'link'),
            ('tf_prefix', '')])

    realsense_eye = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'])]),
        launch_arguments=[
            ('camera_name', 'camera'),
            ('camera_namespace', 'camera'),
            ('serial_no', "'420122071571'"),
            ('enable_color', 'true'),
            ('enable_depth', 'true'),
            ('enable_infra', 'false'),
            ('rgb_camera.color_profile', '1280,720,30'),
            ('depth_module.depth_profile', '1280,720,30'),
            ('publish_tf', 'true'),
            ('base_frame_id', 'link'),
            ('tf_prefix', '')])

    collector = Node(
        package='handeye_calibration',
        executable='sample_collector',
        name='sample_collector',
        output='screen',
        parameters=[{
            'sample_dir': PathJoinSubstitution([
                SAMPLE_ROOT,
                LaunchConfiguration('calibration_setup'),
                LaunchConfiguration('board_type')]),
            'image_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'robot_base_frame': 'fr3_link0',
            'robot_effector_frame': 'fr3_link8',
            'calibration_setup': LaunchConfiguration('calibration_setup'),
            'tracking_base_frame': 'fr3_link0',
            'tracking_marker_frame': 'fr3_link8',
            'board_type': LaunchConfiguration('board_type'),
            'headless': LaunchConfiguration('headless'),
        }])

    return LaunchDescription([
        arg_robot_ip,
        arg_use_fake_hardware,
        arg_fake_sensor_commands,
        arg_board_type,
        arg_calibration_setup,
        arg_headless,
        franka_launch,
        controller_spawner,
        GroupAction(
            actions=[realsense_hand],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('calibration_setup'),
                "' != 'eye_to_hand'"])),
            scoped=True,
            forwarding=False),
        GroupAction(
            actions=[realsense_eye],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('calibration_setup'),
                "' == 'eye_to_hand'"])),
            scoped=True,
            forwarding=False),
        collector,
    ])
