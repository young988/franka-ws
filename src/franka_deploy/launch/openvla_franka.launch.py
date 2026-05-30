"""OpenVLA + Franka FR3 launch.

Includes the standard franka_fr3_moveit_config moveit.launch.py and
adds only the VLA-specific components (inference server, planner node,
RealSense camera).
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_yaml(pkg, path):
    import yaml
    from ament_index_python.packages import get_package_share_directory
    with open(os.path.join(get_package_share_directory(pkg), path)) as f:
        return yaml.safe_load(f)


def launch_setup(context):
    # ---- VLA planner params ----
    planner_params_path = os.path.join(
        FindPackageShare('franka_deploy').find('franka_deploy'),
        'config', 'vla_planner.yaml',
    )

    # ---- RealSense ----
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch', 'rs_launch.py',
            ])
        ]),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'serial_no': "''",
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_infra': 'false',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'publish_tf': 'true',
            'base_frame_id': 'link',
            'tf_prefix': '',
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_sensor')),
    )

    # ---- OpenVLA HTTP inference server ----
    openvla_server = ExecuteProcess(
        cmd=[
            LaunchConfiguration('openvla_python'),
            os.path.join(
                FindPackageShare('franka_deploy').find('franka_deploy'),
                'scripts', 'openvla_quant_server.py',
            ),
            '--openvla_path', LaunchConfiguration('openvla_path'),
            '--host', LaunchConfiguration('server_host'),
            '--port', LaunchConfiguration('server_port'),
            '--load_in_4bit', LaunchConfiguration('load_in_4bit'),
            '--load_in_8bit', LaunchConfiguration('load_in_8bit'),
            '--attn_implementation', LaunchConfiguration('attn_implementation'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_openvla_server')),
    )

    # ---- VLA planner node ----
    vla_planner = Node(
        package='franka_deploy',
        executable='vla_planner_node',
        name='openvla_planner',
        output='screen',
        parameters=[planner_params_path],
    )

    return [realsense_launch, openvla_server, vla_planner]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='172.16.0.2'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false'),
        DeclareLaunchArgument('launch_sensor', default_value='true'),
        DeclareLaunchArgument('start_openvla_server', default_value='true'),
        DeclareLaunchArgument(
            'openvla_python',
            default_value='/home/young/miniconda3/envs/openvla/bin/python',
        ),
        DeclareLaunchArgument('openvla_path', default_value='openvla/openvla-7b'),
        DeclareLaunchArgument('server_host', default_value='127.0.0.1'),
        DeclareLaunchArgument('server_port', default_value='8000'),
        DeclareLaunchArgument('load_in_4bit', default_value='true'),
        DeclareLaunchArgument('load_in_8bit', default_value='false'),
        DeclareLaunchArgument('attn_implementation', default_value='sdpa'),
        DeclareLaunchArgument('instruction', default_value='move the object'),
        DeclareLaunchArgument('unnorm_key', default_value='bridge_orig'),

        # ---- Standard MoveIt launch (handles all robot config) ----
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('franka_fr3_moveit_config'),
                    'launch', 'moveit.launch.py',
                ])
            ]),
            launch_arguments={
                'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
                'fake_sensor_commands': LaunchConfiguration('fake_sensor_commands'),
            }.items(),
        ),

        OpaqueFunction(function=launch_setup),
    ])
