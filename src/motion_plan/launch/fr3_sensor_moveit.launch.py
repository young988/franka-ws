"""Launch FR3 MoveIt with RealSense octomap input.

Examples:
  ros2 launch motion_plan fr3_sensor_moveit.launch.py planner:=ompl
  ros2 launch motion_plan fr3_sensor_moveit.launch.py planner:=rrt
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, relative_path):
    path = os.path.join(get_package_share_directory(package_name), relative_path)
    with open(path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def fr3_ompl_config():
    config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/ResolveConstraintFrames '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    if 'fr3_arm' not in ompl_yaml and 'panda_arm' in ompl_yaml:
        ompl_yaml['fr3_arm'] = ompl_yaml['panda_arm']
    config['move_group'].update(ompl_yaml)
    return config


def rrt_config():
    config = {
        'move_group': {
            'planning_plugin': 'motion_plan/RRTPlannerManager',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/ResolveConstraintFrames '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
        }
    }
    config['move_group'].update(load_yaml('motion_plan', 'config/rrt_planning.yaml'))
    return config


def selected_planning_config(context):
    planner = LaunchConfiguration('planner').perform(context).strip().lower()
    if planner in ('rrt', 'custom', 'motion_plan'):
        return rrt_config()
    return fr3_ompl_config()


def launch_setup(context):
    namespace = LaunchConfiguration('namespace')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    robot_ip = LaunchConfiguration('robot_ip')

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        'fr3',
        'fr3.urdf.xacro',
    )
    robot_description_config = Command([
        FindExecutable(name='xacro'),
        ' ',
        franka_xacro_file,
        ' hand:=true',
        ' robot_ip:=',
        robot_ip,
        ' use_fake_hardware:=',
        use_fake_hardware,
        ' fake_sensor_commands:=',
        fake_sensor_commands,
        ' ros2_control:=true',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_config, value_type=str)
    }

    srdf_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        'fr3',
        'fr3.srdf.xacro',
    )
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            Command([FindExecutable(name='xacro'), ' ', srdf_xacro_file, ' hand:=true']),
            value_type=str,
        )
    }

    kinematics_yaml = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': load_yaml(
            'franka_fr3_moveit_config', 'config/fr3_controllers.yaml'
        ),
        'moveit_controller_manager': (
            'moveit_simple_controller_manager/MoveItSimpleControllerManager'
        ),
    }
    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }
    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    octomap_parameters = {
        'octomap_frame': LaunchConfiguration('octomap_frame'),
        'octomap_resolution': ParameterValue(
            LaunchConfiguration('octomap_resolution'), value_type=float),
        'max_range': ParameterValue(
            LaunchConfiguration('octomap_max_range'), value_type=float),
        'sensors': ['realsense_pointcloud'],
        'realsense_pointcloud': {
            'sensor_plugin': 'occupancy_map_monitor/PointCloudOctomapUpdater',
            'point_cloud_topic': LaunchConfiguration('point_cloud_topic'),
            'max_range': ParameterValue(
                LaunchConfiguration('point_cloud_max_range'), value_type=float),
            'point_subsample': ParameterValue(
                LaunchConfiguration('point_subsample'), value_type=int),
            'max_update_rate': ParameterValue(
                LaunchConfiguration('octomap_max_update_rate'), value_type=float),
            'padding_offset': ParameterValue(
                LaunchConfiguration('padding_offset'), value_type=float),
            'padding_scale': ParameterValue(
                LaunchConfiguration('padding_scale'), value_type=float),
            'filtered_cloud_topic': LaunchConfiguration('filtered_cloud_topic'),
        },
    }

    planning_config = selected_planning_config(context)

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        namespace=namespace,
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            octomap_parameters,
        ],
    )

    rviz_config = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'rviz',
        'moveit.rviz',
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            planning_config,
            kinematics_yaml,
        ],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='both',
        parameters=[robot_description],
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=namespace,
        parameters=[
            robot_description,
            os.path.join(
                get_package_share_directory('franka_fr3_moveit_config'),
                'config',
                'fr3_ros_controllers.yaml',
            ),
        ],
        remappings=[('joint_states', 'franka/joint_states')],
        output={'stdout': 'screen', 'stderr': 'screen'},
        on_exit=Shutdown(),
    )

    controller_spawners = [
        ExecuteProcess(
            cmd=[
                'ros2',
                'run',
                'controller_manager',
                'spawner',
                controller,
                '--controller-manager-timeout',
                '60',
                '--controller-manager',
                PathJoinSubstitution([namespace, 'controller_manager']),
            ],
            output='screen',
        )
        for controller in ['fr3_arm_controller', 'joint_state_broadcaster']
    ]

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[{'source_list': ['franka/joint_states', 'fr3_gripper/joint_states'], 'rate': 30}],
    )

    franka_robot_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=['franka_robot_state_broadcaster'],
        output='screen',
        condition=UnlessCondition(use_fake_hardware),
    )

    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_gripper'),
                'launch',
                'gripper.launch.py',
            ])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake_hardware,
            'namespace': namespace,
        }.items(),
    )

    return [
        move_group_node,
        rviz_node,
        robot_state_publisher,
        ros2_control_node,
        joint_state_publisher,
        franka_robot_state_broadcaster,
        gripper_launch,
    ] + controller_spawners


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py',
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

    handeye_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('handeye_calibration'),
                'launch',
                'publish_handeye_tf.launch.py',
            ])
        ]),
        launch_arguments={
            'result_file': LaunchConfiguration('handeye_result_file'),
            'sample_dir': LaunchConfiguration('handeye_sample_dir'),
            'board_type': LaunchConfiguration('board_type'),
            'calibration_setup': LaunchConfiguration('calibration_setup'),
            'method': LaunchConfiguration('handeye_method'),
            'parent_frame': LaunchConfiguration('handeye_parent_frame'),
            'child_frame': LaunchConfiguration('handeye_child_frame'),
            'optical_frame': LaunchConfiguration('handeye_optical_frame'),
            'invert_result': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('publish_handeye_tf')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value='ompl',
                              description='Planner backend: ompl or rrt'),
        DeclareLaunchArgument('robot_ip', default_value='172.16.0.2',
                              description='Franka robot IP / hostname'),
        DeclareLaunchArgument('namespace', default_value='',
                              description='ROS namespace for Franka nodes'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false',
                              description='Use Franka fake hardware'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false',
                              description='Use fake commands with fake hardware'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Start RViz with the MoveIt config'),
        DeclareLaunchArgument('launch_sensor', default_value='true',
                              description='Start the RealSense camera launch file'),
        DeclareLaunchArgument('publish_handeye_tf', default_value='true',
                              description='Publish calibrated franka link8 to camera_link TF'),
        DeclareLaunchArgument('handeye_result_file', default_value='',
                              description='Path to handeye_results.csv; auto-resolved when empty'),
        DeclareLaunchArgument('handeye_sample_dir', default_value='',
                              description='Calibration sample directory; used when result file is empty'),
        DeclareLaunchArgument('board_type', default_value='chessboard',
                              description='Board type used to resolve the hand-eye result path'),
        DeclareLaunchArgument('calibration_setup', default_value='eye_in_hand',
                              description='Hand-eye setup used to resolve the result path'),
        DeclareLaunchArgument('handeye_method', default_value='best',
                              description='Calibration row to publish: best or method name'),
        DeclareLaunchArgument('handeye_parent_frame', default_value='fr3_link8',
                              description='Parent frame for hand-eye TF, e.g. fr3_link8 or franka_link8'),
        DeclareLaunchArgument('handeye_child_frame', default_value='camera_link',
                              description='Target frame for hand-eye TF'),
        DeclareLaunchArgument('handeye_optical_frame', default_value='camera_color_optical_frame',
                              description='Optical frame used during calibration (bridge to child_frame)'),
        DeclareLaunchArgument('point_cloud_topic',
                              default_value='/camera/camera/depth/color/points',
                              description='PointCloud2 topic consumed by MoveIt octomap'),
        DeclareLaunchArgument('filtered_cloud_topic',
                              default_value='/move_group/filtered_cloud',
                              description='Filtered point cloud debug topic'),
        DeclareLaunchArgument('octomap_frame', default_value='fr3_link0',
                              description='Frame used by MoveIt for the octomap'),
        DeclareLaunchArgument('octomap_resolution', default_value='0.03',
                              description='Octomap voxel resolution in meters'),
        DeclareLaunchArgument('octomap_max_range', default_value='3.0',
                              description='Global octomap max range'),
        DeclareLaunchArgument('point_cloud_max_range', default_value='3.0',
                              description='Point cloud updater max range'),
        DeclareLaunchArgument('point_subsample', default_value='1',
                              description='Use every Nth point from the cloud'),
        DeclareLaunchArgument('octomap_max_update_rate', default_value='5.0',
                              description='Maximum octomap updates per second; <=0 disables throttling'),
        DeclareLaunchArgument('padding_offset', default_value='0.03',
                              description='Robot padding offset for self-filtering'),
        DeclareLaunchArgument('padding_scale', default_value='1.0',
                              description='Robot padding scale for self-filtering'),
        realsense_launch,
        handeye_tf_launch,
        OpaqueFunction(function=launch_setup),
    ])
