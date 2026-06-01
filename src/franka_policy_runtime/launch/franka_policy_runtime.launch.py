from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare("franka_policy_runtime"),
        "config",
        "franka_policy_runtime.yaml",
    ])
    return LaunchDescription([
        Node(
            package="franka_policy_runtime",
            executable="policy_runtime_node",
            name="franka_policy_runtime",
            output="screen",
            parameters=[params],
        ),
    ])
