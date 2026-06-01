import importlib.util
import os
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node


def load_pixel_to_robot_launch():
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/handeye_calibration_launch_test_log")
    launch_path = (
        Path(__file__).resolve().parents[1] / "launch" / "pixel_to_robot.launch.py"
    )
    spec = importlib.util.spec_from_file_location("pixel_to_robot_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def launch_argument_default(launch_description, name):
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.name == name:
            default = entity.default_value
            if len(default) == 1 and isinstance(default[0], TextSubstitution):
                return default[0].text
            return default
    raise AssertionError("Missing launch argument: {}".format(name))


def substitution_text(value):
    if isinstance(value, TextSubstitution):
        return value.text
    if isinstance(value, LaunchConfiguration):
        return value.variable_name[0].text
    if isinstance(value, PathJoinSubstitution):
        return str(value)
    if isinstance(value, list):
        return "".join(substitution_text(item) for item in value)
    if isinstance(value, tuple):
        return "".join(substitution_text(item) for item in value)
    return value


def node_parameters(node):
    params = getattr(node, "_Node__parameters")[0]  # pylint: disable=protected-access
    return {substitution_text(key): substitution_text(value) for key, value in params.items()}


def find_pixel_node(launch_description):
    for entity in launch_description.entities:
        if isinstance(entity, TimerAction):
            for action in entity.actions:
                if isinstance(action, Node) and action.node_executable == "pixel_to_robot":
                    return action
    raise AssertionError("Missing pixel_to_robot node")


def find_moveit_include(launch_description):
    for entity in launch_description.entities:
        if isinstance(entity, IncludeLaunchDescription):
            source_text = substitution_text(entity.launch_description_source.location)
            if "franka_fr3_moveit_config" in source_text and "moveit.launch.py" in source_text:
                return entity
    raise AssertionError("Missing MoveIt include")


def find_realsense_include(launch_description):
    return find_realsense_includes(launch_description)[0]


def find_realsense_includes(launch_description):
    includes = []
    entities = list(launch_description.entities)
    while entities:
        entity = entities.pop(0)
        if isinstance(entity, IncludeLaunchDescription):
            source_text = substitution_text(entity.launch_description_source.location)
            if "realsense2_camera" in source_text and "rs_launch.py" in source_text:
                includes.append(entity)
        if isinstance(entity, GroupAction):
            entities.extend(
                getattr(entity, "_GroupAction__actions"))  # pylint: disable=protected-access
    if not includes:
        raise AssertionError("Missing RealSense include")
    return includes


def find_realsense_groups(launch_description):
    groups = []
    for entity in launch_description.entities:
        if not isinstance(entity, GroupAction):
            continue
        actions = getattr(entity, "_GroupAction__actions")  # pylint: disable=protected-access
        for action in actions:
            if not isinstance(action, IncludeLaunchDescription):
                continue
            source_text = substitution_text(action.launch_description_source.location)
            if "realsense2_camera" in source_text and "rs_launch.py" in source_text:
                groups.append(entity)
    if not groups:
        raise AssertionError("Missing grouped RealSense include")
    return groups


def include_launch_arguments(include_action):
    return {
        substitution_text(key): substitution_text(value)
        for key, value in include_action.launch_arguments
    }


def test_pixel_to_robot_launch_exposes_only_core_inputs():
    launch_description = load_pixel_to_robot_launch()

    assert launch_argument_default(launch_description, "robot_ip") == "172.16.0.2"
    assert launch_argument_default(launch_description, "board_type") == "chessboard"
    assert launch_argument_default(launch_description, "use_fake_hardware") == "false"
    assert launch_argument_default(launch_description, "fake_sensor_commands") == "false"
    assert launch_argument_default(launch_description, "intrinsics_source") == "camera_info"
    with pytest.raises(AssertionError):
        launch_argument_default(launch_description, "planning_plugin")

    with pytest.raises(AssertionError):
        launch_argument_default(launch_description, "experiment_dir")
    with pytest.raises(AssertionError):
        launch_argument_default(launch_description, "results_csv")
    with pytest.raises(AssertionError):
        launch_argument_default(launch_description, "intrinsics_file")


def test_pixel_to_robot_launch_internalizes_paths_and_defaults():
    launch_description = load_pixel_to_robot_launch()

    params = node_parameters(find_pixel_node(launch_description))
    assert params["board_type"] == "board_type"
    assert params["intrinsics_source"] == "intrinsics_source"
    assert "experiment_dir" in params
    assert params["planning_time"] == 1.0
    assert params["trajectory_action"] == "trajectory_action"
    assert (
        launch_argument_default(launch_description, "trajectory_action")
        == "fr3_arm_controller/follow_joint_trajectory"
    )
    assert params["gripper_move_action"] == "gripper_move_action"
    assert params["gripper_grasp_action"] == "gripper_grasp_action"
    assert launch_argument_default(launch_description, "gripper_move_action") == "/franka_gripper/move"
    assert launch_argument_default(launch_description, "gripper_grasp_action") == "/franka_gripper/grasp"


def test_pixel_to_robot_launch_uses_realsense_namespaced_image_topics():
    launch_description = load_pixel_to_robot_launch()

    params = node_parameters(find_pixel_node(launch_description))
    assert params["color_topic"] == "/camera/camera/color/image_raw"
    assert params["depth_topic"] == "/camera/camera/aligned_depth_to_color/image_raw"
    assert params["camera_info_topic"] == "/camera/camera/color/camera_info"
    assert params["camera_frame"] == "camera_color_optical_frame"


def test_pixel_to_robot_launch_forwards_only_supported_moveit_arguments():
    launch_description = load_pixel_to_robot_launch()

    args = include_launch_arguments(find_moveit_include(launch_description))
    assert args["robot_ip"] == "robot_ip"
    assert args["use_fake_hardware"] == "use_fake_hardware"
    assert args["fake_sensor_commands"] == "fake_sensor_commands"
    assert args["namespace"] == ""
    assert "planning_plugin" not in args
    assert "launch_camera_tf" not in args


def test_pixel_to_robot_launch_does_not_configure_octomap_persistence():
    launch_description = load_pixel_to_robot_launch()

    args = include_launch_arguments(find_moveit_include(launch_description))
    assert "launch_octomap_manager" not in args


def test_pixel_to_robot_launch_configures_realsense_official_arguments():
    launch_description = load_pixel_to_robot_launch()

    args = include_launch_arguments(find_realsense_include(launch_description))
    assert args["camera_name"] == "camera"
    assert args["camera_namespace"] == "camera"
    assert args["enable_color"] == "true"
    assert args["enable_depth"] == "true"
    assert args["enable_sync"] == "true"
    assert args["align_depth.enable"] == "true"
    assert args["pointcloud.enable"] == "false"


def test_pixel_to_robot_launch_keeps_realsense_serials_as_strings():
    launch_description = load_pixel_to_robot_launch()

    serials = [
        include_launch_arguments(include)["serial_no"]
        for include in find_realsense_includes(launch_description)
    ]

    assert serials == ["'044322073013'", "'420122071571'"]


def test_pixel_to_robot_launch_uses_realsense_profile_arguments():
    launch_description = load_pixel_to_robot_launch()

    args = include_launch_arguments(find_realsense_include(launch_description))
    assert args["rgb_camera.color_profile"] == "1280,720,30"
    assert args["depth_module.depth_profile"] == "1280,720,30"
    assert "color_width" not in args
    assert "color_height" not in args
    assert "depth_width" not in args
    assert "depth_height" not in args


def test_pixel_to_robot_launch_scopes_realsense_includes():
    launch_description = load_pixel_to_robot_launch()

    for group in find_realsense_groups(launch_description):
        forwarding = getattr(group, "_GroupAction__forwarding")  # pylint: disable=protected-access
        assert forwarding is False


def test_pixel_to_robot_launch_exposes_auto_grasp_parameters():
    launch_description = load_pixel_to_robot_launch()

    assert launch_argument_default(launch_description, "enable_auto_grasp") == "true"
    assert launch_argument_default(launch_description, "pregrasp_width") == "0.08"
    assert launch_argument_default(launch_description, "grasp_speed") == "0.03"
    assert launch_argument_default(launch_description, "grasp_force") == "10.0"
    assert launch_argument_default(launch_description, "min_grasp_width") == "0.005"

    params = node_parameters(find_pixel_node(launch_description))
    assert params["enable_auto_grasp"] == "enable_auto_grasp"
    assert params["pregrasp_width"] == "pregrasp_width"
    assert params["grasp_speed"] == "grasp_speed"
    assert params["grasp_force"] == "grasp_force"
    assert params["min_grasp_width"] == "min_grasp_width"


def test_pixel_to_robot_launch_has_only_pixel_node_in_timer():
    launch_description = load_pixel_to_robot_launch()

    timer_actions = []
    for entity in launch_description.entities:
        if isinstance(entity, TimerAction):
            timer_actions.extend(entity.actions)
    assert len(timer_actions) == 1
    assert timer_actions[0].node_executable == "pixel_to_robot"
