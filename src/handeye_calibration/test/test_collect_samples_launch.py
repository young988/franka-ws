import importlib.util
import os
from pathlib import Path

from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node


def load_collect_samples_launch():
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/handeye_calibration_launch_test_log")
    launch_path = (
        Path(__file__).resolve().parents[1] / "launch" / "collect_samples.launch.py"
    )
    spec = importlib.util.spec_from_file_location("collect_samples_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


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


def launch_argument_default(launch_description, name):
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.name == name:
            default = entity.default_value
            if len(default) == 1 and isinstance(default[0], TextSubstitution):
                return default[0].text
            return default
    raise AssertionError("Missing launch argument: {}".format(name))


def include_launch_arguments(include_action):
    return {
        substitution_text(key): substitution_text(value)
        for key, value in include_action.launch_arguments
    }


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


def find_collector_node(launch_description):
    for entity in launch_description.entities:
        if isinstance(entity, Node) and entity.node_executable == "sample_collector":
            return entity
    raise AssertionError("Missing sample_collector node")


def node_parameters(node):
    params = getattr(node, "_Node__parameters")[0]  # pylint: disable=protected-access
    return {substitution_text(key): substitution_text(value) for key, value in params.items()}


def test_collect_samples_launch_keeps_realsense_serials_as_strings():
    launch_description = load_collect_samples_launch()

    serials = [
        include_launch_arguments(include)["serial_no"]
        for include in find_realsense_includes(launch_description)
    ]

    assert serials == ["'044322073013'", "'420122071571'"]


def test_collect_samples_launch_uses_realsense_profile_arguments():
    launch_description = load_collect_samples_launch()

    args = include_launch_arguments(find_realsense_includes(launch_description)[0])
    assert args["rgb_camera.color_profile"] == "1280,720,30"
    assert args["depth_module.depth_profile"] == "1280,720,30"
    assert "color_width" not in args
    assert "color_height" not in args
    assert "depth_width" not in args
    assert "depth_height" not in args


def test_collect_samples_launch_exposes_headless_mode():
    launch_description = load_collect_samples_launch()

    assert launch_argument_default(launch_description, "headless") == "false"
    params = node_parameters(find_collector_node(launch_description))
    assert params["headless"] == "headless"


def test_collect_samples_launch_scopes_realsense_includes():
    launch_description = load_collect_samples_launch()

    for group in find_realsense_groups(launch_description):
        forwarding = getattr(group, "_GroupAction__forwarding")  # pylint: disable=protected-access
        assert forwarding is False
