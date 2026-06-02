from pathlib import Path

_CONTROLLER_HPP = Path(__file__).parents[2] / "franka_policy_controller" / "include" / "franka_policy_controller" / "franka_cartesian_pose_controller.hpp"
_CONTROLLER_CPP = Path(__file__).parents[2] / "franka_policy_controller" / "src" / "franka_cartesian_pose_controller.cpp"
_PLUGIN_XML = Path(__file__).parents[2] / "franka_policy_controller" / "franka_policy_controller_plugin.xml"
_RUNTIME_BASE = Path(__file__).parents[1] / "launch" / "robot_base.launch.py"
_CONTROLLERS = Path(__file__).parents[2] / "franka_policy_controller" / "config" / "franka_bringup_cartesian_pose_controllers.yaml"
_OLD_CONTROLLER_CPP = Path(__file__).parents[2] / "franka_policy_controller" / "src" / "franka_policy_controller.cpp"


def test_cartesian_pose_controller_plugin_declares_pose_interfaces_and_subscription():
    header = _CONTROLLER_HPP.read_text(encoding="utf-8")
    source = _CONTROLLER_CPP.read_text(encoding="utf-8")
    plugin = _PLUGIN_XML.read_text(encoding="utf-8")

    assert "FrankaCartesianPoseController" in header
    assert "geometry_msgs::msg::PoseStamped" in header
    assert "cartesian_pose_command" in source
    assert 'create_subscription<geometry_msgs::msg::PoseStamped>' in source
    assert "franka_policy_controller/FrankaCartesianPoseController" in plugin


def test_robot_base_launch_uses_cartesian_pose_controller_and_no_move_group():
    source = _RUNTIME_BASE.read_text(encoding="utf-8")
    assert "franka_bringup_cartesian_pose_controllers.yaml" in source
    assert "franka_cartesian_pose_controller" in source
    assert "move_group" not in source


def test_cartesian_controller_yaml_registers_expected_controller():
    source = _CONTROLLERS.read_text(encoding="utf-8")
    assert "franka_cartesian_pose_controller" in source
    assert "franka_policy_controller/FrankaCartesianPoseController" in source


def test_legacy_joint_reference_controller_is_removed_from_mainline():
    assert not _OLD_CONTROLLER_CPP.exists()
