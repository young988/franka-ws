from pathlib import Path

_CONTROLLER_HPP = Path(__file__).parents[2] / "franka_policy_controller" / "include" / "franka_policy_controller" / "franka_cartesian_pose_controller.hpp"
_CONTROLLER_CPP = Path(__file__).parents[2] / "franka_policy_controller" / "src" / "franka_cartesian_pose_controller.cpp"
_PLUGIN_XML = Path(__file__).parents[2] / "franka_policy_controller" / "franka_policy_controller_plugin.xml"


def test_cartesian_pose_controller_plugin_declares_pose_interfaces_and_subscription():
    header = _CONTROLLER_HPP.read_text(encoding="utf-8")
    source = _CONTROLLER_CPP.read_text(encoding="utf-8")
    plugin = _PLUGIN_XML.read_text(encoding="utf-8")

    assert "FrankaCartesianPoseController" in header
    assert "geometry_msgs::msg::PoseStamped" in header
    assert "cartesian_pose_command" in source
    assert 'create_subscription<geometry_msgs::msg::PoseStamped>' in source
    assert "franka_policy_controller/FrankaCartesianPoseController" in plugin
