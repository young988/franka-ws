from pathlib import Path
from xml.etree import ElementTree


def test_preview_urdf_contains_fr3_arm_and_gripper_joints():
    urdf_path = Path(__file__).parents[1] / "urdf" / "fr3_teleop_preview.urdf"
    root = ElementTree.parse(urdf_path).getroot()
    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}

    for index in range(1, 8):
        assert f"fr3_joint{index}" in joint_names
    assert "fr3_finger_joint1" in joint_names
    assert "fr3_finger_joint2" in joint_names


def test_legacy_franka_telep_launch_removed():
    launch_path = Path(__file__).parents[1] / "launch" / "franka_telep.launch.py"
    assert not launch_path.exists()
