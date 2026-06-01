import numpy as np
import pytest

from franka_policy_runtime.observers import RLObserver, VLAObserver


JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


class FakeImage:
    height = 0
    width = 0
    encoding = ""
    data = b""


class FakeJointState:
    name = []
    position = []
    velocity = []


class FakeString:
    data = ""


def make_image() -> FakeImage:
    msg = FakeImage()
    msg.height = 2
    msg.width = 3
    msg.encoding = "rgb8"
    msg.data = bytes(range(18))
    return msg


def make_joint_state() -> FakeJointState:
    msg = FakeJointState()
    msg.name = ["extra_joint", *JOINT_NAMES]
    msg.position = [9.0, *np.linspace(0.1, 0.7, 7).tolist()]
    msg.velocity = [8.0, *np.linspace(-0.1, -0.7, 7).tolist()]
    return msg


def test_vla_observer_returns_current_image_only_observation():
    observer = VLAObserver(instruction="move the object")
    observer.update_image(make_image())

    observation = observer.observe()

    assert observation.ready
    assert observation.instruction == "move the object"
    assert observation.image.shape == (2, 3, 3)
    assert observation.image.dtype == np.uint8
    assert observation.image[0, 0].tolist() == [0, 1, 2]


def test_vla_observer_updates_instruction_from_string_message():
    observer = VLAObserver(instruction="move the object")
    msg = FakeString()
    msg.data = "pick up the cube"

    observer.update_instruction(msg)

    assert observer.observe().instruction == "pick up the cube"


def test_vla_observer_reports_not_ready_without_image():
    observer = VLAObserver(instruction="move the object")

    observation = observer.observe()

    assert not observation.ready
    assert observation.image is None
    assert observation.instruction == "move the object"


def test_rl_observer_collects_available_isaaclab_style_terms():
    observer = RLObserver(joint_names=JOINT_NAMES)
    tcp_position = np.array([0.4, -0.2, 0.6], dtype=float)
    tcp_quat = np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=float)
    last_action = np.arange(7, dtype=float) * 0.01

    observer.update_image(make_image())
    observer.update_joint_state(make_joint_state())
    observer.update_tcp_pose(tcp_position, tcp_quat)
    observer.update_gripper_width(0.035)
    observer.update_last_action(last_action)
    observation = observer.observe()

    assert observation.ready
    assert observation.terms["joint_pos"].tolist() == pytest.approx(np.linspace(0.1, 0.7, 7).tolist())
    assert observation.terms["joint_vel"].tolist() == pytest.approx(np.linspace(-0.1, -0.7, 7).tolist())
    assert observation.terms["eef_pos"].tolist() == pytest.approx(tcp_position.tolist())
    assert observation.terms["eef_quat"].tolist() == pytest.approx(tcp_quat.tolist())
    assert observation.terms["gripper_pos"].tolist() == pytest.approx([0.035])
    assert observation.terms["last_action"].tolist() == pytest.approx(last_action.tolist())
    assert observation.images["image"].shape == (2, 3, 3)
    assert not observation.availability["object_pose_in_eef"]
    assert observation.terms["object_pose_in_eef"].shape == (7,)


def test_rl_observer_is_not_ready_until_joint_state_and_tcp_pose_arrive():
    observer = RLObserver(joint_names=JOINT_NAMES)
    observer.update_joint_state(make_joint_state())

    assert not observer.observe().ready

    observer.update_tcp_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    assert observer.observe().ready
