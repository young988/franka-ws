import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from sensor_msgs.msg import Image as RosImage

from franka_telep.openvla_dataset import (
    OpenVLAEpisodeWriter,
    binary_gripper_action,
    center_crop_resize_rgb,
    openvla_action,
    openvla_state,
)
from franka_telep.openvla_recorder_node import ros_image_to_rgb


def _yaw_quaternion(angle_rad):
    return np.array([
        0.0,
        0.0,
        math.sin(angle_rad / 2.0),
        math.cos(angle_rad / 2.0),
    ])


def test_openvla_gripper_uses_bridge_binary_convention():
    assert binary_gripper_action(0.08, open_threshold=0.04) == 1.0
    assert binary_gripper_action(0.0, open_threshold=0.04) == 0.0


def test_openvla_state_has_pos_euler_padding_and_gripper():
    state = openvla_state(
        np.array([0.4, -0.2, 0.3]),
        _yaw_quaternion(math.pi / 2.0),
        0.08,
        gripper_open_threshold=0.04,
    )

    assert state.shape == (8,)
    assert state[:3] == pytest.approx([0.4, -0.2, 0.3])
    assert state[3:6] == pytest.approx([0.0, 0.0, math.pi / 2.0])
    assert state[6:].tolist() == [0.0, 1.0]


def test_openvla_action_matches_bridge_reached_state_relabeling():
    action = openvla_action(
        np.array([0.4, -0.2, 0.3]),
        _yaw_quaternion(0.2),
        np.array([0.41, -0.22, 0.33]),
        _yaw_quaternion(0.3),
        0.0,
        gripper_open_threshold=0.04,
    )

    assert action.shape == (7,)
    assert action == pytest.approx([0.01, -0.02, 0.03, 0.0, 0.0, 0.1, 0.0])


def test_openvla_action_wraps_euler_boundary():
    action = openvla_action(
        np.zeros(3),
        _yaw_quaternion(math.pi - 0.01),
        np.zeros(3),
        _yaw_quaternion(-math.pi + 0.01),
        0.08,
        gripper_open_threshold=0.04,
    )

    assert action[5] == pytest.approx(0.02)


def test_center_crop_resize_produces_openvla_rgb_shape():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[:, 40:160, 1] = 255

    resized = center_crop_resize_rgb(image, 256)

    assert resized.shape == (256, 256, 3)
    assert resized.dtype == np.uint8
    assert resized[:, :, 1].mean() > 250


def test_episode_writer_emits_raw_rlds_source(tmp_path):
    writer = OpenVLAEpisodeWriter(
        tmp_path,
        dataset_name="Franka Teleop",
        instruction="pick up the red block",
        image_size=256,
        has_wrist_image=False,
    )
    image = np.full((256, 256, 3), [20, 40, 60], dtype=np.uint8)
    writer.append(
        image_rgb=image,
        wrist_image_rgb=None,
        state=np.arange(8, dtype=np.float32),
        joint_positions=np.arange(7, dtype=np.float32),
        action=np.arange(7, dtype=np.float32),
        timestamp_sec=12.5,
    )

    episode_dir = writer.finalize()
    metadata = json.loads((episode_dir / "episode.json").read_text())
    with np.load(episode_dir / "steps.npz", allow_pickle=False) as steps:
        assert steps["state"].shape == (1, 8)
        assert steps["joint_positions"].shape == (1, 7)
        assert steps["action"].shape == (1, 7)
        assert steps["image_path"].tolist() == ["images/000000.jpg"]
    decoded = np.asarray(Image.open(episode_dir / "images" / "000000.jpg"))

    assert episode_dir.name == "episode_000000"
    assert metadata["dataset_name"] == "franka_teleop"
    assert metadata["instruction"] == "pick up the red block"
    assert metadata["gripper_semantics"] == {"closed": 0.0, "open": 1.0}
    assert decoded.shape == (256, 256, 3)


def test_ros_image_to_rgb_handles_rgb8_row_padding():
    message = RosImage()
    message.height = 1
    message.width = 2
    message.encoding = "rgb8"
    message.step = 8
    message.data = bytes([1, 2, 3, 4, 5, 6, 99, 99])

    image = ros_image_to_rgb(message)

    assert image.tolist() == [[[1, 2, 3], [4, 5, 6]]]


def test_ros_image_to_rgb_converts_bgr8():
    message = RosImage()
    message.height = 1
    message.width = 1
    message.encoding = "bgr8"
    message.step = 3
    message.data = bytes([30, 20, 10])

    image = ros_image_to_rgb(message)

    assert image.tolist() == [[[10, 20, 30]]]


def test_episode_writer_rejects_non_openvla_image_size(tmp_path):
    with pytest.raises(ValueError, match="256x256"):
        OpenVLAEpisodeWriter(
            tmp_path,
            dataset_name="franka_teleop",
            instruction="test",
            image_size=224,
            has_wrist_image=False,
        )


def test_episode_writer_abort_removes_unfinished_episode(tmp_path):
    writer = OpenVLAEpisodeWriter(
        tmp_path,
        dataset_name="franka_teleop",
        instruction="test",
        image_size=256,
        has_wrist_image=False,
    )
    temporary_dir = writer._temporary_dir

    writer.abort()

    assert not temporary_dir.exists()


def test_recorder_config_matches_openvla_contract():
    config_path = Path(__file__).parents[1] / "config" / "franka_telep.yaml"
    config = yaml.safe_load(config_path.read_text())
    recorder = config["openvla_dataset_recorder"]["ros__parameters"]

    assert recorder["image_size"] == 256
    assert recorder["sample_rate_hz"] > 0.0
    assert recorder["joint_state_topic"] == "/joint_states"
    assert recorder["gripper_joint_state_topic"] == "/fr3_gripper/joint_states"
    assert recorder["tcp_pose_topic"].endswith("/current_pose")
    assert len(recorder["joint_names"]) == 7
