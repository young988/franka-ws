import numpy as np
import pytest

from franka_policy_runtime.observers import (
    ColorCubeObjectPoseProvider,
    ColorCubeStackObjectProvider,
    IsaacLabStackBCObserver,
    OpenVLAObserver,
    estimate_object_pose_in_eef,
)


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


class FakeCameraInfo:
    k = []


class FakeTransform:
    class Transform:
        class Translation:
            x = 0.0
            y = 0.0
            z = 0.0

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        translation = Translation()
        rotation = Rotation()

    transform = Transform()


class FakeTfBuffer:
    def __init__(self, transform):
        self.transform = transform
        self.lookup_args = None
        self.lookup_time = None

    def lookup_transform(self, target_frame, source_frame, time, timeout=None):
        del time, timeout
        self.lookup_args = (target_frame, source_frame)
        return self.transform


class FakeStrictTfBuffer(FakeTfBuffer):
    def lookup_transform(self, target_frame, source_frame, time, timeout=None):
        del timeout
        if time is None:
            raise TypeError("time must be a rclpy.time.Time instance")
        self.lookup_args = (target_frame, source_frame)
        self.lookup_time = time
        return self.transform


def make_image(start: int = 0) -> FakeImage:
    msg = FakeImage()
    msg.height = 2
    msg.width = 3
    msg.encoding = "rgb8"
    msg.data = bytes(range(start, start + 18))
    return msg


def make_rgb_image(array: np.ndarray) -> FakeImage:
    msg = FakeImage()
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = "rgb8"
    msg.data = np.asarray(array, dtype=np.uint8).tobytes()
    return msg


def make_depth_image(array: np.ndarray) -> FakeImage:
    msg = FakeImage()
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = "16UC1"
    msg.data = np.asarray(array, dtype=np.uint16).tobytes()
    return msg


def make_camera_info() -> FakeCameraInfo:
    msg = FakeCameraInfo()
    msg.k = [100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]
    return msg


def make_joint_state() -> FakeJointState:
    msg = FakeJointState()
    msg.name = ["extra_joint", *JOINT_NAMES]
    msg.position = [9.0, *np.linspace(0.1, 0.7, 7).tolist()]
    msg.velocity = [8.0, *np.linspace(-0.1, -0.7, 7).tolist()]
    return msg


def test_openvla_observer_returns_primary_image_and_instruction_only():
    observer = OpenVLAObserver(instruction="move the object")
    observer.update_image(make_image(), name="eye_to_hand")
    observer.update_image(make_image(18), name="eye_in_hand")

    observation = observer.observe()

    assert observation.ready
    assert observation.payload["instruction"] == "move the object"
    assert observation.payload["image"][0, 0].tolist() == [0, 1, 2]
    assert "images" not in observation.payload
    assert "terms" not in observation.payload


def test_openvla_observer_updates_instruction_from_string_message():
    observer = OpenVLAObserver(instruction="move the object")
    msg = FakeString()
    msg.data = "pick up the cube"

    observer.update_instruction(msg)

    assert observer.observe().payload["instruction"] == "pick up the cube"


def test_openvla_observer_reports_not_ready_without_primary_image():
    observer = OpenVLAObserver(instruction="move the object")

    observation = observer.observe()

    assert not observation.ready
    assert observation.payload["instruction"] == "move the object"
    assert "image" not in observation.payload


def test_isaaclab_stack_bc_observer_collects_robot_terms_and_optional_obj2ee():
    object_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=float)
    object_term = np.arange(39, dtype=float) * 0.01
    observer = IsaacLabStackBCObserver(
        joint_names=JOINT_NAMES,
        object_pose_provider=lambda _observer: object_pose,
        object_provider=lambda _observer: object_term,
    )
    tcp_position = np.array([0.4, -0.2, 0.6], dtype=float)
    tcp_quat = np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=float)
    last_action = np.arange(7, dtype=float) * 0.01

    observer.update_image(make_image(), name="eye_to_hand")
    observer.update_image(make_image(18), name="eye_in_hand")
    observer.update_joint_state(make_joint_state())
    observer.update_tcp_pose(tcp_position, tcp_quat)
    observer.update_gripper_width(0.035)
    observer.update_last_action(last_action)
    observation = observer.observe()

    terms = observation.payload["terms"]
    assert observation.ready
    assert terms["joint_pos"].tolist() == pytest.approx(np.linspace(0.1, 0.7, 7).tolist())
    assert terms["joint_vel"].tolist() == pytest.approx(np.linspace(-0.1, -0.7, 7).tolist())
    assert terms["eef_pos"].tolist() == pytest.approx(tcp_position.tolist())
    assert terms["eef_quat"].tolist() == pytest.approx(tcp_quat.tolist())
    assert terms["gripper_pos"].tolist() == pytest.approx([0.0175, 0.0175])
    assert terms["last_action"].tolist() == pytest.approx(last_action.tolist())
    assert terms["object_pose_in_eef"].tolist() == pytest.approx(object_pose.tolist())
    assert terms["object"].tolist() == pytest.approx(object_term.tolist())
    assert sorted(observation.payload["images"]) == ["eye_in_hand", "eye_to_hand"]


def test_isaaclab_stack_bc_observer_omits_obj2ee_when_provider_unavailable():
    observer = IsaacLabStackBCObserver(joint_names=JOINT_NAMES)
    observer.update_joint_state(make_joint_state())
    observer.update_tcp_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    observation = observer.observe()

    assert observation.ready
    assert "object_pose_in_eef" not in observation.payload["terms"]
    assert not observation.payload["availability"]["object_pose_in_eef"]


def test_isaaclab_stack_bc_observer_waits_for_configured_object_term():
    observer = IsaacLabStackBCObserver(
        joint_names=JOINT_NAMES,
        object_provider=lambda _observer: None,
    )
    observer.update_joint_state(make_joint_state())
    observer.update_tcp_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    observation = observer.observe()

    assert not observation.ready
    assert "object" not in observation.payload["terms"]
    assert not observation.payload["availability"]["object"]


def test_isaaclab_stack_bc_observer_is_not_ready_until_joint_state_and_tcp_pose_arrive():
    observer = IsaacLabStackBCObserver(joint_names=JOINT_NAMES)
    observer.update_joint_state(make_joint_state())

    assert not observer.observe().ready

    observer.update_tcp_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    assert observer.observe().ready


def test_default_obj2ee_estimator_is_explicitly_unavailable():
    assert estimate_object_pose_in_eef(None) is None


def test_color_cube_provider_estimates_target_pose_in_eef_from_rgb_depth_and_tf():
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[1, 2] = [255, 0, 0]
    depth_mm = np.zeros((3, 3), dtype=np.uint16)
    depth_mm[1, 2] = 1000
    transform = FakeTransform()
    transform.transform.translation.x = 0.1
    transform.transform.translation.y = -0.2
    transform.transform.translation.z = 0.3
    tf_buffer = FakeTfBuffer(transform)
    provider = ColorCubeObjectPoseProvider(
        target_color="red",
        camera_frame="eye_to_hand_camera_color_optical_frame",
        tcp_frame="fr3_hand_tcp",
        min_pixels=1,
    )
    observer = IsaacLabStackBCObserver(
        joint_names=JOINT_NAMES,
        object_pose_provider=provider,
    )
    observer.set_tf_buffer(tf_buffer)
    observer.update_image(make_rgb_image(rgb), name="eye_to_hand")
    observer.update_depth(make_depth_image(depth_mm), name="eye_to_hand")
    observer.update_camera_info(make_camera_info(), name="eye_to_hand")

    object_pose = provider(observer)

    assert tf_buffer.lookup_args == ("fr3_hand_tcp", "eye_to_hand_camera_color_optical_frame")
    assert object_pose.tolist() == pytest.approx([0.11, -0.2, 1.3, 0.0, 0.0, 0.0, 1.0])


def test_color_cube_provider_uses_ros_time_for_tf_lookup():
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[1, 2] = [255, 0, 0]
    depth_mm = np.zeros((3, 3), dtype=np.uint16)
    depth_mm[1, 2] = 1000
    tf_buffer = FakeStrictTfBuffer(FakeTransform())
    provider = ColorCubeObjectPoseProvider(
        target_color="red",
        camera_frame="eye_to_hand_camera_color_optical_frame",
        tcp_frame="fr3_hand_tcp",
        min_pixels=1,
    )
    observer = IsaacLabStackBCObserver(joint_names=JOINT_NAMES, object_pose_provider=provider)
    observer.set_tf_buffer(tf_buffer)
    observer.update_image(make_rgb_image(rgb), name="eye_to_hand")
    observer.update_depth(make_depth_image(depth_mm), name="eye_to_hand")
    observer.update_camera_info(make_camera_info(), name="eye_to_hand")

    object_pose = provider(observer)

    assert object_pose is not None
    assert tf_buffer.lookup_args == ("fr3_hand_tcp", "eye_to_hand_camera_color_optical_frame")
    assert tf_buffer.lookup_time is not None


def test_color_cube_provider_returns_none_when_target_color_is_absent():
    provider = ColorCubeObjectPoseProvider(target_color="blue", min_pixels=1)
    observer = IsaacLabStackBCObserver(joint_names=JOINT_NAMES, object_pose_provider=provider)
    observer.update_image(make_rgb_image(np.zeros((3, 3, 3), dtype=np.uint8)), name="eye_to_hand")
    observer.update_depth(make_depth_image(np.ones((3, 3), dtype=np.uint16) * 1000), name="eye_to_hand")
    observer.update_camera_info(make_camera_info(), name="eye_to_hand")

    assert provider(observer) is None


def test_color_cube_stack_provider_builds_isaaclab_object_observation():
    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rgb[1, 1] = [0, 0, 255]
    rgb[1, 2] = [255, 0, 0]
    rgb[1, 3] = [0, 255, 0]
    depth_mm = np.zeros((3, 5), dtype=np.uint16)
    depth_mm[1, 1] = 1000
    depth_mm[1, 2] = 1100
    depth_mm[1, 3] = 1200
    transform = FakeTransform()
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = 0.0
    tf_buffer = FakeTfBuffer(transform)
    provider = ColorCubeStackObjectProvider(min_pixels=1)
    observer = IsaacLabStackBCObserver(joint_names=JOINT_NAMES, object_provider=provider)
    observer.set_tf_buffer(tf_buffer)
    observer.update_tcp_pose(np.array([0.4, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]))
    observer.update_image(make_rgb_image(rgb), name="eye_to_hand")
    observer.update_depth(make_depth_image(depth_mm), name="eye_to_hand")
    observer.update_camera_info(make_camera_info(), name="eye_to_hand")

    object_term = provider(observer)

    assert object_term.shape == (39,)
    assert tf_buffer.lookup_args == ("fr3_link0", "eye_to_hand_camera_color_optical_frame")
    assert object_term[:21].tolist() == pytest.approx([
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.011, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0,
        0.024, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0,
    ])
    assert object_term[21:30].tolist() == pytest.approx([
        -0.4, 0.0, 0.5,
        -0.389, 0.0, 0.6,
        -0.376, 0.0, 0.7,
    ])
