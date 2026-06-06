import base64
import io

import numpy as np
from PIL import Image
import pytest

from policy_server.backends.anygrasp import AnyGraspBackend, rotation_matrix_to_rotvec
from policy_server.config import default_config


class _FakeGrasp:
    def __init__(self, translation=(0.1, -0.2, 0.6), width=0.04):
        self.translation = np.asarray(translation, dtype=float)
        self.rotation_matrix = np.eye(3, dtype=float)
        self.width = float(width)


class _FakeGrasps:
    def __init__(self, grasps=None):
        self._grasps = list(grasps or [_FakeGrasp()])

    def __len__(self):
        return len(self._grasps)

    def nms(self):
        return self

    def sort_by_score(self):
        return self

    def __getitem__(self, index):
        return self._grasps[index]


class _FakeModel:
    def __init__(self):
        self.points = None
        self.colors = None

    def get_grasp(self, points, colors, **kwargs):
        self.points = points
        self.colors = colors
        assert kwargs["collision_detection"] is True
        return _FakeGrasps(), object()


def _image_b64(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _array_b64(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_default_config_registers_anygrasp_backend():
    config = default_config()["backend"]["anygrasp"]

    assert config["max_gripper_width"] == 0.08
    assert config["top_down_grasp"] is True
    assert len(config["workspace"]) == 6


def test_rotation_matrix_to_rotvec_handles_identity_and_quarter_turn():
    assert rotation_matrix_to_rotvec(np.eye(3)).tolist() == [0.0, 0.0, 0.0]
    rotation_z = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert rotation_matrix_to_rotvec(rotation_z) == pytest.approx(
        [0.0, 0.0, np.pi / 2.0]
    )


def test_anygrasp_backend_builds_point_cloud_and_returns_pose_width(tmp_path):
    backend = AnyGraspBackend(
        {
            "sdk_root": str(tmp_path),
            "checkpoint_path": str(tmp_path / "checkpoint.tar"),
            "max_depth": 2.0,
        },
        lazy_load=True,
    )
    backend._model = _FakeModel()
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    depth = np.array([[1000, 0], [500, 2000]], dtype=np.uint16)

    action = backend.predict_payload({
        "image_b64": _image_b64(image),
        "depth_npy_b64": _array_b64(depth),
        "camera_matrix": [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
        "depth_scale": 1000.0,
    })

    assert action == pytest.approx([0.1, -0.2, 0.6, 0.0, 0.0, 0.0, 0.04])
    assert backend._model.points.shape == (2, 3)
    assert backend._model.colors.shape == (2, 3)
    assert backend._model.points.dtype == np.float32


def test_anygrasp_selects_highest_ranked_grasp_inside_target_bbox():
    camera_matrix = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])
    grasps = _FakeGrasps([
        _FakeGrasp(translation=(-0.3, 0.0, 1.0), width=0.06),
        _FakeGrasp(translation=(0.2, 0.0, 1.0), width=0.03),
    ])

    selected = AnyGraspBackend._select_grasp(
        grasps, (60, 40, 20, 20), camera_matrix
    )

    assert selected.width == 0.03
    assert selected.translation.tolist() == [0.2, 0.0, 1.0]


def test_anygrasp_rejects_bbox_outside_image():
    with pytest.raises(ValueError, match="exceeds image size"):
        AnyGraspBackend._validate_target_bbox([90, 90, 20, 20], 100, 100)
