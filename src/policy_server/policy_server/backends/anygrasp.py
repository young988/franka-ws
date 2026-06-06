"""AnyGrasp RGB-D inference backend."""
from __future__ import annotations
import base64
from contextlib import contextmanager
import io
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any, Iterator
import numpy as np
from policy_server.backends.base import BasePolicyBackend, validate_action
ANYGRASP_DEFAULTS = {
    "sdk_root": "",
    "checkpoint_path": "",
    "max_gripper_width": 0.08,
    "gripper_height": 0.03,
    "top_down_grasp": True,
    "depth_scale": 1000.0,
    "max_depth": 1.5,
    "workspace": [-0.5, 0.5, -0.5, 0.5, 0.05, 1.5],
    "apply_object_mask": True,
    "dense_grasp": False,
    "collision_detection": True,
}
def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis-angle rotation vector."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {rotation.shape}")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1.0e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < 1.0e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            for index in range(3):
                if index != largest:
                    axis[index] = (
                        rotation[index, largest] + rotation[largest, index]
                    ) / (4.0 * axis[largest])
            axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64) / (2.0 * np.sin(angle))
    return axis * angle
@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
class AnyGraspBackend(BasePolicyBackend):
    """Return an absolute camera-frame grasp as xyz + rotvec + width."""
    backend_type = "anygrasp"
    @staticmethod
    def default_config() -> dict[str, Any]:
        return dict(ANYGRASP_DEFAULTS)
    def __init__(self, params: dict[str, Any] | None = None, *, lazy_load: bool = False) -> None:
        config = dict(ANYGRASP_DEFAULTS)
        config.update(params or {})
        sdk_root_value = str(config["sdk_root"] or os.environ.get("ANYGRASP_SDK_ROOT", ""))
        if not sdk_root_value:
            sdk_root_value = "src/anygrasp_sdk"
        self.sdk_root = Path(sdk_root_value).expanduser().resolve()
        self.detection_dir = self.sdk_root / "grasp_detection"
        checkpoint_value = str(config["checkpoint_path"] or "")
        checkpoint = (
            Path(checkpoint_value).expanduser()
            if checkpoint_value
            else self.detection_dir / "log" / "checkpoint_detection.tar"
        )
        self.checkpoint_path = checkpoint.resolve()
        self.max_gripper_width = float(np.clip(config["max_gripper_width"], 0.0, 0.1))
        self.gripper_height = float(config["gripper_height"])
        self.top_down_grasp = bool(config["top_down_grasp"])
        self.depth_scale = float(config["depth_scale"])
        self.max_depth = float(config["max_depth"])
        self.workspace = self._validate_workspace(config["workspace"])
        self.apply_object_mask = bool(config["apply_object_mask"])
        self.dense_grasp = bool(config["dense_grasp"])
        self.collision_detection = bool(config["collision_detection"])
        self._model = None
        self._inference_lock = threading.Lock()
        if not lazy_load:
            self._load()
    @staticmethod
    def _validate_workspace(value: object) -> np.ndarray:
        workspace = np.asarray(value, dtype=np.float64)
        if workspace.shape != (6,):
            raise ValueError(f"workspace must contain 6 values, got {workspace.shape}")
        if not np.all(np.isfinite(workspace)):
            raise ValueError("workspace must be finite")
        if np.any(workspace[::2] >= workspace[1::2]):
            raise ValueError("workspace minima must be smaller than maxima")
        return workspace
    @staticmethod
    def _decode_depth(payload: dict[str, Any]) -> np.ndarray:
        if "depth_npy_b64" in payload:
            raw = base64.b64decode(str(payload["depth_npy_b64"]))
            return np.load(io.BytesIO(raw), allow_pickle=False)
        if "depth" in payload:
            return np.asarray(payload["depth"])
        raise ValueError("AnyGrasp payload requires 'depth_npy_b64' or 'depth'")
    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.detection_dir.is_dir():
            raise FileNotFoundError(f"AnyGrasp detection directory not found: {self.detection_dir}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AnyGrasp checkpoint not found: {self.checkpoint_path}")
        detection_path = str(self.detection_dir)
        if detection_path not in sys.path:
            sys.path.insert(0, detection_path)
        with _working_directory(self.detection_dir):
            from gsnet import AnyGrasp
            options = SimpleNamespace(
                checkpoint_path=str(self.checkpoint_path),
                max_gripper_width=self.max_gripper_width,
                gripper_height=self.gripper_height,
                top_down_grasp=self.top_down_grasp,
                debug=False,
            )
            self._model = AnyGrasp(options)
            self._model.load_net()
    def predict_payload(self, payload: dict[str, Any]) -> np.ndarray:
        image = self._decode_image_from_payload(payload)
        depth = self._decode_depth(payload)
        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB and depth dimensions differ: {image.shape[:2]} vs {depth.shape}"
            )
        camera_matrix = np.asarray(payload.get("camera_matrix"), dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix must have shape (3, 3), got {camera_matrix.shape}"
            )
        depth_scale = float(payload.get("depth_scale", self.depth_scale))
        if depth_scale <= 0.0:
            raise ValueError("depth_scale must be positive")
        workspace = self._validate_workspace(payload.get("workspace", self.workspace))
        points, colors = self._make_point_cloud(image, depth, camera_matrix, depth_scale)
        target_bbox = self._validate_target_bbox(
            payload.get("target_bbox"), image.shape[1], image.shape[0]
        )
        self._load()
        assert self._model is not None
        with self._inference_lock:
            grasps, _ = self._model.get_grasp(
                points,
                colors,
                lims=workspace.tolist(),
                apply_object_mask=self.apply_object_mask,
                dense_grasp=self.dense_grasp,
                collision_detection=self.collision_detection,
            )
        if len(grasps) == 0:
            raise ValueError("AnyGrasp found no collision-free grasp")
        grasp = self._select_grasp(
            grasps.nms().sort_by_score(), target_bbox, camera_matrix
        )
        action = np.concatenate([
            np.asarray(grasp.translation, dtype=np.float64),
            rotation_matrix_to_rotvec(grasp.rotation_matrix),
            np.array([float(grasp.width)], dtype=np.float64),
        ])
        return validate_action(action)
    @staticmethod
    def _validate_target_bbox(
        value: object,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        if value is None:
            return None
        bbox = np.asarray(value, dtype=np.int64)
        if bbox.shape != (4,):
            raise ValueError(f"target_bbox must have shape (4,), got {bbox.shape}")
        x, y, width, height = (int(item) for item in bbox)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("target_bbox must be [x>=0, y>=0, width>0, height>0]")
        if x + width > image_width or y + height > image_height:
            raise ValueError(
                f"target_bbox {bbox.tolist()} exceeds image size "
                f"{image_width}x{image_height}"
            )
        return x, y, width, height
    @staticmethod
    def _select_grasp(grasps, target_bbox, camera_matrix: np.ndarray):
        if target_bbox is None:
            return grasps[0]
        x, y, width, height = target_bbox
        fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
        cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
        for index in range(len(grasps)):
            grasp = grasps[index]
            translation = np.asarray(grasp.translation, dtype=np.float64)
            if translation.shape != (3,) or translation[2] <= 0.0:
                continue
            pixel_x = fx * translation[0] / translation[2] + cx
            pixel_y = fy * translation[1] / translation[2] + cy
            if x <= pixel_x < x + width and y <= pixel_y < y + height:
                return grasp
        raise ValueError("AnyGrasp found no collision-free grasp inside target_bbox")
    def _make_point_cloud(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        camera_matrix: np.ndarray,
        depth_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        depth_m = np.asarray(depth, dtype=np.float32) / depth_scale
        height, width = depth_m.shape
        xmap, ymap = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
        cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        points_z = depth_m
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        valid = np.isfinite(points_z) & (points_z > 0.0) & (points_z < self.max_depth)
        if not np.any(valid):
            raise ValueError("depth image contains no valid points")
        points = np.stack([points_x, points_y, points_z], axis=-1)[valid]
        colors = np.asarray(image, dtype=np.float32)[valid] / 255.0
        return points.astype(np.float32), colors.astype(np.float32)
    def metadata(self) -> dict[str, object]:
        data = super().metadata()
        data.update({
            "action_semantics": "camera_xyz_rotvec_gripper_width",
            "checkpoint_path": str(self.checkpoint_path),
            "sdk_root": str(self.sdk_root),
        })
        return data
