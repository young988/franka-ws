from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image


SCHEMA_VERSION = "franka_openvla_rlds_source_v1"
STATE_DIM = 8
ACTION_DIM = 7
JOINT_DIM = 7


def quaternion_to_rpy_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def binary_gripper_action(width: float, *, open_threshold: float) -> float:
    """Return the Bridge/OpenVLA convention: 1=open, 0=closed."""
    return 1.0 if float(width) >= float(open_threshold) else 0.0


def openvla_state(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    gripper_width: float,
    *,
    gripper_open_threshold: float,
) -> np.ndarray:
    """Return POS_EULER state: XYZ, RPY, padding, gripper open/close."""
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {xyz.shape}")
    rpy = quaternion_to_rpy_xyzw(quaternion_xyzw)
    gripper = binary_gripper_action(
        gripper_width, open_threshold=gripper_open_threshold)
    return np.array([*xyz, *rpy, 0.0, gripper], dtype=np.float32)


def openvla_action(
    current_position: np.ndarray,
    current_quaternion_xyzw: np.ndarray,
    next_position: np.ndarray,
    next_quaternion_xyzw: np.ndarray,
    next_gripper_width: float,
    *,
    gripper_open_threshold: float,
) -> np.ndarray:
    """Relabel an action from the current observation to the next observation."""
    current_xyz = np.asarray(current_position, dtype=np.float64)
    next_xyz = np.asarray(next_position, dtype=np.float64)
    if current_xyz.shape != (3,) or next_xyz.shape != (3,):
        raise ValueError("current_position and next_position must have shape (3,)")
    current_rpy = quaternion_to_rpy_xyzw(current_quaternion_xyzw)
    next_rpy = quaternion_to_rpy_xyzw(next_quaternion_xyzw)
    delta_rpy = (next_rpy - current_rpy + math.pi) % (2.0 * math.pi) - math.pi
    gripper = binary_gripper_action(
        next_gripper_width, open_threshold=gripper_open_threshold)
    return np.array(
        [*(next_xyz - current_xyz), *delta_rpy, gripper],
        dtype=np.float32,
    )


def center_crop_resize_rgb(image: np.ndarray, output_size: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (H, W, 3), got {rgb.shape}")
    height, width = rgb.shape[:2]
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    cropped = rgb[start_y:start_y + side, start_x:start_x + side]
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(cropped, mode="RGB").resize(
        (int(output_size), int(output_size)),
        resample=resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.uint8)


class OpenVLAEpisodeWriter:
    """Incrementally write one raw episode consumed by the bundled TFDS builder."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        dataset_name: str,
        instruction: str,
        image_size: int,
        has_wrist_image: bool,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.dataset_name = _safe_dataset_name(dataset_name)
        self.raw_root = self.dataset_root / self.dataset_name / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.episode_id = _next_episode_id(self.raw_root)
        self.instruction = str(instruction).strip()
        if not self.instruction:
            raise ValueError("instruction must not be empty")
        self.image_size = int(image_size)
        if self.image_size != 256:
            raise ValueError(
                "OpenVLA observations must be 256x256; image_size must be 256"
            )
        self.has_wrist_image = bool(has_wrist_image)
        self._temporary_dir = self.raw_root / f".episode_{self.episode_id:06d}.in_progress"
        self.final_dir = self.raw_root / f"episode_{self.episode_id:06d}"
        self._temporary_dir.mkdir(parents=False, exist_ok=False)
        (self._temporary_dir / "images").mkdir()
        if self.has_wrist_image:
            (self._temporary_dir / "wrist_images").mkdir()

        self._states: list[np.ndarray] = []
        self._joint_positions: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._image_paths: list[str] = []
        self._wrist_image_paths: list[str] = []

    @property
    def step_count(self) -> int:
        return len(self._actions)

    def append(
        self,
        *,
        image_rgb: np.ndarray,
        wrist_image_rgb: np.ndarray | None,
        state: np.ndarray,
        joint_positions: np.ndarray,
        action: np.ndarray,
        timestamp_sec: float,
    ) -> None:
        state_array = _finite_vector(state, STATE_DIM, "state")
        joint_array = _finite_vector(joint_positions, JOINT_DIM, "joint_positions")
        action_array = _finite_vector(action, ACTION_DIM, "action")
        index = self.step_count
        image_relative = f"images/{index:06d}.jpg"
        _write_jpeg_rgb(self._temporary_dir / image_relative, image_rgb)

        wrist_relative = ""
        if self.has_wrist_image:
            if wrist_image_rgb is None:
                raise ValueError("wrist image is required for this episode")
            wrist_relative = f"wrist_images/{index:06d}.jpg"
            _write_jpeg_rgb(self._temporary_dir / wrist_relative, wrist_image_rgb)

        self._states.append(state_array)
        self._joint_positions.append(joint_array)
        self._actions.append(action_array)
        self._timestamps.append(float(timestamp_sec))
        self._image_paths.append(image_relative)
        self._wrist_image_paths.append(wrist_relative)

    def finalize(self) -> Path:
        if self.step_count < 1:
            raise ValueError("cannot finalize an episode without transitions")
        np.savez_compressed(
            self._temporary_dir / "steps.npz",
            state=np.stack(self._states).astype(np.float32),
            joint_positions=np.stack(self._joint_positions).astype(np.float32),
            action=np.stack(self._actions).astype(np.float32),
            timestamp_sec=np.asarray(self._timestamps, dtype=np.float64),
            image_path=np.asarray(self._image_paths),
            wrist_image_path=np.asarray(self._wrist_image_paths),
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "num_steps": self.step_count,
            "image_size": self.image_size,
            "has_wrist_image": self.has_wrist_image,
            "state_encoding": "POS_EULER",
            "state_layout": [
                "x", "y", "z", "roll", "pitch", "yaw", "padding", "gripper_open"
            ],
            "action_encoding": "EEF_POS",
            "action_layout": [
                "delta_x", "delta_y", "delta_z",
                "delta_roll", "delta_pitch", "delta_yaw", "gripper_open",
            ],
            "gripper_semantics": {"open": 1.0, "closed": 0.0},
        }
        (self._temporary_dir / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(self._temporary_dir, self.final_dir)
        return self.final_dir

    def abort(self) -> None:
        """Remove an unfinished episode created by this writer."""
        if self._temporary_dir.is_dir():
            shutil.rmtree(self._temporary_dir)


def _safe_dataset_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    if not normalized:
        raise ValueError("dataset_name must contain letters or digits")
    return normalized


def _next_episode_id(raw_root: Path) -> int:
    episode_ids = []
    for path in raw_root.iterdir():
        match = re.search(r"episode_(\d{6})", path.name)
        if match:
            episode_ids.append(int(match.group(1)))
    return max(episode_ids, default=-1) + 1


def _finite_vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array.copy()


def _write_jpeg_rgb(path: Path, image_rgb: np.ndarray) -> None:
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {rgb.shape}")
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG", quality=95)
