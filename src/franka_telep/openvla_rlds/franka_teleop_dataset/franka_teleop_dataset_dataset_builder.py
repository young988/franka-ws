from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow_datasets as tfds


class FrankaTeleopDataset(tfds.core.GeneratorBasedBuilder):
    """OpenVLA-compatible RLDS dataset from franka_telep raw episodes."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial Franka uArm teleoperation dataset."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Primary eye-to-hand RGB image.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist RGB image, or primary image when unavailable.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="EEF XYZ, RPY, padding, binary gripper open/close.",
                        ),
                        "joint_positions": tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc="Measured FR3 joint positions in radians.",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc="Base-frame EEF delta XYZ, delta RPY, binary gripper.",
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    "language_embedding": tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc="Zero placeholder; OpenVLA consumes language_instruction.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "instruction": tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        default_root = Path("~/franka_openvla_data/franka_teleop/raw").expanduser()
        raw_root = Path(
            os.environ.get("FRANKA_TELEOP_RAW_DIR", str(default_root))
        ).expanduser()
        return {"train": self._generate_examples(raw_root)}

    def _generate_examples(self, raw_root: Path) -> Iterator[tuple[str, Any]]:
        for episode_dir in sorted(raw_root.glob("episode_*")):
            metadata_path = episode_dir / "episode.json"
            steps_path = episode_dir / "steps.npz"
            if not metadata_path.is_file() or not steps_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(steps_path, allow_pickle=False) as steps:
                states = steps["state"].astype(np.float32)
                joints = steps["joint_positions"].astype(np.float32)
                actions = steps["action"].astype(np.float32)
                actions[:, 3:6] = (
                    actions[:, 3:6] + np.pi
                ) % (2.0 * np.pi) - np.pi
                image_paths = steps["image_path"]
                wrist_image_paths = steps["wrist_image_path"]
                timestamps = steps["timestamp_sec"]
                del timestamps
                step_count = len(actions)
                if not (
                    len(states)
                    == len(joints)
                    == len(image_paths)
                    == len(wrist_image_paths)
                    == step_count
                ):
                    raise ValueError(f"inconsistent arrays in {episode_dir}")
                episode = []
                for index in range(step_count):
                    is_last = index == step_count - 1
                    primary_path = episode_dir / str(image_paths[index])
                    wrist_relative = str(wrist_image_paths[index])
                    wrist_path = (
                        episode_dir / wrist_relative
                        if wrist_relative
                        else primary_path
                    )
                    episode.append({
                        "observation": {
                            "image": str(primary_path),
                            "wrist_image": str(wrist_path),
                            "state": states[index],
                            "joint_positions": joints[index],
                        },
                        "action": actions[index],
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": index == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": str(metadata["instruction"]),
                        "language_embedding": np.zeros(512, dtype=np.float32),
                    })
            episode_key = f"episode_{int(metadata['episode_id']):06d}"
            yield episode_key, {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(episode_dir),
                    "instruction": str(metadata["instruction"]),
                },
            }
