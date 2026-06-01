"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field


FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "single_step"
    observer_type: str = "vla"
    policy_url: str = "http://127.0.0.1:8000/act"
    instruction_topic: str = "~/instruction"
    reference_topic: str = "/franka_policy_controller/reference"
    joint_names: list[str] = field(default_factory=lambda: FR3_JOINT_NAMES.copy())
    command_frame: str = "fr3_link0"
    tcp_frame: str = "fr3_hand_tcp"
    move_group_name: str = "fr3_arm"
    control_period_sec: float = 0.2
    actions_per_chunk: int = 1
    chunk_size_threshold: float = 0.5
    fusion_new_weight: float = 0.6
    max_translation_delta: float = 0.05
    max_rotation_delta: float = 0.25
    gripper_width: float = 0.04
