"""OpenVLA observer — image + instruction observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from franka_policy_runtime.observers.base import BackendObservation, BaseObserver


class OpenVLAObserver(BaseObserver):
    """Observation schema for OpenVLA: primary image plus instruction."""

    def __init__(self, joint_names: list[str] | None = None, instruction: str = "") -> None:
        super().__init__(joint_names)
        self._instruction = str(instruction)

    def update_instruction(self, msg: Any) -> None:
        with self._lock:
            self._instruction = str(msg.data)

    def observe(self) -> BackendObservation:
        with self._lock:
            image = self._primary_image_locked()
            instruction = self._instruction
        payload: dict[str, Any] = {"instruction": instruction}
        if image is not None:
            payload["image"] = image
        return BackendObservation(ready=image is not None, payload=payload)
