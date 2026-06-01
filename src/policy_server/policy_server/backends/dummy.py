"""Deterministic backend used for tests and dry runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_server.backends.base import BasePolicyBackend, validate_action

DUMMY_DEFAULTS = {
    "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


class DummyBackend(BasePolicyBackend):
    backend_type = "dummy"

    @staticmethod
    def default_config() -> dict[str, Any]:
        return dict(DUMMY_DEFAULTS)

    def __init__(self, params: dict[str, object] | None = None) -> None:
        params = params or {}
        self._action = validate_action(
            params.get("action", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def predict_payload(self, payload: dict[str, Any]) -> np.ndarray:
        """Decode image (to validate payload shape) and return the fixed action."""
        _ = self._decode_image_from_payload(payload)
        return self._action.copy()

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        del image, instruction, unnorm_key
        return self._action.copy()
