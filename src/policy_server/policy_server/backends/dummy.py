"""Deterministic backend used for tests and dry runs."""

from __future__ import annotations

import numpy as np

from policy_server.backends.base import BasePolicyBackend, validate_action


class DummyBackend(BasePolicyBackend):
    backend_type = "dummy"

    def __init__(self, params: dict[str, object] | None = None) -> None:
        params = params or {}
        self._action = validate_action(
            params.get("action", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        del image, instruction, unnorm_key
        return self._action.copy()
