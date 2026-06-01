"""Generic Python backend plugin loader."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from policy_server.backends.base import BasePolicyBackend, validate_action

PYTHON_PLUGIN_DEFAULTS = {
    "class_path": "",
    "params": {},
}


class PythonPluginBackend(BasePolicyBackend):
    backend_type = "python_plugin"

    @staticmethod
    def default_config() -> dict[str, Any]:
        return dict(PYTHON_PLUGIN_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        class_path = str(params.get("class_path", ""))
        if ":" not in class_path:
            raise ValueError("python_plugin.class_path must use 'module:ClassName'")
        module_name, class_name = class_path.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self._plugin = cls(params.get("params", {}))

    def predict_payload(self, payload: dict[str, Any]) -> np.ndarray:
        if hasattr(self._plugin, "predict_payload"):
            action = self._plugin.predict_payload(payload)
            return validate_action(action)
        # Fall back to image-based predict if the plugin doesn't handle payloads.
        image = self._decode_image_from_payload(payload)
        return self.predict(
            image,
            str(payload.get("instruction", "")),
            payload.get("unnorm_key"),
        )

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        if hasattr(self._plugin, "predict"):
            action = self._plugin.predict(image, instruction, unnorm_key)
        else:
            action = self._plugin(image, instruction, unnorm_key)
        return validate_action(action)
