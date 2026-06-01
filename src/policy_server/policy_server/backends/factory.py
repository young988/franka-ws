"""Policy backend factory."""

from __future__ import annotations

from policy_server.config import merge_config
from policy_server.backends.base import BasePolicyBackend
from policy_server.backends.dummy import DummyBackend


def create_backend(config: dict[str, object] | None = None) -> BasePolicyBackend:
    merged = merge_config({"backend": config} if config else None)
    backend_config = merged["backend"]
    backend_type = str(backend_config["type"])

    if backend_type == "dummy":
        return DummyBackend(backend_config.get("dummy", {}))
    if backend_type == "openvla":
        from policy_server.backends.openvla import OpenVLABackend

        return OpenVLABackend(backend_config.get("openvla", {}))
    if backend_type == "python_plugin":
        from policy_server.backends.python_plugin import PythonPluginBackend

        return PythonPluginBackend(backend_config.get("python_plugin", {}))
    raise ValueError(f"unknown policy backend type: {backend_type}")
