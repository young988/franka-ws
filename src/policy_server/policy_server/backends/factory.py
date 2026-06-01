"""Policy backend factory — discovers backends from the registry."""

from __future__ import annotations

from policy_server.config import merge_config
from policy_server.backends.base import BasePolicyBackend

# Import all backend modules so they self-register via __init_subclass__.
import policy_server.backends.bc_isaaclab_stack  # noqa: F401
import policy_server.backends.dummy               # noqa: F401
import policy_server.backends.openvla             # noqa: F401
import policy_server.backends.python_plugin       # noqa: F401


def create_backend(config: dict[str, object] | None = None) -> BasePolicyBackend:
    merged = merge_config({"backend": config} if config else None)
    backend_config = merged["backend"]
    backend_type = str(backend_config["type"])

    cls = BasePolicyBackend._registry.get(backend_type)
    if cls is not None:
        return cls(backend_config.get(backend_type, {}))

    raise ValueError(f"unknown policy backend type: {backend_type}")
