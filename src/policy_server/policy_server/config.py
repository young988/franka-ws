"""Configuration helpers for the policy server."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


# Import all backends so they self-register and we can collect their defaults.
import policy_server.backends.bc_isaaclab_stack  # noqa: F401
import policy_server.backends.dummy               # noqa: F401
import policy_server.backends.openvla             # noqa: F401
import policy_server.backends.python_plugin       # noqa: F401

from policy_server.backends.base import BasePolicyBackend


def default_config() -> dict[str, Any]:
    """Return server defaults collected from all registered backends."""
    backend_defaults: dict[str, Any] = {"type": "openvla"}
    for name, cls in BasePolicyBackend._registry.items():
        if hasattr(cls, "default_config"):
            backend_defaults[name] = cls.default_config()

    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
        },
        "backend": backend_defaults,
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def merge_config(override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = default_config()
    if override:
        _deep_merge(config, override)
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return default_config()

    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("policy server config must be a YAML mapping")
    return merge_config(loaded)
