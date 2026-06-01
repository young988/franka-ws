"""Configuration helpers for the policy server."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def default_config() -> dict[str, Any]:
    """Return server defaults.

    Local hardware can only run the 4-bit OpenVLA model, so 4-bit is the
    default and 8-bit/full precision must be explicitly requested elsewhere.
    """
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
        },
        "backend": {
            "type": "openvla",
            "dummy": {
                "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
            "openvla": {
                "openvla_path": "openvla/openvla-7b",
                "attn_implementation": "sdpa",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "device": "auto",
                "max_gpu_memory": "6500MiB",
                "max_cpu_memory": "12GiB",
            },
            "python_plugin": {
                "class_path": "",
                "params": {},
            },
        },
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
