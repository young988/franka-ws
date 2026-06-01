"""BC backend for IsaacLab stack task payloads."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np

from policy_server.backends.base import BasePolicyBackend, validate_action


DEFAULT_REQUIRED_TERMS = ["eef_pos", "eef_quat", "gripper_pos", "object"]
DEFAULT_TERM_SHAPES = {
    "eef_pos": (3,),
    "eef_quat": (4,),
    "gripper_pos": (2,),
    "object": (39,),
}


BC_ISAACLAB_STACK_DEFAULTS = {
    "required_terms": ["eef_pos", "eef_quat", "gripper_pos", "object"],
    "term_shapes": {
        "eef_pos": [3],
        "eef_quat": [4],
        "gripper_pos": [2],
        "object": [39],
    },
    "checkpoint_path": "src/policy_server/rl_policy/bc_cube_stack/models/model_epoch_2000.pth",
    "device": "auto",
    "fallback_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


class BCIsaacLabStackBackend(BasePolicyBackend):
    backend_type = "bc_isaaclab_stack"

    @staticmethod
    def default_config() -> dict[str, Any]:
        return {
            "required_terms": list(BC_ISAACLAB_STACK_DEFAULTS["required_terms"]),
            "term_shapes": dict(BC_ISAACLAB_STACK_DEFAULTS["term_shapes"]),
            "checkpoint_path": BC_ISAACLAB_STACK_DEFAULTS["checkpoint_path"],
            "device": BC_ISAACLAB_STACK_DEFAULTS["device"],
            "fallback_action": list(BC_ISAACLAB_STACK_DEFAULTS["fallback_action"]),
        }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        self.required_terms = list(params.get("required_terms", DEFAULT_REQUIRED_TERMS))
        configured_shapes = params.get("term_shapes", DEFAULT_TERM_SHAPES)
        self.term_shapes = {name: tuple(shape) for name, shape in configured_shapes.items()}
        self.checkpoint_path = str(params.get("checkpoint_path", ""))
        self.device = str(params.get("device", "auto"))
        self._fallback_action = validate_action(
            params.get("fallback_action", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )
        self._policy = None
        self._policy_lock = threading.Lock()

    def metadata(self) -> dict[str, object]:
        data = super().metadata()
        data.update({
            "required_terms": self.required_terms,
            "term_shapes": {name: list(shape) for name, shape in self.term_shapes.items()},
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_selected_by": "best saved training loss from logs; no validation metrics were logged",
        })
        return data

    def predict_payload(self, payload: dict[str, Any]) -> np.ndarray:
        terms = payload.get("terms", {})
        if not isinstance(terms, dict):
            raise ValueError("bc_isaaclab_stack requires terms to be a mapping")
        missing = [name for name in self.required_terms if name not in terms]
        if missing:
            raise ValueError("bc_isaaclab_stack missing terms: {}".format(", ".join(missing)))
        obs = self._format_terms(terms)
        if not self.checkpoint_path:
            return self._fallback_action.copy()
        self._load_policy()
        with self._policy_lock:
            action = self._policy(obs)
        return validate_action(action)

    def _format_terms(self, terms: dict[str, Any]) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for name in self.required_terms:
            arr = np.asarray(terms[name], dtype=np.float32)
            expected_shape = self.term_shapes.get(name)
            if expected_shape is not None and arr.shape != expected_shape:
                raise ValueError(f"bc_isaaclab_stack term {name} must have shape {expected_shape}, got {arr.shape}")
            obs[name] = arr
        return obs

    def _load_policy(self) -> None:
        if self._policy is not None:
            return
        path = Path(self.checkpoint_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"bc_isaaclab_stack checkpoint not found: {path}")
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError as exc:
            raise ImportError(
                "bc_isaaclab_stack requires robomimic. Run policy_server in the isaaclab conda environment "
                "or install robomimic in the server environment."
            ) from exc
        device = TorchUtils.get_torch_device(try_to_use_cuda=True) if self.device == "auto" else self.device
        self._policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(path), device=device)
        self._policy.start_episode()
