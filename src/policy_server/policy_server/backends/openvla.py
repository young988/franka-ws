"""OpenVLA backend.

The local workstation is constrained to 4-bit OpenVLA inference, so this
backend rejects non-4-bit construction by default. Heavy ML dependencies are
imported lazily to keep tests and ROS package discovery lightweight.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from policy_server.backends.base import BasePolicyBackend, validate_action


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: str | Path) -> str:
    instruction = instruction.lower()
    if "v01" in str(openvla_path):
        return (
            f"{SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{instruction}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {instruction}?\nOut:"


class OpenVLABackend(BasePolicyBackend):
    backend_type = "openvla"

    def __init__(self, params: dict[str, Any] | None = None, *, lazy_load: bool = False) -> None:
        params = params or {}
        self.openvla_path = str(params.get("openvla_path", "openvla/openvla-7b"))
        self.attn_implementation = params.get("attn_implementation", "sdpa")
        self.load_in_4bit = bool(params.get("load_in_4bit", True))
        self.load_in_8bit = bool(params.get("load_in_8bit", False))
        self.max_gpu_memory = str(params.get("max_gpu_memory", "6500MiB"))
        self.max_cpu_memory = str(params.get("max_cpu_memory", "12GiB"))

        if not self.load_in_4bit or self.load_in_8bit:
            raise ValueError("This workstation policy requires 4-bit OpenVLA inference")

        self._processor = None
        self._model = None
        self._device = None
        if not lazy_load:
            self._load()

    def metadata(self) -> dict[str, object]:
        data = super().metadata()
        data.update({
            "openvla_path": self.openvla_path,
            "quantization": "4bit",
            "supports_chunks": False,
        })
        return data

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Work around accelerate / transformers incompatibility with
        # bitsandbytes 4-bit: the quantizer's update_device_map always
        # returns a non-None map, which triggers dispatch_model → .to().
        # The model already has bnb 4-bit params at that point (loaded by
        # _load_pretrained_model) and rejects .to().  Monkey-patch
        # dispatch_model to skip the move when there is only one target
        # device (model is already on it).
        # NOTE: transformers.modeling_utils imports dispatch_model as a
        # module-level name, so we must patch THAT reference, not just
        # accelerate.big_modeling.
        import transformers.modeling_utils as _tmu
        _orig_dispatch = _tmu.dispatch_model

        def _patched_dispatch(model, device_map, *args, **kwargs):
            unique_devices = set(device_map.values())
            gpu_devices = unique_devices - {"cpu", "disk"}
            if len(gpu_devices) <= 1:
                model.hf_device_map = dict(device_map)
                return model
            return _orig_dispatch(model, device_map, *args, **kwargs)

        _tmu.dispatch_model = _patched_dispatch
        try:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.openvla_path,
                attn_implementation=self.attn_implementation,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map=None,
            )
        finally:
            _tmu.dispatch_model = _orig_dispatch
        self._patch_rotary_embeddings(torch)

        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as stream:
                    self._model.norm_stats = json.load(stream)

    def _patch_rotary_embeddings(self, torch_module: Any) -> None:
        if self._model is None or self._device is None:
            return
        for name, module in self._model.named_modules():
            is_rope = "rotary" in name.lower() or "rotary" in type(module).__name__.lower()
            if is_rope and hasattr(module, "inv_freq") and module.inv_freq.device != self._device:
                module.inv_freq = module.inv_freq.to(self._device)
        del torch_module

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        self._load()
        import torch

        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        prompt = get_openvla_prompt(instruction, self.openvla_path)
        inputs = self._processor(prompt, Image.fromarray(image).convert("RGB")).to(
            self._device,
            dtype=torch.bfloat16,
        )
        action = self._model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return validate_action(action)
