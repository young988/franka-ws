import argparse
import json
import logging
import os.path
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class QuantizedOpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        attn_implementation: Optional[str],
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> None:
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")

        self.openvla_path = openvla_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        logging.warning("OpenVLA server using device: %s", self.device)

        # ---- processor ----
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)

        # ---- model ----
        model_kwargs: Dict[str, Any] = {
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "6500MiB", "cpu": "12GiB"}
            self._patch_bnb_to()
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "6500MiB", "cpu": "12GiB"}

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            **model_kwargs,
        )
        if not load_in_8bit and not load_in_4bit:
            self.vla = self.vla.to(self.device)

        # ---- rotary embedding fix (bnb puts inv_freq on CPU) ----
        self._fix_rotary_embeddings()

        # ---- norm stats ----
        # The base model embeds norm_stats in config.json — it is populated
        # automatically by OpenVLAForActionPrediction.__init__().
        # If loading a fine-tuned local directory, also check dataset_statistics.json.
        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

        if getattr(self.vla, "norm_stats", None) is None:
            logging.warning(
                "No norm_stats found — predict_action will fail if unnorm_key is used. "
                "Make sure you are using a fine-tuned model or a base model with norm_stats in config."
            )
        else:
            logging.info(
                "Loaded norm_stats: %d datasets",
                len(self.vla.norm_stats) if isinstance(self.vla.norm_stats, dict) else 1,
            )

        # ---- fastapi app ----
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)

    # ------------------------------------------------------------------
    def _fix_rotary_embeddings(self) -> None:
        """Move LlamaRotaryEmbedding.inv_freq from CPU to the model device.

        bitsandbytes 4-bit quantization + ``device_map`` leaves some buffers
        on CPU, even though the owning module is on GPU.  This causes
        ``RuntimeError: Expected all tensors to be on the same device``
        during generation.
        """
        target = self.device
        fixed = 0
        for name, module in self.vla.named_modules():
            # Module name (e.g. "language_model.model.layers.0.self_attn.rotary_emb")
            # or class name (e.g. "LlamaRotaryEmbedding") — check both.
            is_rope = (
                "rotary" in name.lower()
                or "rotary" in type(module).__name__.lower()
            )
            if is_rope and hasattr(module, "inv_freq"):
                if module.inv_freq.device != target:
                    module.inv_freq = module.inv_freq.to(target)
                    fixed += 1
        if fixed:
            logging.info("Fixed %d rotary embedding layer(s) — moved inv_freq to %s", fixed, target)

    # ------------------------------------------------------------------
    @staticmethod
    def _patch_bnb_to():
        """Patch ``PreTrainedModel.to()`` to be a no-op for bnb-quantized models.

        accelerate's ``dispatch_model()`` calls ``model.to(device)`` when
        all entries in ``device_map`` target the same device.  bitsandbytes
        4/8-bit models reject ``.to()``.  Since the model is already on the
        correct device after loading, making ``.to()`` a no-op is safe.
        """
        import transformers.modeling_utils

        _original_to = transformers.modeling_utils.PreTrainedModel.to

        def _safe_to(self, *args, **kwargs):
            if (
                getattr(self, "is_loaded_in_4bit", False)
                or getattr(self, "is_loaded_in_8bit", False)
                or getattr(self, "is_quantized", False)
                or getattr(self, "hf_quantizer", None) is not None
            ):
                return self
            return _original_to(self, *args, **kwargs)

        transformers.modeling_utils.PreTrainedModel.to = _safe_to
        return _original_to

    # ------------------------------------------------------------------
    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            image = np.asarray(payload["image"], dtype=np.uint8)
            instruction = payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                self.device, dtype=torch.bfloat16,
            )
            action = self.vla.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False,
            )
            return JSONResponse(np.asarray(action, dtype=float).tolist())
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse(
                {"error": "OpenVLA action prediction failed"}, status_code=500,
            )

    def run(self, host: str, port: int) -> None:
        uvicorn.run(self.app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla_path", default="openvla/openvla-7b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--load_in_8bit", default="false")
    parser.add_argument("--load_in_4bit", default="true")
    args = parser.parse_args()

    server = QuantizedOpenVLAServer(
        args.openvla_path,
        args.attn_implementation,
        args.load_in_8bit.lower() == "true",
        args.load_in_4bit.lower() == "true",
    )
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
