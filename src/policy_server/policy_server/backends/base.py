"""Base policy backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import base64
import io
from typing import Any

import numpy as np
from PIL import Image as PILImage


@dataclass(frozen=True)
class PolicyMetadata:
    backend_type: str
    action_dim: int = 7
    supports_chunks: bool = False


class BasePolicyBackend(ABC):
    """Interface implemented by all inference backends.

    Subclasses are auto-registered via ``backend_type``.  The single
    abstract entry point is :meth:`predict_payload` — one HTTP body →
    one 7D action.
    """

    backend_type = "base"
    _registry: dict[str, type[BasePolicyBackend]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        bt = getattr(cls, "backend_type", "")
        if bt and bt != "base":
            cls._registry[bt] = cls

    # ------------------------------------------------------------------
    # Abstract entry point
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_payload(self, payload: dict[str, Any]) -> np.ndarray:
        """Return one 7D action from the backend-specific HTTP payload."""

    # ------------------------------------------------------------------
    # Convenience method for image-based backends
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        """Convenience for image-only inference.  Not all backends support this."""
        raise NotImplementedError(
            f"{self.backend_type} backend requires a full payload; use predict_payload"
        )

    @staticmethod
    def _decode_image_from_payload(payload: dict[str, Any]) -> np.ndarray:
        """Extract and decode a JPEG-encoded (or raw) image from a payload dict."""
        if "image_b64" in payload:
            raw = base64.b64decode(str(payload["image_b64"]))
            return np.asarray(PILImage.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        if "image" in payload:
            return np.asarray(payload["image"], dtype=np.uint8)
        raise ValueError("policy payload does not contain 'image_b64' or 'image'")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, object]:
        return PolicyMetadata(backend_type=self.backend_type).__dict__.copy()


def validate_action(action: object) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"policy action must have shape (7,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("policy action must be finite")
    return arr
