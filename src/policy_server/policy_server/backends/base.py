"""Base policy backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyMetadata:
    backend_type: str
    action_dim: int = 7
    supports_chunks: bool = False


class BasePolicyBackend(ABC):
    """Interface implemented by all inference backends."""

    backend_type = "base"

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        """Return one 7D action."""

    def metadata(self) -> dict[str, object]:
        return PolicyMetadata(backend_type=self.backend_type).__dict__.copy()


def validate_action(action: object) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"policy action must have shape (7,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("policy action must be finite")
    return arr
