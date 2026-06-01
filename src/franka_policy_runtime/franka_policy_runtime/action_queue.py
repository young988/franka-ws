"""Action chunk queue and overlap fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    start_index: int = 0


def _validate_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != action_dim:
        raise ValueError(f"actions must have shape (N, {action_dim}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions must be finite")
    return arr


class WeightedActionQueue:
    """Queue of future actions with weighted replacement for overlap regions."""

    def __init__(self, action_dim: int = 7) -> None:
        self._action_dim = action_dim
        self._queue = np.zeros((0, action_dim), dtype=np.float64)

    @property
    def size(self) -> int:
        return int(self._queue.shape[0])

    def replace(self, chunk: ActionChunk) -> None:
        self._queue = _validate_actions(chunk.actions, self._action_dim).copy()

    def fuse(self, chunk: ActionChunk, new_weight: float = 0.6) -> None:
        new_actions = _validate_actions(chunk.actions, self._action_dim)
        weight = float(np.clip(new_weight, 0.0, 1.0))
        old_len = self.size
        new_len = new_actions.shape[0]
        overlap = min(old_len, new_len)

        if overlap > 0:
            fused_overlap = (1.0 - weight) * self._queue[:overlap] + weight * new_actions[:overlap]
            if new_len > overlap:
                self._queue = np.vstack([fused_overlap, new_actions[overlap:]])
            elif old_len > overlap:
                self._queue = np.vstack([fused_overlap, self._queue[overlap:]])
            else:
                self._queue = fused_overlap
        else:
            self._queue = new_actions.copy()

    def pop_next(self) -> np.ndarray | None:
        if self.size == 0:
            return None
        action = self._queue[0].copy()
        self._queue = self._queue[1:].copy()
        return action
