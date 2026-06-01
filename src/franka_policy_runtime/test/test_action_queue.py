import numpy as np
import pytest

from franka_policy_runtime.action_queue import ActionChunk, WeightedActionQueue


def test_single_step_queue_pops_one_action():
    queue = WeightedActionQueue(action_dim=7)
    queue.replace(ActionChunk(actions=np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)))

    assert queue.size == 1
    assert queue.pop_next().tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert queue.pop_next() is None


def test_weighted_overlap_fusion_prefers_newer_chunk():
    queue = WeightedActionQueue(action_dim=7)
    old = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    new = np.array([
        [10, 0, 0, 0, 0, 0, 0],
        [20, 0, 0, 0, 0, 0, 0],
        [30, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    queue.replace(ActionChunk(actions=old))
    queue.fuse(ActionChunk(actions=new), new_weight=0.75)

    fused = [queue.pop_next()[0] for _ in range(3)]
    assert fused == pytest.approx([7.75, 15.5, 23.25])


def test_streaming_replacement_discards_old_future():
    queue = WeightedActionQueue(action_dim=7)
    queue.replace(ActionChunk(actions=np.ones((4, 7), dtype=float)))
    queue.replace(ActionChunk(actions=np.zeros((2, 7), dtype=float)))

    assert queue.size == 2
    assert queue.pop_next().tolist() == [0.0] * 7


def test_rejects_non_finite_actions():
    queue = WeightedActionQueue(action_dim=7)
    actions = np.zeros((2, 7), dtype=float)
    actions[1, 3] = np.nan

    with pytest.raises(ValueError, match="finite"):
        queue.replace(ActionChunk(actions=actions))
