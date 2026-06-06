"""Unit tests for ActionTesterRuntime helpers."""
import numpy as np
import pytest

from franka_policy_runtime.utils.pose_math import DummyObserver, action_dim_label
from franka_policy_runtime.observers.base import BackendObservation


def test_dummy_observer_always_ready():
    observer = DummyObserver(joint_names=["fr3_joint1"])
    result = observer.observe()
    assert isinstance(result, BackendObservation)
    assert result.ready is True
    assert result.payload == {}


def test_dummy_observer_inherits_base_methods():
    from types import SimpleNamespace

    observer = DummyObserver(joint_names=["fr3_joint1", "fr3_joint2"])
    assert observer.latest_joint_positions() is None

    msg = SimpleNamespace(
        name=["fr3_joint1", "fr3_joint2"],
        position=[0.1, 0.2],
        velocity=[0.0, 0.0],
    )
    observer.update_joint_state(msg)
    pos = observer.latest_joint_positions()
    assert pos is not None
    assert pos.tolist() == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize(
    "action,expected",
    [
        ([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "+dx"),
        ([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "-dx"),
        ([0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0], "+dy"),
        ([0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0], "-dy"),
        ([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0], "+dz"),
        ([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0], "-dz"),
        ([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], "+rx"),
        ([0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0], "-rx"),
        ([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], "+ry"),
        ([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0], "-ry"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], "+rz"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0], "-rz"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "gripper_open"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5], "gripper_close"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "zero"),
    ],
)
def testaction_dim_label(action, expected):
    assert action_dim_label(np.array(action, dtype=float)) == expected


def testaction_dim_label_multi_dim():
    action = np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    assert action_dim_label(action) == "+dx,+dy"
