# Cartesian Pose Runtime Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current IK + `JointTrajectory`-based FR3 policy runtime with a direct Franka Cartesian pose control pipeline while preserving the current 7D policy action semantics.

**Architecture:** Keep `PolicyRuntimeBase` as the orchestration shell, but move arm control into a dedicated Cartesian backend that maintains `target_pose` and `commanded_pose`, interpolates on each control tick, converts policy TCP targets into Franka `O_T_EE`, and publishes pose commands to a new ros2_control Cartesian pose controller. Remove MoveIt IK and joint-reference publishing from the normal runtime path.

**Tech Stack:** ROS 2 Humble, `rclpy`, ros2_control controller plugins in C++17, Franka ROS 2 Cartesian pose command interfaces, `pytest`, `colcon`.

---

## File structure

### Files to create
- `src/franka_policy_runtime/franka_policy_runtime/cartesian_backend.py` — owns pose-state management, interpolation, reset/resync, and TCP→EE conversion.
- `src/franka_policy_runtime/test/test_cartesian_backend.py` — unit tests for target pose accumulation, commanded pose stepping, reset/resync, and pose conversion.
- `src/franka_policy_runtime/test/test_cartesian_controller_launch.py` — launch/source-level assertions for the new controller wiring.
- `src/franka_policy_controller/include/franka_policy_controller/franka_cartesian_pose_controller.hpp` — controller plugin interface for Cartesian pose references.
- `src/franka_policy_controller/src/franka_cartesian_pose_controller.cpp` — controller plugin implementation subscribing to pose commands and writing `cartesian_pose_command`.
- `src/franka_policy_controller/config/franka_bringup_cartesian_pose_controllers.yaml` — ros2_control config loading the new Cartesian controller.

### Files to modify
- `src/franka_policy_runtime/franka_policy_runtime/reference.py` — replace ambiguous helpers with explicit base-frame pose math helpers and interpolation utilities.
- `src/franka_policy_runtime/franka_policy_runtime/base_node.py` — remove IK/JointTrajectory arm path and route control ticks through the Cartesian backend + pose command publisher.
- `src/franka_policy_runtime/config/franka_policy_runtime.yaml` — remove IK/joint-reference params, add Cartesian interpolation and frame-mapping params.
- `src/franka_policy_runtime/launch/robot_base.launch.py` — load the Cartesian controller config instead of the joint-reference controller and remove `move_group` from the runtime path.
- `src/franka_policy_runtime/launch/vla_policy.launch.py` — update launch comments/arguments if control topics or params rename.
- `src/franka_policy_runtime/launch/bc_cube_stack.launch.py` — same as above.
- `src/franka_policy_runtime/test/test_runtime_config.py` — update pose-math tests to explicit helper names and add step-limit utility coverage.
- `src/franka_policy_runtime/test/test_policy_runtime_executor.py` — replace IK-specific assertions with Cartesian-backend assertions.
- `src/franka_policy_runtime/test/test_policy_launch_files.py` — extend checks to assert MoveIt is no longer part of the runtime path.
- `src/franka_policy_runtime/setup.py` — only if new module placement or entry points require packaging tweaks.
- `src/franka_policy_runtime/package.xml` — drop `moveit_msgs` / `trajectory_msgs` runtime deps if no longer needed; keep only required deps.
- `src/franka_policy_controller/CMakeLists.txt` — build the new controller source file.
- `src/franka_policy_controller/package.xml` — replace `trajectory_msgs` dependency with `geometry_msgs` where needed.
- `src/franka_policy_controller/franka_policy_controller_plugin.xml` — export the new controller plugin class.
- `src/franka_policy_controller/config/franka_policy_controller.yaml` — repurpose or replace package-level config to the new Cartesian controller parameters.
- `src/franka_policy_runtime/docs/superpowers/specs/2026-06-03-cartesian-pose-runtime-design.md` — no code changes expected; only update if implementation uncovers a spec mismatch.

### Files likely to delete
- `src/franka_policy_controller/include/franka_policy_controller/franka_policy_controller.hpp` — if the old joint-reference controller is fully removed.
- `src/franka_policy_controller/src/franka_policy_controller.cpp` — same.
- `src/franka_policy_controller/config/franka_bringup_policy_controllers.yaml` — replaced by Cartesian version.

If you keep the old files temporarily during the migration, do not delete them until the new controller is compiled, launched, and tested.

---

### Task 1: Refactor pose math into explicit Cartesian helpers

**Files:**
- Modify: `src/franka_policy_runtime/franka_policy_runtime/reference.py`
- Modify: `src/franka_policy_runtime/test/test_runtime_config.py`

- [ ] **Step 1: Write the failing tests for explicit base-frame helper names and per-tick step limiting**

Replace the old `apply_tcp_delta`/`clamp_joint_step` expectations in `src/franka_policy_runtime/test/test_runtime_config.py` with these tests:

```python
import numpy as np
import pytest

from franka_policy_runtime.runtime_config import FR3_JOINT_NAMES
from franka_policy_runtime.reference import (
    _quat_xyzw_from_axis_angle,
    apply_tcp_delta_in_base_frame,
    gripper_width_from_binary_action,
    split_policy_action,
    step_toward_pose,
)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    quat = np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=float)
    return quat / np.linalg.norm(quat)


def _same_orientation(actual: np.ndarray, expected: np.ndarray) -> bool:
    return abs(float(np.dot(actual, expected))) == pytest.approx(1.0)


def test_fr3_joint_names_has_7_joints():
    assert len(FR3_JOINT_NAMES) == 7
    assert all(name.startswith("fr3_joint") for name in FR3_JOINT_NAMES)


def test_split_policy_action_separates_tcp_delta_and_gripper():
    action = np.array([0.01, 0.02, -0.03, 0.1, -0.2, 0.3, 0.04], dtype=float)
    tcp_delta, gripper_delta = split_policy_action(action)
    assert tcp_delta.tolist() == [0.01, 0.02, -0.03, 0.1, -0.2, 0.3]
    assert gripper_delta == 0.04


def test_apply_tcp_delta_in_base_frame_scales_translation_and_rotation():
    position = np.zeros(3, dtype=float)
    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    action = np.array([0.2, -0.2, 0.04, 0.0, 0.0, np.pi, 1.0], dtype=float)

    target_position, target_quat = apply_tcp_delta_in_base_frame(
        position,
        quat_xyzw,
        action,
        action_scale=0.5,
    )

    assert target_position.tolist() == pytest.approx([0.1, -0.1, 0.02])
    expected_delta = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi * 0.5], dtype=float))
    assert _same_orientation(target_quat, expected_delta)


def test_apply_tcp_delta_in_base_frame_composes_axis_angle_in_command_frame():
    position = np.zeros(3, dtype=float)
    current_quat = _quat_xyzw_from_axis_angle(np.array([0.3, -0.4, 0.2], dtype=float))
    axis_angle_delta = np.array([0.2, 0.1, -0.3], dtype=float)
    action = np.array([0.0, 0.0, 0.0, *axis_angle_delta, 0.0], dtype=float)

    _, target_quat = apply_tcp_delta_in_base_frame(
        position,
        current_quat,
        action,
        action_scale=1.0,
    )

    delta_quat = _quat_xyzw_from_axis_angle(axis_angle_delta)
    expected_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    assert _same_orientation(target_quat, expected_quat)


def test_step_toward_pose_limits_translation_and_rotation_per_tick():
    current_position = np.array([0.0, 0.0, 0.0], dtype=float)
    current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    target_position = np.array([0.3, 0.0, 0.0], dtype=float)
    target_quat = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi / 2], dtype=float))

    next_position, next_quat = step_toward_pose(
        current_position=current_position,
        current_quat_xyzw=current_quat,
        target_position=target_position,
        target_quat_xyzw=target_quat,
        max_translation_step=0.1,
        max_rotation_step=np.pi / 6,
    )

    assert next_position.tolist() == pytest.approx([0.1, 0.0, 0.0])
    expected_quat = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi / 6], dtype=float))
    assert _same_orientation(next_quat, expected_quat)


def test_gripper_width_from_binary_action_matches_sign_semantics():
    assert gripper_width_from_binary_action(-0.1, min_width=0.0, max_width=0.08) == 0.0
    assert gripper_width_from_binary_action(0.0, min_width=0.0, max_width=0.08) == 0.08
```

- [ ] **Step 2: Run the focused tests to verify they fail with missing helper names**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_runtime_config.py -q
```

Expected: FAIL with import errors for `apply_tcp_delta_in_base_frame` and `step_toward_pose`.

- [ ] **Step 3: Implement the explicit helper API and stepping logic in `reference.py`**

Edit `src/franka_policy_runtime/franka_policy_runtime/reference.py` so the public arm helpers become:

```python
def apply_tcp_delta_in_base_frame(
    current_position: np.ndarray,
    current_quat_xyzw: np.ndarray,
    action: np.ndarray,
    *,
    action_scale: float,
    rotation_format: str = "axis_angle",
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(current_position, dtype=np.float64)
    quat_xyzw = np.asarray(current_quat_xyzw, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"current_position must have shape (3,), got {position.shape}")
    if quat_xyzw.shape != (4,):
        raise ValueError(f"current_quat_xyzw must have shape (4,), got {quat_xyzw.shape}")

    tcp_delta, _ = split_policy_action(action)
    scaled_delta = tcp_delta * float(action_scale)
    translation = scaled_delta[:3]
    rotation_delta = scaled_delta[3:6]
    target_position = position + translation
    current_quat = quat_xyzw / np.linalg.norm(quat_xyzw)
    if rotation_format == "rpy":
        delta_quat = _quat_xyzw_from_rpy(rotation_delta)
    elif rotation_format == "axis_angle":
        delta_quat = _quat_xyzw_from_axis_angle(rotation_delta)
    else:
        raise ValueError(f"unknown rotation_format: {rotation_format!r}")
    target_quat = _quat_multiply_xyzw(delta_quat, current_quat)
    return target_position, target_quat


def step_toward_pose(
    *,
    current_position: np.ndarray,
    current_quat_xyzw: np.ndarray,
    target_position: np.ndarray,
    target_quat_xyzw: np.ndarray,
    max_translation_step: float,
    max_rotation_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    current_position = np.asarray(current_position, dtype=np.float64)
    current_quat_xyzw = np.asarray(current_quat_xyzw, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64)

    delta_position = target_position - current_position
    distance = float(np.linalg.norm(delta_position))
    if distance <= max_translation_step or distance == 0.0:
        next_position = target_position.copy()
    else:
        next_position = current_position + (delta_position / distance) * float(max_translation_step)

    current_quat = current_quat_xyzw / np.linalg.norm(current_quat_xyzw)
    target_quat = target_quat_xyzw / np.linalg.norm(target_quat_xyzw)
    dot = float(np.dot(current_quat, target_quat))
    if dot < 0.0:
        target_quat = -target_quat
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)

    if angle <= max_rotation_step or angle < 1.0e-9:
        next_quat = target_quat
    else:
        ratio = float(max_rotation_step) / angle
        sin_total = np.sin(angle / 2.0)
        if abs(sin_total) < 1.0e-9:
            next_quat = target_quat
        else:
            next_quat = (
                np.sin((1.0 - ratio) * angle / 2.0) / sin_total * current_quat
                + np.sin(ratio * angle / 2.0) / sin_total * target_quat
            )
            next_quat = next_quat / np.linalg.norm(next_quat)

    return next_position, next_quat
```

Delete `clamp_joint_step(...)` if nothing else uses it, and rename the old `apply_tcp_delta(...)` call sites in later tasks.

- [ ] **Step 4: Run the focused tests to verify the new helper API passes**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_runtime_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the helper refactor**

```bash
git add src/franka_policy_runtime/franka_policy_runtime/reference.py \
        src/franka_policy_runtime/test/test_runtime_config.py
git commit -m "refactor: make cartesian pose helpers explicit"
```

### Task 2: Add the Cartesian backend with target/commanded pose state

**Files:**
- Create: `src/franka_policy_runtime/franka_policy_runtime/cartesian_backend.py`
- Create: `src/franka_policy_runtime/test/test_cartesian_backend.py`
- Modify: `src/franka_policy_runtime/setup.py` (only if package discovery needs no-op confirmation; otherwise leave untouched)

- [ ] **Step 1: Write failing unit tests for pose accumulation, stepping, and resync**

Create `src/franka_policy_runtime/test/test_cartesian_backend.py` with:

```python
import numpy as np
import pytest

from franka_policy_runtime.cartesian_backend import CartesianPoseBackend, PoseState
from franka_policy_runtime.reference import _quat_xyzw_from_axis_angle


def _same_orientation(actual: np.ndarray, expected: np.ndarray) -> bool:
    return abs(float(np.dot(actual, expected))) == pytest.approx(1.0)


def test_backend_initializes_from_measured_pose():
    backend = CartesianPoseBackend(
        action_scale=0.5,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 8,
    )
    measured = PoseState(
        position=np.array([0.4, 0.1, 0.2], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )

    backend.reset(measured)

    assert backend.target_pose.position.tolist() == pytest.approx([0.4, 0.1, 0.2])
    assert backend.commanded_pose.position.tolist() == pytest.approx([0.4, 0.1, 0.2])


def test_backend_accumulates_target_pose_from_previous_target_not_measured_pose():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 8,
    )
    measured = PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    )
    backend.reset(measured)

    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    backend.ingest_action(action)
    backend.ingest_action(action)

    assert backend.target_pose.position.tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_backend_step_returns_commanded_pose_limited_toward_target():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 6,
    )
    backend.reset(PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))
    backend.ingest_action(np.array([0.2, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0], dtype=float))

    next_pose = backend.step_commanded_pose()

    assert next_pose.position.tolist() == pytest.approx([0.05, 0.0, 0.0])
    expected_quat = _quat_xyzw_from_axis_angle(np.array([0.0, 0.0, np.pi / 6], dtype=float))
    assert _same_orientation(next_pose.quat_xyzw, expected_quat)


def test_backend_resyncs_when_measured_pose_drift_exceeds_threshold():
    backend = CartesianPoseBackend(
        action_scale=1.0,
        rotation_format="axis_angle",
        max_translation_step_per_tick=0.05,
        max_rotation_step_per_tick=np.pi / 6,
        pose_sync_reset_threshold=0.2,
    )
    backend.reset(PoseState(
        position=np.array([0.0, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))
    backend.ingest_action(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    backend.maybe_resync(PoseState(
        position=np.array([0.5, 0.0, 0.0], dtype=float),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    ))

    assert backend.target_pose.position.tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert backend.commanded_pose.position.tolist() == pytest.approx([0.5, 0.0, 0.0])
```

- [ ] **Step 2: Run the new backend tests to confirm they fail with missing module/class errors**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_cartesian_backend.py -q
```

Expected: FAIL because `franka_policy_runtime.cartesian_backend` does not exist yet.

- [ ] **Step 3: Implement the backend and pose-state dataclass**

Create `src/franka_policy_runtime/franka_policy_runtime/cartesian_backend.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_policy_runtime.reference import apply_tcp_delta_in_base_frame, step_toward_pose


@dataclass
class PoseState:
    position: np.ndarray
    quat_xyzw: np.ndarray


class CartesianPoseBackend:
    def __init__(
        self,
        *,
        action_scale: float,
        rotation_format: str,
        max_translation_step_per_tick: float,
        max_rotation_step_per_tick: float,
        pose_sync_reset_threshold: float = float("inf"),
    ) -> None:
        self._action_scale = float(action_scale)
        self._rotation_format = str(rotation_format)
        self._max_translation_step = float(max_translation_step_per_tick)
        self._max_rotation_step = float(max_rotation_step_per_tick)
        self._pose_sync_reset_threshold = float(pose_sync_reset_threshold)
        self.target_pose: PoseState | None = None
        self.commanded_pose: PoseState | None = None

    def reset(self, measured_pose: PoseState) -> None:
        pose = PoseState(
            position=np.asarray(measured_pose.position, dtype=float).copy(),
            quat_xyzw=np.asarray(measured_pose.quat_xyzw, dtype=float).copy(),
        )
        self.target_pose = PoseState(position=pose.position.copy(), quat_xyzw=pose.quat_xyzw.copy())
        self.commanded_pose = PoseState(position=pose.position.copy(), quat_xyzw=pose.quat_xyzw.copy())

    def ingest_action(self, action: np.ndarray) -> PoseState:
        if self.target_pose is None:
            raise RuntimeError("backend must be reset before ingesting actions")
        next_position, next_quat = apply_tcp_delta_in_base_frame(
            self.target_pose.position,
            self.target_pose.quat_xyzw,
            action,
            action_scale=self._action_scale,
            rotation_format=self._rotation_format,
        )
        self.target_pose = PoseState(position=next_position, quat_xyzw=next_quat)
        return self.target_pose

    def step_commanded_pose(self) -> PoseState:
        if self.target_pose is None or self.commanded_pose is None:
            raise RuntimeError("backend must be reset before stepping")
        next_position, next_quat = step_toward_pose(
            current_position=self.commanded_pose.position,
            current_quat_xyzw=self.commanded_pose.quat_xyzw,
            target_position=self.target_pose.position,
            target_quat_xyzw=self.target_pose.quat_xyzw,
            max_translation_step=self._max_translation_step,
            max_rotation_step=self._max_rotation_step,
        )
        self.commanded_pose = PoseState(position=next_position, quat_xyzw=next_quat)
        return self.commanded_pose

    def maybe_resync(self, measured_pose: PoseState) -> bool:
        if self.commanded_pose is None:
            self.reset(measured_pose)
            return True
        measured_position = np.asarray(measured_pose.position, dtype=float)
        drift = float(np.linalg.norm(measured_position - self.commanded_pose.position))
        if drift > self._pose_sync_reset_threshold:
            self.reset(measured_pose)
            return True
        return False
```

- [ ] **Step 4: Run the backend unit tests to verify they pass**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_cartesian_backend.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the backend module**

```bash
git add src/franka_policy_runtime/franka_policy_runtime/cartesian_backend.py \
        src/franka_policy_runtime/test/test_cartesian_backend.py
git commit -m "feat: add cartesian pose backend state machine"
```

### Task 3: Replace IK + JointTrajectory in the runtime with Cartesian pose command publication

**Files:**
- Modify: `src/franka_policy_runtime/franka_policy_runtime/base_node.py`
- Modify: `src/franka_policy_runtime/config/franka_policy_runtime.yaml`
- Modify: `src/franka_policy_runtime/test/test_policy_runtime_executor.py`
- Modify: `src/franka_policy_runtime/package.xml`

- [ ] **Step 1: Write failing source-level tests for the new runtime wiring**

Update `src/franka_policy_runtime/test/test_policy_runtime_executor.py` to replace IK-specific expectations with these assertions:

```python
from pathlib import Path

_BASE = Path(__file__).parents[1] / "franka_policy_runtime" / "base_node.py"
_VLA = Path(__file__).parents[1] / "franka_policy_runtime" / "vla_node.py"
_BC = Path(__file__).parents[1] / "franka_policy_runtime" / "bc_cube_stack_node.py"


def _read(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def test_policy_runtime_uses_multithreaded_executor():
    source = _BASE.read_text(encoding="utf-8")
    assert "MultiThreadedExecutor" in source
    assert "executor.spin()" in source


def test_policy_runtime_delegates_arm_control_to_cartesian_backend():
    source = _BASE.read_text(encoding="utf-8")
    assert "CartesianPoseBackend" in source
    assert "self._cartesian_backend.ingest_action(action)" in source
    assert "self._cartesian_backend.step_commanded_pose()" in source
    assert "GetPositionIK" not in source
    assert "make_joint_trajectory" not in source


def test_policy_runtime_publishes_pose_stamped_commands():
    source = _BASE.read_text(encoding="utf-8")
    assert "PoseStamped" in source
    assert 'self.declare_parameter("cartesian_command_topic"' in source
    assert "self._cartesian_command_pub" in source


def test_policy_runtime_still_delegates_observation_to_observer():
    source = _read(_BASE, _VLA, _BC)
    assert "from franka_policy_runtime.observers.base import BaseObserver" in source
    assert "self._observer" in source
    assert "OpenVLAObserver" in source
    assert "IsaacLabStackBCObserver" in source


def test_policy_runtime_refreshes_tcp_pose_before_observing_for_inference():
    source = _BASE.read_text(encoding="utf-8")
    inference_loop = source[source.index("    def _inference_loop"):]
    refresh_index = inference_loop.index("self._update_observer_tcp_pose()")
    observe_index = inference_loop.index("observation = self._observer.observe()")
    assert refresh_index < observe_index
```

- [ ] **Step 2: Run the runtime source tests and verify they fail on old IK/JointTrajectory assumptions**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_policy_runtime_executor.py -q
```

Expected: FAIL because `GetPositionIK` and `make_joint_trajectory` are still present and the Cartesian backend is absent.

- [ ] **Step 3: Rewrite `base_node.py` around the Cartesian backend and pose command publishing**

Make these concrete changes in `src/franka_policy_runtime/franka_policy_runtime/base_node.py`:

1. Remove these imports:

```python
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory
```

2. Add/import these instead:

```python
from geometry_msgs.msg import PoseStamped
from franka_policy_runtime.cartesian_backend import CartesianPoseBackend, PoseState
from franka_policy_runtime.reference import gripper_width_from_binary_action, split_policy_action
```

3. Replace old arm-control parameters with:

```python
self.declare_parameter("cartesian_command_topic", "/franka_cartesian_pose_controller/reference")
self.declare_parameter("policy_tcp_frame", "fr3_hand_tcp")
self.declare_parameter("franka_ee_frame", "fr3_hand_tcp")
self.declare_parameter("max_translation_step_per_tick", 0.01)
self.declare_parameter("max_rotation_step_per_tick", 0.1)
self.declare_parameter("pose_sync_reset_threshold", 0.05)
```

4. Create the backend and publisher in `__init__`:

```python
self._cartesian_backend = CartesianPoseBackend(
    action_scale=float(self.get_parameter("action_scale").value),
    rotation_format=self._rotation_format,
    max_translation_step_per_tick=float(self.get_parameter("max_translation_step_per_tick").value),
    max_rotation_step_per_tick=float(self.get_parameter("max_rotation_step_per_tick").value),
    pose_sync_reset_threshold=float(self.get_parameter("pose_sync_reset_threshold").value),
)
self._cartesian_command_pub = self.create_publisher(
    PoseStamped,
    str(self.get_parameter("cartesian_command_topic").value),
    10,
)
```

5. Replace `_control_tick()` with this structure:

```python
def _control_tick(self) -> None:
    t0 = time.perf_counter()
    tcp_pose = self._update_observer_tcp_pose()
    if tcp_pose is None:
        return

    measured_pose = PoseState(position=tcp_pose[0], quat_xyzw=tcp_pose[1])
    if self._cartesian_backend.commanded_pose is None:
        self._cartesian_backend.reset(measured_pose)
    else:
        self._cartesian_backend.maybe_resync(measured_pose)

    with self._queue_lock:
        action = self._queue.pop_next()
    if action is not None:
        self._observer.update_last_action(action)
        self._cartesian_backend.ingest_action(action)
        self._handle_gripper(action)

    commanded_pose = self._cartesian_backend.step_commanded_pose()
    msg = PoseStamped()
    msg.header.frame_id = str(self.get_parameter("command_frame").value)
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.pose.position.x = float(commanded_pose.position[0])
    msg.pose.position.y = float(commanded_pose.position[1])
    msg.pose.position.z = float(commanded_pose.position[2])
    msg.pose.orientation.x = float(commanded_pose.quat_xyzw[0])
    msg.pose.orientation.y = float(commanded_pose.quat_xyzw[1])
    msg.pose.orientation.z = float(commanded_pose.quat_xyzw[2])
    msg.pose.orientation.w = float(commanded_pose.quat_xyzw[3])
    t_pub = time.perf_counter()
    self._cartesian_command_pub.publish(msg)
    self._timings["queue_ops"].append(t_pub - t0)
    self._timings["publish"].append(time.perf_counter() - t_pub)
    self._maybe_log_timings()
```

6. Delete `_action_to_joint_reference()` and `_compute_ik()` entirely.

7. Update `_update_observer_tcp_pose()` to read `policy_tcp_frame` instead of `tcp_frame` for the control backend path:

```python
base_frame = str(self.get_parameter("command_frame").value)
policy_tcp_frame = str(self.get_parameter("policy_tcp_frame").value)
transform = self._tf_buffer.lookup_transform(base_frame, policy_tcp_frame, rclpy.time.Time())
```

8. Update timing labels to remove the IK-specific label:

```python
self._timings = {
    "encode": [], "inference": [], "queue_ops": [],
    "tf_lookup": [], "apply_delta": [], "publish": [],
}
```

and:

```python
labels = [
    ("encode", "1. JPEG+base64"),
    ("inference", "2. Server infer"),
    ("queue_ops", "3. Queue/step"),
    ("tf_lookup", "4. TF lookup"),
    ("apply_delta", "5. Target update"),
    ("publish", "6. Publish pose"),
]
```

9. Update `src/franka_policy_runtime/config/franka_policy_runtime.yaml` to remove:

```yaml
reference_topic: /franka_policy_controller/reference
tcp_frame: fr3_hand_tcp
move_group_name: fr3_arm
ik_service: /compute_ik
max_joint_delta_per_tick: 0.04
```

and add:

```yaml
cartesian_command_topic: /franka_cartesian_pose_controller/reference
policy_tcp_frame: fr3_hand_tcp
franka_ee_frame: fr3_hand_tcp
max_translation_step_per_tick: 0.01
max_rotation_step_per_tick: 0.1
pose_sync_reset_threshold: 0.05
```

10. Update `src/franka_policy_runtime/package.xml` to remove unused runtime dependencies if `moveit_msgs` and `trajectory_msgs` are no longer imported by the package.

- [ ] **Step 4: Run runtime/source tests to verify the Cartesian path is wired correctly**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest \
  src/franka_policy_runtime/test/test_policy_runtime_executor.py \
  src/franka_policy_runtime/test/test_runtime_config.py \
  src/franka_policy_runtime/test/test_cartesian_backend.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the runtime rewrite**

```bash
git add src/franka_policy_runtime/franka_policy_runtime/base_node.py \
        src/franka_policy_runtime/config/franka_policy_runtime.yaml \
        src/franka_policy_runtime/test/test_policy_runtime_executor.py \
        src/franka_policy_runtime/package.xml
git commit -m "feat: route policy runtime through cartesian backend"
```

### Task 4: Build the Cartesian pose ros2_control controller plugin

**Files:**
- Create: `src/franka_policy_controller/include/franka_policy_controller/franka_cartesian_pose_controller.hpp`
- Create: `src/franka_policy_controller/src/franka_cartesian_pose_controller.cpp`
- Modify: `src/franka_policy_controller/CMakeLists.txt`
- Modify: `src/franka_policy_controller/package.xml`
- Modify: `src/franka_policy_controller/franka_policy_controller_plugin.xml`
- Modify: `src/franka_policy_controller/config/franka_policy_controller.yaml`

- [ ] **Step 1: Write the failing controller source-level test**

Create `src/franka_policy_runtime/test/test_cartesian_controller_launch.py` with:

```python
from pathlib import Path

_CONTROLLER_HPP = Path(__file__).parents[2] / "franka_policy_controller" / "include" / "franka_policy_controller" / "franka_cartesian_pose_controller.hpp"
_CONTROLLER_CPP = Path(__file__).parents[2] / "franka_policy_controller" / "src" / "franka_cartesian_pose_controller.cpp"
_PLUGIN_XML = Path(__file__).parents[2] / "franka_policy_controller" / "franka_policy_controller_plugin.xml"


def test_cartesian_pose_controller_plugin_declares_pose_interfaces_and_subscription():
    header = _CONTROLLER_HPP.read_text(encoding="utf-8")
    source = _CONTROLLER_CPP.read_text(encoding="utf-8")
    plugin = _PLUGIN_XML.read_text(encoding="utf-8")

    assert "FrankaCartesianPoseController" in header
    assert "geometry_msgs::msg::PoseStamped" in header
    assert "cartesian_pose_command" in source
    assert "create_subscription<geometry_msgs::msg::PoseStamped>" in source
    assert "franka_policy_controller/FrankaCartesianPoseController" in plugin
```

- [ ] **Step 2: Run the new source-level test and confirm it fails before the controller exists**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_cartesian_controller_launch.py -q
```

Expected: FAIL because the new controller files and plugin export do not exist.

- [ ] **Step 3: Implement the new controller plugin and build wiring**

Create `src/franka_policy_controller/include/franka_policy_controller/franka_cartesian_pose_controller.hpp`:

```cpp
#ifndef FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_
#define FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/subscription.hpp"
#include "realtime_tools/realtime_buffer.hpp"

namespace franka_policy_controller
{

struct CartesianPoseReference
{
  std::array<double, 3> position;
  std::array<double, 4> quat_xyzw;
  rclcpp::Time stamp;
};

class FrankaCartesianPoseController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void reference_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  std::vector<std::string> command_interface_names() const;
  std::vector<std::string> state_interface_names() const;

  std::string arm_id_;
  double reference_timeout_sec_{0.5};
  realtime_tools::RealtimeBuffer<std::shared_ptr<CartesianPoseReference>> reference_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr reference_sub_;
};

}  // namespace franka_policy_controller

#endif  // FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_
```

Create `src/franka_policy_controller/src/franka_cartesian_pose_controller.cpp`:

```cpp
#include "franka_policy_controller/franka_cartesian_pose_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace franka_policy_controller
{

namespace
{
bool all_finite(const std::array<double, 3> & position, const std::array<double, 4> & quat)
{
  return std::all_of(position.begin(), position.end(), [](double v) { return std::isfinite(v); }) &&
         std::all_of(quat.begin(), quat.end(), [](double v) { return std::isfinite(v); });
}

std::array<double, 16> to_column_major_pose(
  const std::array<double, 3> & position,
  const std::array<double, 4> & quat_xyzw)
{
  const double x = quat_xyzw[0];
  const double y = quat_xyzw[1];
  const double z = quat_xyzw[2];
  const double w = quat_xyzw[3];

  return {
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0,
    2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0,
    2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0,
    position[0], position[1], position[2], 1.0};
}
}  // namespace

controller_interface::CallbackReturn FrankaCartesianPoseController::on_init()
{
  arm_id_ = auto_declare<std::string>("arm_id", "fr3");
  auto_declare<double>("reference_timeout_sec", 0.5);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration FrankaCartesianPoseController::command_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::INDIVIDUAL, command_interface_names()};
}

controller_interface::InterfaceConfiguration FrankaCartesianPoseController::state_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::INDIVIDUAL, state_interface_names()};
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_configure(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  const auto node = get_node();
  arm_id_ = node->get_parameter("arm_id").as_string();
  reference_timeout_sec_ = node->get_parameter("reference_timeout_sec").as_double();
  reference_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "~/reference", rclcpp::SystemDefaultsQoS(),
    [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) { reference_callback(std::move(msg)); });
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  auto reference = std::make_shared<CartesianPoseReference>();
  reference->position = {state_interfaces_[12].get_value(), state_interfaces_[13].get_value(), state_interfaces_[14].get_value()};
  reference->quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  reference->stamp = get_node()->now();
  reference_buffer_.writeFromNonRT(reference);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_deactivate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type FrankaCartesianPoseController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration & period)
{
  (void)period;
  auto reference_ptr = reference_buffer_.readFromRT();
  const auto reference = reference_ptr ? *reference_ptr : nullptr;
  if (!reference || (time - reference->stamp).seconds() > reference_timeout_sec_) {
    return controller_interface::return_type::OK;
  }

  const auto pose = to_column_major_pose(reference->position, reference->quat_xyzw);
  for (std::size_t i = 0; i < pose.size(); ++i) {
    command_interfaces_[i].set_value(pose[i]);
  }
  return controller_interface::return_type::OK;
}

void FrankaCartesianPoseController::reference_callback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  auto reference = std::make_shared<CartesianPoseReference>();
  reference->position = {msg->pose.position.x, msg->pose.position.y, msg->pose.position.z};
  reference->quat_xyzw = {
    msg->pose.orientation.x,
    msg->pose.orientation.y,
    msg->pose.orientation.z,
    msg->pose.orientation.w,
  };
  if (!all_finite(reference->position, reference->quat_xyzw)) {
    return;
  }
  reference->stamp = get_node()->now();
  reference_buffer_.writeFromNonRT(reference);
}

std::vector<std::string> FrankaCartesianPoseController::command_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(16);
  for (std::size_t i = 0; i < 16; ++i) {
    names.push_back(std::to_string(i) + "/cartesian_pose_command");
  }
  return names;
}

std::vector<std::string> FrankaCartesianPoseController::state_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(16);
  for (std::size_t i = 0; i < 16; ++i) {
    names.push_back(std::to_string(i) + "/cartesian_pose_state");
  }
  return names;
}

}  // namespace franka_policy_controller

PLUGINLIB_EXPORT_CLASS(
  franka_policy_controller::FrankaCartesianPoseController,
  controller_interface::ControllerInterface)
```

Update `src/franka_policy_controller/CMakeLists.txt`:

```cmake
find_package(geometry_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/franka_cartesian_pose_controller.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  controller_interface
  geometry_msgs
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
)

ament_export_dependencies(
  controller_interface
  geometry_msgs
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
)
```

Update `src/franka_policy_controller/package.xml`:

```xml
<depend>geometry_msgs</depend>
```

and remove:

```xml
<depend>trajectory_msgs</depend>
```

Update `src/franka_policy_controller/franka_policy_controller_plugin.xml`:

```xml
<library path="franka_policy_controller">
  <class
    name="franka_policy_controller/FrankaCartesianPoseController"
    type="franka_policy_controller::FrankaCartesianPoseController"
    base_class_type="controller_interface::ControllerInterface">
    <description>
      Cartesian pose controller for policy-generated Franka FR3 pose references.
    </description>
  </class>
</library>
```

Update `src/franka_policy_controller/config/franka_policy_controller.yaml` to:

```yaml
franka_cartesian_pose_controller:
  ros__parameters:
    arm_id: fr3
    reference_timeout_sec: 0.5
```

- [ ] **Step 4: Run the source-level test and build the controller package**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_cartesian_controller_launch.py -q
colcon build --packages-select franka_policy_controller
```

Expected: the pytest passes; `colcon build` succeeds.

- [ ] **Step 5: Commit the controller plugin**

```bash
git add src/franka_policy_controller/include/franka_policy_controller/franka_cartesian_pose_controller.hpp \
        src/franka_policy_controller/src/franka_cartesian_pose_controller.cpp \
        src/franka_policy_controller/CMakeLists.txt \
        src/franka_policy_controller/package.xml \
        src/franka_policy_controller/franka_policy_controller_plugin.xml \
        src/franka_policy_controller/config/franka_policy_controller.yaml \
        src/franka_policy_runtime/test/test_cartesian_controller_launch.py
git commit -m "feat: add franka cartesian pose controller plugin"
```

### Task 5: Switch robot launch/config to the new Cartesian controller

**Files:**
- Create: `src/franka_policy_controller/config/franka_bringup_cartesian_pose_controllers.yaml`
- Modify: `src/franka_policy_runtime/launch/robot_base.launch.py`
- Modify: `src/franka_policy_runtime/launch/vla_policy.launch.py`
- Modify: `src/franka_policy_runtime/launch/bc_cube_stack.launch.py`
- Modify: `src/franka_policy_runtime/test/test_policy_launch_files.py`
- Modify: `src/franka_policy_runtime/test/test_cartesian_controller_launch.py`

- [ ] **Step 1: Write failing launch/source tests for the new stack**

Extend `src/franka_policy_runtime/test/test_cartesian_controller_launch.py` with:

```python
from pathlib import Path

_RUNTIME_BASE = Path(__file__).parents[1] / "launch" / "robot_base.launch.py"
_CONTROLLERS = Path(__file__).parents[2] / "franka_policy_controller" / "config" / "franka_bringup_cartesian_pose_controllers.yaml"


def test_robot_base_launch_uses_cartesian_pose_controller_and_no_move_group():
    source = _RUNTIME_BASE.read_text(encoding="utf-8")
    assert "franka_bringup_cartesian_pose_controllers.yaml" in source
    assert "franka_cartesian_pose_controller" in source
    assert "move_group" not in source


def test_cartesian_controller_yaml_registers_expected_controller():
    source = _CONTROLLERS.read_text(encoding="utf-8")
    assert "franka_cartesian_pose_controller" in source
    assert "franka_policy_controller/FrankaCartesianPoseController" in source
```

Also extend `src/franka_policy_runtime/test/test_policy_launch_files.py` with:

```python

def test_robot_base_launch_no_longer_mentions_moveit_move_group():
    source = (_LAUNCH_DIR / "robot_base.launch.py").read_text(encoding="utf-8")
    assert "moveit_ros_move_group" not in source
    assert "franka_cartesian_pose_controller" in source
```

- [ ] **Step 2: Run the launch/source tests and verify they fail against the old launch wiring**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest \
  src/franka_policy_runtime/test/test_cartesian_controller_launch.py \
  src/franka_policy_runtime/test/test_policy_launch_files.py -q
```

Expected: FAIL because `robot_base.launch.py` still loads `franka_policy_controller` and `move_group`.

- [ ] **Step 3: Add the new controller YAML and rewire launch files**

Create `src/franka_policy_controller/config/franka_bringup_cartesian_pose_controllers.yaml`:

```yaml
/**:
  controller_manager:
    ros__parameters:
      update_rate: 1000
      thread_priority: 98

      joint_state_broadcaster:
        type: joint_state_broadcaster/JointStateBroadcaster

      franka_cartesian_pose_controller:
        type: franka_policy_controller/FrankaCartesianPoseController

      franka_robot_state_broadcaster:
        type: franka_robot_state_broadcaster/FrankaRobotStateBroadcaster

/**:
  franka_cartesian_pose_controller:
    ros__parameters:
      arm_id: fr3
      reference_timeout_sec: 2.0
```

Update `src/franka_policy_runtime/launch/robot_base.launch.py`:

1. Change the module docstring opening to:

```python
"""Launch the robot base stack (no policy, no sensors, no RViz).

This launch owns the minimal graph needed to control the FR3 arm:
robot_state_publisher, ros2_control with the Cartesian pose controller,
Franka gripper, and joint state aggregation.
"""
```

2. Delete the `fr3_ompl_config()` helper entirely.

3. Delete the whole `move_group_node = Node(...)` block.

4. Change the controller config path to:

```python
controllers_yaml = os.path.join(
    get_package_share_directory("franka_policy_controller"),
    "config",
    "franka_bringup_cartesian_pose_controllers.yaml",
)
```

5. Change the controller spawner list to:

```python
for controller in ["joint_state_broadcaster", "franka_cartesian_pose_controller"]
```

6. Remove `move_group_node` from the returned launch actions list.

Update `src/franka_policy_runtime/launch/vla_policy.launch.py` and `src/franka_policy_runtime/launch/bc_cube_stack.launch.py` comments/descriptions so they no longer mention IK or joint-reference control if they currently do.

- [ ] **Step 4: Run the launch/source tests and the Python package tests**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest \
  src/franka_policy_runtime/test/test_cartesian_controller_launch.py \
  src/franka_policy_runtime/test/test_policy_launch_files.py \
  src/franka_policy_runtime/test/test_policy_runtime_executor.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the launch/controller switch**

```bash
git add src/franka_policy_controller/config/franka_bringup_cartesian_pose_controllers.yaml \
        src/franka_policy_runtime/launch/robot_base.launch.py \
        src/franka_policy_runtime/launch/vla_policy.launch.py \
        src/franka_policy_runtime/launch/bc_cube_stack.launch.py \
        src/franka_policy_runtime/test/test_policy_launch_files.py \
        src/franka_policy_runtime/test/test_cartesian_controller_launch.py
git commit -m "feat: launch runtime with cartesian pose controller"
```

### Task 6: Remove the old joint-reference controller path and run verification

**Files:**
- Delete: `src/franka_policy_controller/include/franka_policy_controller/franka_policy_controller.hpp`
- Delete: `src/franka_policy_controller/src/franka_policy_controller.cpp`
- Delete: `src/franka_policy_controller/config/franka_bringup_policy_controllers.yaml`
- Modify: `src/franka_policy_controller/CMakeLists.txt`
- Modify: `src/franka_policy_runtime/package.xml`
- Modify: `src/franka_policy_runtime/setup.py` (only if cleanup is needed)

- [ ] **Step 1: Write the failing cleanup/regression source test**

Extend `src/franka_policy_runtime/test/test_cartesian_controller_launch.py` with:

```python
_OLD_CONTROLLER_CPP = Path(__file__).parents[2] / "franka_policy_controller" / "src" / "franka_policy_controller.cpp"


def test_legacy_joint_reference_controller_is_removed_from_mainline():
    assert not _OLD_CONTROLLER_CPP.exists()
```

- [ ] **Step 2: Run the cleanup test and confirm it fails while the old controller still exists**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_cartesian_controller_launch.py -q
```

Expected: FAIL because `src/franka_policy_controller/src/franka_policy_controller.cpp` still exists.

- [ ] **Step 3: Delete the old controller implementation and remove stale dependencies**

Delete these files:

```text
src/franka_policy_controller/include/franka_policy_controller/franka_policy_controller.hpp
src/franka_policy_controller/src/franka_policy_controller.cpp
src/franka_policy_controller/config/franka_bringup_policy_controllers.yaml
```

Confirm `src/franka_policy_controller/CMakeLists.txt` only builds:

```cmake
add_library(${PROJECT_NAME} SHARED
  src/franka_cartesian_pose_controller.cpp
)
```

Confirm `src/franka_policy_runtime/package.xml` no longer lists runtime dependencies that were only needed for IK/joint trajectories.

- [ ] **Step 4: Run the focused tests plus package builds**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest \
  src/franka_policy_runtime/test/test_runtime_config.py \
  src/franka_policy_runtime/test/test_cartesian_backend.py \
  src/franka_policy_runtime/test/test_policy_runtime_executor.py \
  src/franka_policy_runtime/test/test_policy_launch_files.py \
  src/franka_policy_runtime/test/test_cartesian_controller_launch.py -q
colcon build --packages-select franka_policy_controller franka_policy_runtime
```

Expected: all pytest targets PASS; both packages build successfully.

- [ ] **Step 5: Commit the cleanup**

```bash
git add -A src/franka_policy_controller src/franka_policy_runtime/package.xml
git commit -m "refactor: remove legacy joint reference controller path"
```

### Task 7: Final verification and documentation sanity pass

**Files:**
- Modify only if needed: `docs/superpowers/specs/2026-06-03-cartesian-pose-runtime-design.md`

- [ ] **Step 1: Run the full Python test suite for the runtime package**

Run:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/ -q
```

Expected: PASS.

- [ ] **Step 2: Run package builds from the workspace root**

Run:

```bash
colcon build --packages-select franka_policy_controller franka_policy_runtime
```

Expected: PASS.

- [ ] **Step 3: Run targeted package tests**

Run:

```bash
colcon test --packages-select franka_policy_runtime
colcon test-result --verbose
```

Expected: `franka_policy_runtime` tests pass with no regressions.

- [ ] **Step 4: Compare implementation against the approved design doc**

Check manually that the finished code matches these approved spec points in `docs/superpowers/specs/2026-06-03-cartesian-pose-runtime-design.md`:

- target/commanded pose dual-state backend exists
- IK + `JointTrajectory` are removed from the normal arm path
- runtime publishes direct Cartesian pose commands
- MoveIt is no longer part of the normal runtime launch stack
- frame mapping is explicit rather than implicit

If a mismatch is discovered and the code is correct, update the spec. If the spec is still correct, fix the code instead.

- [ ] **Step 5: Commit final verification or spec-touchup changes**

If no files changed after verification, record that no commit is needed for this task.

If files changed:

```bash
git add docs/superpowers/specs/2026-06-03-cartesian-pose-runtime-design.md \
        src/franka_policy_runtime \
        src/franka_policy_controller
git commit -m "docs: align cartesian runtime implementation with design"
```

---

## Self-review

### Spec coverage
- Direct Cartesian backend with `target_pose` and `commanded_pose`: Task 2 and Task 3.
- Explicit base-frame action semantics and interpolation utilities: Task 1.
- Runtime-side smoothing and resync behavior: Task 2 and Task 3.
- Franka Cartesian controller plugin and direct pose command publication: Task 4.
- Launch/config switch away from IK + `JointTrajectory`: Task 5.
- Removal of the legacy path and dependency cleanup: Task 6.
- Final verification against the approved design: Task 7.

### Placeholder scan
- No `TODO`, `TBD`, or “implement later” placeholders remain.
- All code-changing steps contain concrete code blocks or exact file deletions.
- All verification steps contain exact commands and expected outcomes.

### Type consistency
- Runtime uses `CartesianPoseBackend` + `PoseState` throughout.
- Pose helpers use `apply_tcp_delta_in_base_frame(...)` and `step_toward_pose(...)` consistently.
- Controller plugin class name is consistently `FrankaCartesianPoseController` in header, source, plugin XML, and launch config.
