# Franka Policy Observers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move policy observations out of `PolicyRuntimeNode` into dedicated VLA and RL observers.

**Architecture:** `PolicyRuntimeNode` owns ROS scheduling and passes incoming sensor messages to an observer. `VLAObserver` preserves the current image-only OpenVLA observation path. `RLObserver` produces IsaacLab-style structured observations from joint states, TF TCP pose snapshots, gripper width, last action, and explicit unavailable placeholders.

**Tech Stack:** ROS 2 Humble Python messages, NumPy, pytest, existing `franka_policy_runtime` package.

---

### Task 1: Observer Unit Tests

**Files:**
- Create: `src/franka_policy_runtime/test/test_observers.py`
- Create: `src/franka_policy_runtime/franka_policy_runtime/observers.py`

- [ ] Write tests proving `VLAObserver` returns the latest image and that `RLObserver` returns joint position, joint velocity, TCP pose, gripper position, last action, and unavailable placeholders.
- [ ] Run `PYTHONPATH=/home/young/ros2_ws/src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_observers.py -q` and verify it fails because `observers.py` is missing.
- [ ] Implement minimal observer classes and snapshot dataclasses.
- [ ] Re-run the observer test and verify it passes.

### Task 2: Runtime Wiring

**Files:**
- Modify: `src/franka_policy_runtime/franka_policy_runtime/policy_runtime_node.py`
- Modify: `src/franka_policy_runtime/config/franka_policy_runtime.yaml`
- Test: `src/franka_policy_runtime/test/test_policy_runtime_executor.py`

- [ ] Add `observer_type` parameter with default `vla`.
- [ ] Instantiate `VLAObserver` or `RLObserver` in `PolicyRuntimeNode`.
- [ ] Forward image, joint state, TCP pose, last action, and gripper width into the observer.
- [ ] Keep `_request_policy()` payload image/instruction behavior unchanged by reading from `VLAObserver`.
- [ ] Add a source-level test that runtime imports and creates observers.
- [ ] Run `PYTHONPATH=/home/young/ros2_ws/src/franka_policy_runtime pytest src/franka_policy_runtime/test -q` and verify it passes.

### Task 3: Build Check

**Files:**
- Build package: `franka_policy_runtime`

- [ ] Run `colcon build --packages-select franka_policy_runtime`.
- [ ] If unrelated package identification warnings appear, report them separately from the selected package result.
