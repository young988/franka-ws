# OpenArm-Style Franka Policy Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenArm/LeRobot-style policy deployment stack for Franka FR3 with a custom realtime ros2_control controller, 4-bit OpenVLA policy server, action chunk fusion, and runtime modes for single-step, chunked, and streaming policies.

**Architecture:** ML inference runs outside the realtime loop in `policy_server`. `franka_policy_runtime` owns observations, inference scheduling, action chunk fusion, and conversion to joint reference trajectories. `franka_policy_controller` is a C++ effort controller that claims FR3 effort command interfaces and consumes joint trajectory references from a realtime-safe buffer.

**Tech Stack:** ROS 2 Humble, `ament_python`, `ament_cmake`, `controller_interface`, `pluginlib`, `realtime_tools`, FastAPI/uvicorn, HuggingFace Transformers, bitsandbytes 4-bit quantization, pytest.

---

### Task 1: Policy Server Config and Backend Factory

**Files:**
- Create: `src/policy_server/package.xml`
- Create: `src/policy_server/setup.py`
- Create: `src/policy_server/setup.cfg`
- Create: `src/policy_server/resource/policy_server`
- Create: `src/policy_server/policy_server/config.py`
- Create: `src/policy_server/policy_server/backends/base.py`
- Create: `src/policy_server/policy_server/backends/dummy.py`
- Create: `src/policy_server/policy_server/backends/factory.py`
- Test: `src/policy_server/test/test_config_and_factory.py`

- [ ] Write tests proving defaults select the 4-bit OpenVLA backend and dummy backend returns a 7D action.
- [ ] Run `pytest src/policy_server/test/test_config_and_factory.py -q` and verify failures are due to missing package.
- [ ] Implement config loading, backend base classes, dummy backend, and backend factory.
- [ ] Re-run the test and verify it passes.

### Task 2: OpenVLA 4-bit Backend and HTTP App

**Files:**
- Create: `src/policy_server/policy_server/backends/openvla.py`
- Create: `src/policy_server/policy_server/app.py`
- Create: `src/policy_server/policy_server/server.py`
- Create: `src/policy_server/config/policy_server.yaml`
- Test: `src/policy_server/test/test_app_dummy_backend.py`

- [ ] Write tests for `/health`, `/metadata`, and `/act` using dummy backend.
- [ ] Run the tests and verify they fail before the app exists.
- [ ] Implement FastAPI app, CLI entry point, OpenVLA prompt builder, and 4-bit-only default loader path.
- [ ] Re-run tests and verify they pass without importing heavy ML dependencies unless the OpenVLA backend is selected.

### Task 3: Runtime Action Chunk Fusion

**Files:**
- Create: `src/franka_policy_runtime/package.xml`
- Create: `src/franka_policy_runtime/setup.py`
- Create: `src/franka_policy_runtime/setup.cfg`
- Create: `src/franka_policy_runtime/resource/franka_policy_runtime`
- Create: `src/franka_policy_runtime/franka_policy_runtime/action_queue.py`
- Test: `src/franka_policy_runtime/test/test_action_queue.py`

- [ ] Write tests for single-step queueing, weighted overlap fusion, and streaming replacement.
- [ ] Run the tests and verify failures are due to missing implementation.
- [ ] Implement `ActionChunk`, `WeightedActionQueue`, and validation for finite `(N, 7)` actions.
- [ ] Re-run tests and verify they pass.

### Task 4: Runtime ROS Node Scaffold

**Files:**
- Create: `src/franka_policy_runtime/franka_policy_runtime/policy_runtime_node.py`
- Create: `src/franka_policy_runtime/config/franka_policy_runtime.yaml`
- Create: `src/franka_policy_runtime/launch/franka_policy_runtime.launch.py`
- Test: `src/franka_policy_runtime/test/test_runtime_config.py`

- [ ] Write tests for parameter defaults and action-to-joint-reference conversion bounds.
- [ ] Run tests and verify expected failures.
- [ ] Implement node scaffold with modes `single_step`, `chunk_async`, and `streaming`, publishing `trajectory_msgs/JointTrajectory` references.
- [ ] Re-run tests and verify they pass.

### Task 5: Franka Policy Controller

**Files:**
- Create: `src/franka_policy_controller/package.xml`
- Create: `src/franka_policy_controller/CMakeLists.txt`
- Create: `src/franka_policy_controller/franka_policy_controller_plugin.xml`
- Create: `src/franka_policy_controller/include/franka_policy_controller/franka_policy_controller.hpp`
- Create: `src/franka_policy_controller/src/franka_policy_controller.cpp`
- Create: `src/franka_policy_controller/config/franka_policy_controller.yaml`

- [ ] Implement a C++ `controller_interface::ControllerInterface` plugin named `franka_policy_controller/FrankaPolicyController`.
- [ ] Claim `effort` command interfaces and `position`/`velocity` state interfaces for configured FR3 joints.
- [ ] Subscribe to `~/reference` as `trajectory_msgs/msg/JointTrajectory`, validate joint order and finite positions, and update a realtime buffer.
- [ ] In `update()`, compute PD effort against the current reference with per-joint effort limits and hold current position when no reference exists.

### Task 6: Build and Test

**Files:**
- Modify as needed only in new packages.

- [ ] Run `pytest src/policy_server/test src/franka_policy_runtime/test -q`.
- [ ] Run `colcon build --symlink-install --packages-select policy_server franka_policy_runtime franka_policy_controller`.
- [ ] Run `colcon test --packages-select policy_server franka_policy_runtime`.
- [ ] Report any missing ROS dependencies or build failures exactly.
