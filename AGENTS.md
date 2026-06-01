# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 Humble colcon workspace rooted at `/home/young/ros2_ws` for a Franka FR3 arm with a RealSense D435i camera. Always run workspace commands from this root and source the overlay before running ROS nodes:

```bash
source install/setup.bash
```

In-house packages live under `src/`:

- `franka_policy_runtime`: Python policy execution bridge from camera/joint observations to MoveIt IK and controller references.
- `franka_policy_controller`: C++ ros2_control effort-level PD controller plugin for Franka joint reference tracking.
- `policy_server`: FastAPI/uvicorn inference service with pluggable learned-policy backends.
- `motion_plan`: C++17 MoveIt RRT planner plugin.
- `handeye_calibration`: Python calibration, hand-eye solving, and pixel-to-robot utilities.

Vendor or external code also lives under `src/`, including `franka_ros`, `realsense-ros`, `openvla`, `anygrasp_sdk`, and `IsaacLab`. `src/openvla`, `src/anygrasp_sdk`, and `src/IsaacLab` are external non-ROS directories; `IsaacLab` is ignored by colcon via `COLCON_IGNORE`.

Tests are package-local in `src/<package>/test`. Runtime configuration belongs in `config/`, launch entry points in `launch/`, C++ headers in `include/`, and C++ implementations in `src/`.

Do not edit generated workspace directories: `build/`, `install/`, or `log/`.

## Build, Test, and Development Commands

```bash
colcon build                                           # full workspace
colcon build --packages-select franka_policy_runtime   # Python-only package
colcon build --packages-select policy_server           # Python-only package
colcon build --packages-select franka_policy_controller # C++ package
colcon build --packages-up-to franka_policy_runtime     # package + deps
source install/setup.bash                              # overlay after build
colcon test --packages-select franka_policy_runtime
colcon test-result --verbose                           # inspect failures
```

`franka_policy_runtime` and `policy_server` are pure Python packages. `franka_policy_controller` and `motion_plan` are C++ (`ament_cmake`) packages.

For fast Python-only test iteration without a full build:

```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/ -q
```

Prefer focused package builds and tests before full-workspace verification.

## Architecture

### Policy Control Pipeline

The learned policy stack is:

```text
Camera image -> PolicyRuntimeNode (franka_policy_runtime)
                 -> Observer (VLAObserver or RLObserver) assembles observation
                 -> HTTP POST /act -> policy_server (FastAPI + uvicorn)
                      -> backend.predict() -> 7D action array(s)
                 <- JSON response
               -> ActionChunk -> WeightedActionQueue
               -> pop_next() on control tick
               -> TF lookup for current TCP pose in command_frame
               -> apply_tcp_delta() with translation and rotation clamps
               -> MoveIt GetPositionIK service
               -> JointTrajectory to /franka_policy_controller/reference
               -> FrankaPolicyController effort-level PD tracking
               -> effort commands to Franka FR3
```

Important design decisions:

- The runtime no longer uses Cartesian path planning for each policy step. Each policy action becomes a single target joint configuration through MoveIt IK.
- `FrankaPolicyController` is a ros2_control effort-level PD tracking controller, not a trajectory controller. It extracts the first point from incoming `JointTrajectory` references and applies per-joint P/D gains and effort limits.
- Stale controller references older than `reference_timeout_sec` are ignored so the controller holds position.
- `PolicyRuntimeNode` supports `single_step`, `chunk_async`, and `streaming` scheduling modes.
- The 7th action dimension drives the gripper as a cumulative width delta through the `franka_gripper/move` action.

### `franka_policy_runtime`

Central Python bridge from policy inference to realtime control.

Key modules:

- `policy_runtime_node.py`: `PolicyRuntimeNode`, camera/joint subscriptions, async inference loop, control timer, TF delta application, MoveIt IK requests, trajectory reference publishing, gripper commands, and per-cycle timing.
- `observers.py`: `BaseObserver`, `VLAObserver`, and `RLObserver`; selected with the `observer_type` parameter.
- `action_queue.py`: `WeightedActionQueue` with overlap fusion and FIFO pop behavior for chunked policy outputs.
- `reference.py`: pure helpers for `split_policy_action()`, `apply_tcp_delta()`, and `make_joint_trajectory()`.
- `runtime_config.py`: `RuntimeConfig` and `FR3_JOINT_NAMES`.

Primary config: `config/franka_policy_runtime.yaml`.

Launch files:

- `launch/franka_policy_runtime.launch.py`: standalone runtime node.
- `launch/policy_stack.launch.py`: full stack with robot state publisher, ros2_control/controller manager, MoveIt move_group, Franka gripper, RealSense camera, policy server, and policy runtime. This is the primary real-hardware launch file.

### `franka_policy_controller`

C++ ros2_control `ControllerInterface` plugin for effort-level joint reference tracking.

- Reads joint names, per-joint P/D gains, effort limits, and `reference_timeout_sec` from ROS parameters.
- Subscribes to `~/reference` (`JointTrajectory`) and passes references to the realtime update loop through `realtime_tools::RealtimeBuffer`.
- Computes `effort = P * (q_des - q) + D * (-q_dot)` with per-joint clamping.
- Registers through `franka_policy_controller_plugin.xml` with `PLUGINLIB_EXPORT_CLASS`.
- Primary config: `config/franka_bringup_policy_controllers.yaml`.

### `policy_server`

FastAPI inference service started as a standalone uvicorn process so GPU memory use is isolated from the realtime control loop.

- Entry point: `policy_server.server:main`.
- Config: `config/policy_server.yaml`, with server settings, backend type, and backend-specific params.
- Backend plugin system in `policy_server/backends/`:
  - `base.py`: `BasePolicyBackend`; backends return a 7D action or an `(N, 7)` action chunk.
  - `factory.py`: dispatches by `config["type"]`.
  - `dummy.py`: fixed action backend for tests and dry runs.
  - `openvla.py`: HuggingFace OpenVLA backend with quantization, flash attention, and auto device selection.
  - `python_plugin.py`: generic `module:ClassName` backend escape hatch.
- HTTP API:
  - `GET /health`
  - `GET /metadata`
  - `POST /act` with `image_b64`, `instruction`, `unnorm_key`, and `actions_per_chunk`.

### `motion_plan`

C++17 MoveIt `planning_interface::PlannerManager` plugin loaded by `move_group`. It provides `RRTBaseline` and `RRTImproved` planner IDs. The templated `RRTCore` solver supports goal biasing, clearance-based adaptive step sizing, random shortcut smoothing, and post-processing through `TimeOptimalTrajectoryGeneration`.

Key files include `rrt_planner_manager.hpp/cpp`, `rrt_planning_context.hpp/cpp`, `rrt_core.hpp/cpp`, and `motion_plan_plugin.xml`.

### `handeye_calibration`

Python `ament_python` package with console scripts for camera calibration, ArUco hand-eye solving (`AX=XB` with OpenCV methods and optional RANSAC), interactive sample collection, pixel-to-robot click-to-grasp, hand-eye TF publishing, and point cloud filtering.

Sample directory convention:

```text
samples/{eye_in_hand|eye_to_hand}/{board_type}/
```

## Coding Style & Naming Conventions

Python uses 4-space indentation, `snake_case.py` modules, explicit `main()` entry points, and ROS nodes that cleanly create, spin, and destroy nodes. Nodes may use `rclpy.spin(node)` or `MultiThreadedExecutor` when concurrency is required.

C++ uses `snake_case` filenames, `CamelCase` classes, package-qualified includes, and existing warning flags (`-Wall -Wextra -Wpedantic`). Plugin classes should be registered with `PLUGINLIB_EXPORT_CLASS`.

Launch files should be named `*.launch.py` and expose `generate_launch_description()`.

## Testing Guidelines

Prefer focused package tests before full-workspace verification. Python packages use pytest through colcon, with tests named `test_*.py` under each package's `test/` directory. Add or update regression tests for config parsing, backend selection, runtime queue behavior, observer behavior, transform/reference generation, and safety limits.

For fast Python iteration, use direct pytest with the relevant package on `PYTHONPATH` when that covers the changed code. Use colcon tests before reporting changes that affect package integration, launch behavior, ROS interfaces, or installed entry points.

For C++ packages, add package-local tests when changing planner or controller behavior. Treat controller changes as safety-sensitive: verify effort limits, stale-reference behavior, parameter defaults, and launch wiring.

## Commit & Pull Request Guidelines

Recent history uses short, direct commit messages, sometimes in Chinese, for example `Update README with todo and notes` or `修改为controller实现...`. Keep commits scoped to one logical change.

Pull requests should include a short problem statement, implementation summary, test commands run, affected ROS packages, and any hardware assumptions. Include launch or runtime notes when changes affect Franka, RealSense, MoveIt, policy server ports, GPU usage, realtime behavior, gripper behavior, controller gains, or safety limits.

## Agent-Specific Instructions

Preserve local user changes and avoid broad refactors. Prefer existing package patterns over new abstractions. Do not edit vendor sources (`franka_ros`, `realsense-ros`) unless the task explicitly requires it.

Treat robot-motion, controller, and inference-server changes as safety-sensitive. Before reporting completion, verify relevant defaults, limits, ROS topics/services/actions, launch arguments, and failure behavior. For hardware-facing changes, clearly state whether verification was limited to static checks, tests, simulation, fake hardware, or real hardware.
