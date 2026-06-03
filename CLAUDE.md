# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 Humble (realtime kernel) colcon workspace for a Franka FR3 arm with RealSense D435i camera. Five in-house ROS packages (`motion_plan`, `handeye_calibration`, `franka_policy_runtime`, `franka_policy_controller`, `policy_server`) plus vendor packages (`franka_ros`, `realsense-ros`) and three external non-ROS directories (`openvla` — OpenVLA training/eval code; `anygrasp_sdk` — AnyGrasp grasp detection SDK; `IsaacLab` — ignored via `COLCON_IGNORE`).

Always source the workspace overlay before running nodes: `source install/setup.bash`.

**Important:** If you use a conda environment (e.g. `isaaclab` for robomimic backends, `openvla` for OpenVLA), activate the conda env **before** sourcing `install/setup.bash`. Otherwise the conda env's Python won't see the workspace packages. The `policy_server.launch.py` uses `sys.executable` (the Python running the launch file) to spawn the server subprocess, so the correct conda env + workspace PYTHONPATH must be active at launch time.

## Build & Test

```bash
colcon build                                          # full workspace
colcon build --packages-select franka_policy_runtime   # Python-only package
colcon build --packages-select policy_server           # Python-only package
colcon build --packages-select handeye_calibration     # Python-only package
colcon build --packages-select franka_policy_controller # C++ package
colcon build --packages-select motion_plan             # C++ package
colcon build --packages-up-to franka_policy_runtime    # package + deps
source install/setup.bash                             # overlay after build
colcon test --packages-select franka_policy_runtime
colcon test --packages-select policy_server
colcon test --packages-select handeye_calibration
colcon test-result --verbose                          # inspect failures
```

`franka_policy_runtime`, `policy_server`, and `handeye_calibration` are pure Python packages — no C++ compilation needed. `franka_policy_controller` and `motion_plan` are C++ (`ament_cmake`). `franka_policy_controller` and `motion_plan` have no automated tests.

For fast Python-only test iteration without a full build:
```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/ -q
PYTHONPATH=src/policy_server pytest src/policy_server/test/ -q
PYTHONPATH=src/handeye_calibration pytest src/handeye_calibration/test/ -q
```

## Architecture

### Policy Control Pipeline

The end-to-end learned policy stack uses MoveIt IK to convert Cartesian policy deltas into joint trajectories:

```
Camera image → PolicyRuntimeBase subclass (VLAPolicyRuntime or BCCubeStackPolicyRuntime)
                 → Observer (OpenVLAObserver or IsaacLabStackBCObserver) assembles observation
                 → HTTP POST /act → policy_server (FastAPI + uvicorn)
                      → backend.predict_payload() → 7D action array [dx,dy,dz,ax,ay,az,gripper]
                 ← JSON response
               → apply_tcp_delta() in base frame (axis-angle or RPY format)
               → MoveIt GetPositionIK (/compute_ik) → joint positions
               → send_goal_async(FollowJointTrajectory) → joint_trajectory_controller
                  → PID effort control on Franka FR3 joints
```

Key design decisions:
- **Template method pattern.** `PolicyRuntimeBase` in `base_node.py` contains ALL shared logic (subscriptions, inference, IK, trajectory goal, gripper, timing). Subclasses (`VLAPolicyRuntime`, `BCCubeStackPolicyRuntime`) only override `_declare_parameters()` and `_create_observer()`, plus `_unnorm_key` and `_rotation_format` properties.
- **Policy delta → MoveIt IK → joint trajectory.** Policy actions are 7D Cartesian TCP deltas (6 DoF + gripper). `apply_tcp_delta()` composes the delta onto the current TCP pose in the base frame. MoveIt IK converts the target Cartesian pose to joint positions. The result is sent as a one-point `JointTrajectory` via `FollowJointTrajectory` action to the standard `joint_trajectory_controller`.
- **Two rotation delta formats**: `"axis_angle"` (IsaacLab convention, default) and `"rpy"` (OpenVLA convention), set via `_rotation_format` property on the runtime subclass.
- **Single-step control loop.** On each `_control_tick()`, the runtime observes, requests one action, runs IK, sends the trajectory goal, and waits for `_trajectory_result_cb` before requesting the next action. The `WeightedActionQueue` (`action_queue.py`) provides chunk and streaming modes but is not wired in the current single-step flow.
- **Gripper** is controlled directly by the runtime node via `franka_gripper/move` action, integrating the 7th action dimension as a binary open/close decision.
- **`run_node(node_cls, *, args, num_threads)`** is the shared entry-point utility in `base_node.py`. Every `main()` calls `run_node(TheirClass)`.

### `franka_policy_runtime` — Policy Runtime Bridge (Python, `ament_python`)

The central node that bridges policy inference to the robot controller.

**Node hierarchy** (template method):
- **`base_node.py`** — `PolicyRuntimeBase(Node)`: all shared logic (~510 lines). Declares common parameters, creates subscriptions, runs the single-step control loop (observe → infer → IK → trajectory goal), manages gripper. Uses `MultiThreadedExecutor` (2 threads) with `ReentrantCallbackGroup` for control and IK. Includes per-cycle timing instrumentation (encode / inference / IK). Also provides `run_node()` utility.
- **`vla_node.py`** — `VLAPolicyRuntime(PolicyRuntimeBase)`: declares `instruction` + `unnorm_key` params, creates `OpenVLAObserver`, overrides `_rotation_format` to `"rpy"`. Entry point: `vla_policy_runtime`.
- **`bc_cube_stack_node.py`** — `BCCubeStackPolicyRuntime(PolicyRuntimeBase)`: declares `object_pose_provider` + `object_target_color` + `object_camera_frame` + `object_min_pixels` params, creates `IsaacLabStackBCObserver` (with `ColorCubeObjectPoseProvider` + `ColorCubeStackObjectProvider` when `object_pose_provider == "color_cube"`). Entry point: `bc_cube_stack_runtime`.

**Observer package** (`observers/`):
- **`base.py`** — `BaseObserver` (thread-safe sensor sink), `BackendObservation` dataclass, `ObjectPoseProvider` type alias, utility functions (`image_msg_to_array`, `depth_msg_to_array`, `camera_info_to_k`, `estimate_object_pose_in_eef`).
- **`openvla.py`** — `OpenVLAObserver(BaseObserver)`: image + instruction observation for OpenVLA.
- **`bc_isaaclab.py`** — `IsaacLabStackBCObserver(BaseObserver)`: structured robot-state terms observation (joint positions/velocities, TCP pose, gripper position, last action, object poses).
- **`color_cube.py`** — `ColorCubeObjectPoseProvider` and `ColorCubeStackObjectProvider`: color-based cube detection for the BC stack task.

**Other modules:**
- **`reference.py`** — Pure functions: `split_policy_action()`, `apply_tcp_delta()` (axis-angle or RPY delta composition in base frame), `step_toward_pose()` (slerp-interpolated pose stepping with clamping), `gripper_width_from_binary_action()`, `make_joint_trajectory()`.
- **`action_queue.py`** — `WeightedActionQueue`: fixed-dimension action buffer with weighted overlap fusion (`fuse()`) and FIFO pop (`pop_next()`). Defined but not wired in the current single-step flow.
- **`runtime_config.py`** — Only `FR3_JOINT_NAMES` constant.

**Config:** `config/franka_policy_runtime.yaml` — shared runtime parameters (policy_url, topics, frames, trajectory_action, ik_service, move_group_name, control_period_sec, trajectory_duration_sec, action_scale, gripper settings, joint_names). Per-policy parameters (instruction, unnorm_key, object_*) are overridden by their respective launch files.

**Launch file hierarchy** (base → per-policy):
- `robot_base.launch.py` — Pure robot stack: robot_state_publisher + ros2_control (joint_trajectory_controller + joint_state_broadcaster + franka_robot_state_broadcaster) + MoveIt move_group + joint_state_publisher + Franka gripper. **No sensors, no inference, no RViz.** Other launches include this via `IncludeLaunchDescription` and append their own cameras + inference.
- `vla_policy.launch.py` — robot_base + eye-to-hand RealSense (color only, depth disabled) + handeye TF + policy_server (OpenVLA) + `vla_policy_runtime` node. Args: instruction, unnorm_key, policy_mode, etc.
- `bc_cube_stack.launch.py` — robot_base + eye-to-hand RealSense (color + depth) + handeye TF + policy_server (bc_isaaclab_stack) + `bc_cube_stack_runtime` node. Args: object_pose_provider, object_target_color, object_camera_frame, object_min_pixels.

### `franka_policy_controller` — Custom Effort Controller (C++, `ament_cmake`)

A custom ros2_control `ControllerInterface` plugin that receives `JointTrajectory` references and runs a PD control law directly in effort space (bypassing the standard `joint_trajectory_controller`).

- **`FrankaPolicyController`** — Lifecycle-managed controller. On configure: reads `joints`, `p_gains`, `d_gains`, `effort_limits`, `reference_timeout_sec` params; creates a subscription to `~/reference` (`JointTrajectory`). On activate: seeds the realtime buffer with current joint positions. On update: reads position + velocity from state interfaces, computes `effort = P * position_error + D * (-velocity)`, clamps to effort limits, writes to command interfaces. On reference timeout: holds current position.
- **`JointReference`** — struct: `positions` (vector of doubles), `stamp`.
- **Plugin registration:** `franka_policy_controller_plugin.xml` → `PLUGINLIB_EXPORT_CLASS`
- **Config:** `config/franka_policy_controller.yaml` — controller params with default PD gains and effort limits. `config/franka_bringup_policy_controllers.yaml` — controller manager config using `joint_trajectory_controller` (standard, not the custom one).

**Note:** The custom `FrankaPolicyController` exists as an alternative to the standard `joint_trajectory_controller` but is **not wired into any launch file** on the `main` branch. The `robot_base.launch.py` uses the standard `joint_trajectory_controller` with `franka_bringup_policy_controllers.yaml`.

### `policy_server` — HTTP Inference Server (Python, `ament_python`)

Serves learned policy models over HTTP. Runs as a standalone uvicorn subprocess (not a ROS node), so it can use GPU memory without interfering with the realtime control loop.

**Entry point:** `policy_server.server:main` — CLI (`--config`, `--backend`, `--host`, `--port`), loads config via `load_config()`, creates backend via `create_backend()`, serves FastAPI app via uvicorn.

**Backend plugin system** (`policy_server/backends/`):
- **`base.py`** — `BasePolicyBackend(ABC)`: `predict_payload(payload)` is the sole abstract method. `predict(image, instruction, unnorm_key)` is a non-abstract convenience method (default raises `NotImplementedError`). `_decode_image_from_payload()` static helper for JPEG→numpy decoding shared by image backends. `__init_subclass__` auto-registers every subclass by its `backend_type` class attribute into `_registry`.
- **`factory.py`** — `create_backend(config)`: looks up `config["type"]` in `BasePolicyBackend._registry`. Imports all backend modules (triggers registration), then does a simple dict lookup. No hardcoded if/elif chain.
- **`config.py`** — `default_config()`: collects per-backend defaults from each registered backend's `default_config()` static method. `merge_config()` / `load_config()` for YAML deep-merge.
- **`openvla.py`** — `OpenVLABackend`: loads OpenVLA via HuggingFace `AutoModelForVision2Seq`. 4-bit quantization default. Implements both `predict_payload()` and `predict()`.
- **`bc_isaaclab_stack.py`** — `BCIsaacLabStackBackend`: structured-terms backend for robomimic BC checkpoints. Validates required_terms shape, formats observation dict, runs policy. Lazy-loads robomimic at first inference.
- **`dummy.py`** — `DummyBackend`: returns a fixed configured action. For testing/dry-run.
- **`python_plugin.py`** — `PythonPluginBackend`: generic `module:ClassName` loader. Escape hatch for custom backends without server changes.

**HTTP API** (FastAPI in `app.py`):
- `GET /health` → `{"ok": true, "backend_type": "..."}`
- `GET /metadata` → per-backend info dict
- `POST /act` → accepts JSON with `image_b64` (JPEG base64), `instruction` (string), `unnorm_key`, `terms` (dict of named arrays), `images_b64` (multi-camera); delegates to `backend.predict_payload()`; returns `{"action": [...]}`.

### `motion_plan` — MoveIt RRT Planner Plugin (C++17, `ament_cmake`)

A MoveIt `planning_interface::PlannerManager` plugin loaded by `move_group` at runtime. Provides `RRTBaseline` and `RRTImproved` algorithm IDs. Templated `RRTCore` solver with goal biasing, adaptive step sizing (clearance-based), and random shortcut path smoothing. Post-processes solutions with `TimeOptimalTrajectoryGeneration`.

**Key files:** `rrt_planner_manager.hpp/cpp` (plugin entry), `rrt_planning_context.hpp/cpp` (per-request instance), `rrt_core.hpp/cpp` (generic solver), `motion_plan_plugin.xml` (pluginlib descriptor).

**Launch:** `fr3_sensor_moveit.launch.py` — full MoveIt + RealSense octomap + hand-eye TF stack. Select planner via `planner:=ompl` (default) or `planner:=rrt`.

**Config:** `config/rrt_planning.yaml` — per-algorithm parameters.

### `handeye_calibration` — Hand-Eye Calibration & Pixel-to-Robot (Python, `ament_python`)

Six console scripts for camera calibration, ArUco-based hand-eye solving (`AX=XB` via OpenCV with 5 methods + RANSAC), interactive sample collection, pixel-to-robot click-to-grasp, hand-eye TF publishing, and point cloud filtering.

Scripts are installed to `lib/handeye_calibration/` via `data_files` (ROS 2 launch `Node` looks for executables there). Sample convention: `samples/{eye_in_hand|eye_to_hand}/{board_type}/`.

**Key modules:** `board_detection.py` (ArUco/chessboard), `calibration_config.py` (`CalibrationConfig`), `grasp_logic.py` (pixel+depth → grasp pose).

### External (non-ROS) directories

- **`src/openvla`** — OpenVLA model training/evaluation/finetuning code (Prismatic VLA framework). Not built by colcon.
- **`src/anygrasp_sdk`** — AnyGrasp grasp detection SDK with prebuilt `.so` files. Requires license registration.
- **`src/IsaacLab`** — Ignored by colcon (`COLCON_IGNORE`).

## Coding Conventions

- C++: `snake_case` filenames, `CamelCase` class names, `-Wall -Wextra -Wpedantic`; plugins use `PLUGINLIB_EXPORT_CLASS`
- Python: `snake_case.py`, 4-space indent, explicit `main()` entry points; ROS nodes use `MultiThreadedExecutor` with `run_node()` from `base_node.py`
- Launch files: `*.launch.py` with `generate_launch_description()`; use `LaunchDescription(description=...)` only when ROS distro ≥ Iron (NOT in Humble); `DeclareLaunchArgument(description=...)` IS supported in Humble
- New backends: create a file in `policy_server/backends/`, subclass `BasePolicyBackend` with a unique `backend_type` class attribute, implement `predict_payload()`, add `default_config()` static method. Import in `factory.py`. No changes to `config.py` or `factory.py` logic needed
- New policy runtime: subclass `PolicyRuntimeBase`, override `_declare_parameters()` + `_create_observer()`, add entry point in `setup.py`, create launch file that includes `robot_base.launch.py`
- ROS 2 Humble distro; run commands from workspace root `/home/young/ros2_ws`
- Never edit `build/`, `install/`, `log/`, or vendor sources (`franka_ros`, `realsense-ros`) unless explicitly asked
