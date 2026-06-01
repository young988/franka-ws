# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 Humble (realtime kernel) colcon workspace for a Franka FR3 arm with RealSense D435i camera. Five in-house ROS packages (`motion_plan`, `handeye_calibration`, `franka_policy_runtime`, `franka_policy_controller`, `policy_server`) plus vendor packages (`franka_ros`, `realsense-ros`) and three external non-ROS directories (`openvla` — OpenVLA training/eval code; `anygrasp_sdk` — AnyGrasp grasp detection SDK; `IsaacLab` — ignored via `COLCON_IGNORE`).

Always source the workspace overlay before running nodes: `source install/setup.bash`.

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
colcon test --packages-select handeye_calibration
colcon test-result --verbose                          # inspect failures
```

`franka_policy_runtime`, `policy_server`, and `handeye_calibration` are pure Python packages — no C++ compilation needed. `franka_policy_controller` and `motion_plan` are C++ (`ament_cmake`). `franka_policy_controller` and `motion_plan` have no automated tests.

For fast Python-only test iteration without a full build:
```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/ -q
PYTHONPATH=src/handeye_calibration pytest src/handeye_calibration/test/ -q
```

## Architecture

### Policy Control Pipeline

The end-to-end learned policy stack:

```
Camera image → PolicyRuntimeNode (franka_policy_runtime)
                 → Observer (VLAObserver or RLObserver) assembles observation
                 → HTTP POST /act → policy_server (FastAPI + uvicorn)
                      → backend.predict() → 7D action array(s) [dx,dy,dz,drx,dpitch,dyaw,gripper]
                 ← JSON response
               → ActionChunk → WeightedActionQueue (single_step / chunk_async / streaming modes)
               → pop_next() on control tick
               → TF lookup (current TCP pose in command_frame)
               → apply_tcp_delta() — clamped translation + RPY→quaternion composition
               → MoveIt GetPositionIK service (collision-aware)
               → JointTrajectory published to /franka_policy_controller/reference
               → FrankaPolicyController (ros2_control effort-level PD plugin)
                 → effort commands to Franka FR3
```

Key design decisions:
- **No Cartesian path planning.** The old `GetCartesianPath` approach was replaced. Now each policy action is converted to a single target joint configuration via MoveIt IK, and the controller tracks it as a joint-space reference. The controller natively preempts when a new reference arrives.
- **Controller, not planner.** `FrankaPolicyController` is a ros2_control effort-level PD tracking controller (not a trajectory controller). It receives `JointTrajectory` messages on `~/reference`, extracts the first point's positions, and applies `effort = P*(q_des - q) + D*(-q_dot)` with configurable per-joint gains and effort limits. References that are older than `reference_timeout_sec` are ignored (controller holds position).
- **Three scheduling modes** in `PolicyRuntimeNode`:
  - `single_step`: wait for each action to be consumed before requesting the next.
  - `chunk_async`: request a new chunk before the queue is exhausted, fuse overlapping actions with weighted averaging.
  - `streaming`: replace the entire queue with each new policy output.
- **Gripper** is controlled directly by `PolicyRuntimeNode` via the `franka_gripper/move` action (`franka_msgs/Move`), integrating the 7th action dimension as a cumulative width delta.

### `franka_policy_runtime` — Policy Runtime Bridge (Python, `ament_python`)

The central node that bridges policy inference to the realtime controller.

**Key modules:**
- **`policy_runtime_node.py`** — `PolicyRuntimeNode`: subscribes to camera images and joint states, runs an async inference thread (JPEG-encodes images, POSTs to policy server), populates a `WeightedActionQueue`, and on a control timer pops actions, applies TF-based delta transforms, calls MoveIt IK, publishes `JointTrajectory` references, and handles gripper commands. Uses `MultiThreadedExecutor` with two threads. Includes per-cycle timing instrumentation.
- **`observers.py`** — `BaseObserver` (thread-safe sensor sink), `VLAObserver` (image-only OpenVLA path, returns `VLAObservation` with image + instruction), `RLObserver` (IsaacLab-style structured observation with joint positions/velocities, TCP pose, gripper position, last action, and explicit unavailable placeholders). Selectable via `observer_type` parameter.
- **`action_queue.py`** — `WeightedActionQueue`: fixed-dimension action buffer with weighted overlap fusion (`fuse()`) and FIFO pop (`pop_next()`). Used to implement chunk_async mode's temporal smoothing.
- **`reference.py`** — Pure functions: `split_policy_action()` (splits 7D into arm delta + gripper), `apply_tcp_delta()` (clamped translation + RPY-to-quaternion composition in the command frame), `make_joint_trajectory()`.
- **`runtime_config.py`** — `RuntimeConfig` dataclass and `FR3_JOINT_NAMES` constant.

**Config:** `config/franka_policy_runtime.yaml` — all runtime parameters (mode, observer_type, policy URL, frames, delta limits, gripper settings, IK service, control period).

**Launch files:**
- `launch/franka_policy_runtime.launch.py` — standalone runtime node only.
- `launch/policy_stack.launch.py` — full stack: robot_state_publisher, ros2_control + controller_manager (with `franka_policy_controller` + `joint_state_broadcaster`), MoveIt move_group (OMPL, for IK), Franka gripper, RealSense camera, policy_server, and policy_runtime_node. All components toggleable via launch arguments. This is the primary launch file for real hardware.

### `franka_policy_controller` — Realtime Effort Controller (C++, `ament_cmake`)

A ros2_control `ControllerInterface` plugin that tracks joint position references with PD + effort limits.

- **`FrankaPolicyController`** — Lifecycle-managed controller. On configure: reads joint names, per-joint P/D gains, effort limits, and `reference_timeout_sec` from ROS params; creates a subscription to `~/reference` (`JointTrajectory`). On update: reads current joint state from hardware interfaces, checks if the buffered reference is fresh (within timeout), computes PD effort with per-joint clamping, writes to effort command interfaces. Uses `realtime_tools::RealtimeBuffer` for lock-free reference passing between the non-RT subscription callback and the RT update loop.
- **Plugin registration:** `franka_policy_controller_plugin.xml` → `PLUGINLIB_EXPORT_CLASS`
- **Config:** `config/franka_bringup_policy_controllers.yaml` — controller manager config (1000 Hz update rate, RT priority 98) and per-joint gains/limits.
- Default gains (code defaults, overridable by yaml): `[600, 600, 600, 600, 250, 150, 50]` for P, `[30, 30, 30, 30, 10, 10, 5]` for D, `[30, 30, 30, 30, 15, 12, 10]` for effort limits. `reference_timeout_sec` defaults to 0.5 in code. Note: the YAML config (`franka_bringup_policy_controllers.yaml`) uses much lower gains for safety — `[60, 60, 60, 60, 25, 15, 5]` P / `[6, 6, 6, 6, 2, 2, 1]` D with 2.0 s timeout.

### `policy_server` — HTTP Inference Server (Python, `ament_python`)

Serves learned policy models over HTTP. Runs as a standalone uvicorn process (launched via `IncludeLaunchDescription` from the policy stack launch, not as a ROS node), so it can use GPU memory without interfering with the realtime control loop.

**Entry point:** `policy_server.server:main` — parses CLI args, loads config, creates backend, starts FastAPI app.
**Config:** `config/policy_server.yaml` — multi-section: `server` (host/port/log_level), `backend` (type + per-backend params). Defaults are embedded in `config.py` with YAML deep-merge.

**Backend plugin system** (`policy_server/backends/`):
- **`base.py`** — `BasePolicyBackend` abstract class. All backends return a 7D numpy action `[dx, dy, dz, drx, dry, drz, gripper]` or an `(N, 7)` array for action chunks.
- **`factory.py`** — `create_backend(config)` dispatches by `config["type"]`.
- **`dummy.py`** — Returns a fixed configured action. For testing/dry-run.
- **`openvla.py`** — Loads an OpenVLA model via HuggingFace `AutoModelForVision2Seq`. Supports 4-bit/8-bit quantization, flash attention, auto device selection.
- **`python_plugin.py`** — Generic `module:ClassName` loader. Escape hatch: define `class_path` like `my_package.bc:BCPolicy` with any `params`, no server changes needed.

**HTTP API** (FastAPI in `app.py`):
- `GET /health` → `{"ok": true, "backend_type": "..."}`
- `GET /metadata` → per-backend info dict
- `POST /act` → accepts JSON with `image_b64` (JPEG base64), `instruction` (string), `unnorm_key`, `actions_per_chunk`; returns actions array.

### `motion_plan` — MoveIt RRT Planner Plugin (C++17, `ament_cmake`)

A MoveIt `planning_interface::PlannerManager` plugin loaded by `move_group` at runtime. Provides `RRTBaseline` and `RRTImproved` algorithm IDs. Templated `RRTCore` solver with goal biasing, adaptive step sizing (clearance-based), and random shortcut path smoothing. Post-processes solutions with `TimeOptimalTrajectoryGeneration`.

**Key files:** `rrt_planner_manager.hpp/cpp` (plugin entry), `rrt_planning_context.hpp/cpp` (per-request instance), `rrt_core.hpp/cpp` (generic solver), `motion_plan_plugin.xml` (pluginlib descriptor).

**Launch:** `fr3_sensor_moveit.launch.py` — full MoveIt + RealSense octomap + hand-eye TF stack. Select planner via `planner:=ompl` (default) or `planner:=rrt`. When using `rrt`, the launch loads `motion_plan/RRTPlannerManager` as the planning plugin. Includes the `handeye_calibration` `publish_handeye_tf.launch.py` (toggleable via `publish_handeye_tf:=true`).

**Config:** `config/rrt_planning.yaml` — per-algorithm parameters (range, goal bias, max iterations, adaptive step sizing, path simplification).

### `handeye_calibration` — Hand-Eye Calibration & Pixel-to-Robot (Python, `ament_python`)

Six console scripts for camera calibration, ArUco-based hand-eye solving (`AX=XB` via OpenCV with 5 methods + RANSAC), interactive sample collection, pixel-to-robot click-to-grasp (depth image → TF → MoveIt planning → trajectory execution → auto-grasp), hand-eye TF publishing, and point cloud filtering.

**Console scripts** (installed to both `bin/` and `lib/handeye_calibration/`):
- `aruco_camera_calibrator` — camera intrinsic calibration from chessboard images.
- `aruco_handeye_calibrator` — solve `AX=XB` from collected robot-camera pose pairs.
- `sample_collector` — interactive tool to capture synchronized robot poses + camera images.
- `pixel_to_robot` — click on an image pixel → compute 3D grasp pose → plan and execute with MoveIt.
- `target_cloud_filter` — filter a point cloud to isolate a target object.
- `handeye_tf_publisher` — publish the solved hand-eye transform as a TF frame.

**Wrapper script pattern:** Scripts are installed to `lib/handeye_calibration/` via `data_files` (not just `entry_points`) because ROS 2 launch `Node` looks for executables in `lib/<pkg>/`. Each script in `scripts/` is a thin wrapper that imports and calls the corresponding module's `main()`.

**Launch files:**
- `collect_samples.launch.py` — camera + robot + sample collector node.
- `pixel_to_robot.launch.py` — camera + handeye TF + MoveIt + pixel_to_robot node.
- `publish_handeye_tf.launch.py` — reads calibration CSV, publishes camera_link → robot frame TF.
- `calibrate_offline.launch.py` — offline calibration from previously collected samples.

Sample directory convention: `samples/{eye_in_hand|eye_to_hand}/{board_type}/`.

**Key modules:**
- `board_detection.py` — ArUco/chessboard detection utilities.
- `calibration_config.py` — `CalibrationConfig` dataclass and path resolution.
- `grasp_logic.py` — grasp computation from pixel + depth (shared between pixel_to_robot and external consumers).

### External (non-ROS) directories

- **`src/openvla`** — OpenVLA model training/evaluation/finetuning code (Prismatic VLA framework). Not built by colcon. The `policy_server` OpenVLA backend loads trained checkpoints from here by path.
- **`src/anygrasp_sdk`** — AnyGrasp grasp detection SDK with prebuilt `.so` files. Requires license registration. Contains `grasp_detection/` (GSNet), `grasp_tracking/`, and `pointnet2/`.
- **`src/IsaacLab`** — Ignored by colcon (`COLCON_IGNORE`).

## Coding Conventions

- C++: `snake_case` filenames, `CamelCase` class names, `-Wall -Wextra -Wpedantic`; plugins use `PLUGINLIB_EXPORT_CLASS`
- Python: `snake_case.py`, 4-space indent, explicit `main()` entry points; ROS nodes use `rclpy.spin(node)` or `MultiThreadedExecutor`
- Launch files: `*.launch.py` with `generate_launch_description()`
- ROS 2 Humble distro; run commands from workspace root `/home/young/ros2_ws`
- Never edit `build/`, `install/`, `log/`, or vendor sources (`franka_ros`, `realsense-ros`) unless the task explicitly requires it
