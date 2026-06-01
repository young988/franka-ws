# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 Humble (realtime kernel) colcon workspace for a Franka FR3 arm with RealSense D435i camera. Four in-house ROS packages (`motion_plan`, `handeye_calibration`, `franka_policy_deploy`, `policy_server`) plus vendor packages (`franka_ros`, `realsense-ros`) and two external non-ROS directories (`openvla` — OpenVLA training/eval code; `anygrasp_sdk` — AnyGrasp grasp detection SDK). `IsaacLab` checkout is ignored (`COLCON_IGNORE`).

Always source the workspace overlay before running nodes: `source install/setup.bash`.

## Build & Test

```bash
colcon build                                          # full workspace
colcon build --packages-select motion_plan            # single C++ package
colcon build --packages-select franka_policy_deploy   # Python-only package
colcon build --packages-select policy_server          # Python-only package
colcon build --packages-up-to franka_policy_deploy    # package + deps
source install/setup.bash                             # overlay after build
colcon test --packages-select policy_server
colcon test-result --verbose                          # inspect failures
```

`franka_policy_deploy` is a pure Python package — no C++ compilation needed.

## Architecture

### Policy Pipeline (`policy_server` → `franka_policy_deploy` → MoveIt Planning)

The end-to-end learned policy stack:

```
Camera image → policy_client_node (franka_policy_deploy)
                 → HTTP POST /act → policy_server (FastAPI + uvicorn)
                      → backend.predict() → 7D action array [dx,dy,dz,drx,dry,drz,gripper]
                 ← JSON response
               → delta → target EE pose (current pose from TF + delta)
               → GetCartesianPath service → move_group (OMPL planning plugin)
                 → collision-checked IK-interpolated Cartesian path
               → JointTrajectory → fr3_arm_controller (joint_trajectory_controller)
                 → effort commands to Franka
```

Key design decisions:
- `policy_server` is kept as the ML inference backend (HTTP API remains unchanged).
- The custom `PolicyDeltaController` ros2_control plugin was **removed**. MoveIt Servo was also **replaced** — the system now uses MoveIt's `GetCartesianPath` service via move_group, which provides IK interpolation with collision checking. OMPL is configured as move_group's planning plugin for any free-space planning needs.
- Motion parsing: policy outputs position/rotation deltas per inference step; `policy_client_node` converts to a target end-effector pose (current pose + clamped delta, using TF2 + scipy `Rotation.from_rotvec` for rotation).
- Trajectory execution: the planned `JointTrajectory` is published directly to `/fr3_arm_controller/joint_trajectory` — the controller handles preemption natively, supporting the replan-at-every-step policy control loop.
- Gripper is controlled directly by `policy_client_node` via `franka_gripper/move` and `franka_gripper/grasp` action clients.

### `policy_server` — HTTP Inference Server (Python, `ament_python`)

Serves learned policy models over HTTP. Runs as a standalone uvicorn process (launched via `ExecuteProcess` from its launch file, not as a ROS node), so it can use GPU memory without interfering with the realtime control loop.

**Entry point:** `policy_server.server:main` — parses CLI args, loads config, creates backend, starts FastAPI app.
**Config:** `config/policy_server.yaml` — multi-section: `server` (host/port/log_level), `backend` (type + per-backend params). Defaults are embedded in `config.py` with YAML deep-merge.

**Backend plugin system** (`policy_server/backends/`):
- **`base.py`** — `BasePolicyBackend` abstract class and `PolicyRequest` dataclass (image as `np.ndarray`, instruction string, optional unnorm_key). All backends return a 7D numpy action `[dx, dy, dz, drx, dry, drz, gripper]`.
- **`factory.py`** — `create_backend(config)` dispatches to the right class by `config["type"]`.
- **`dummy.py`** — Returns a fixed configured action. Used for testing or dry-run deployment.
- **`openvla.py`** — Loads an OpenVLA model from a local path via HuggingFace `AutoModelForVision2Seq`. Supports 4-bit/8-bit quantization, flash attention, auto device selection. Constructs the OpenVLA-format prompt from the instruction string.
- **`python_plugin.py`** — Generic `module:ClassName` loader. The class receives `params` dict and must implement `predict(image, instruction, request) -> 7D array` or be callable. This is the escape hatch: define a `class_path` like `my_package.bc:BCPolicy` with any `params`, and the server routes inference to it — no changes to the server code needed.

**HTTP API** (FastAPI in `app.py`):
- `GET /health` → `{"ok": true, "backend_type": "..."}`
- `GET /metadata` → per-backend info dict
- `POST /act` → accepts JSON with `image` (H×W×3 uint8 array) and `instruction` (string), returns `[dx, dy, dz, drx, dry, drz, gripper]`. Also supports a `"encoded"` wrapper for json_numpy-serialized payloads.

### `franka_policy_deploy` — Policy Client & MoveIt Planning Deployment (Python, `ament_cmake`)

Python-only package that connects the policy server to MoveIt for Cartesian path planning.

**`policy_client_node`** (`franka_policy_deploy_py/policy_client_node.py`):
- Subscribes to camera images, sends them to `policy_server` via HTTP POST `/act`, converts the 7D response `[dx,dy,dz,drx,dry,drz,gripper]` to a target end-effector pose (current pose from TF2 + clamped delta via `scipy.spatial.transform.Rotation.from_rotvec`).
- Calls MoveIt's `/compute_cartesian_path` service (move_group, OMPL-configured) to plan a collision-checked, IK-validated Cartesian path from current to target pose.
- Publishes the resulting `JointTrajectory` directly to `/fr3_arm_controller/joint_trajectory` for execution. The controller natively handles preemption when new trajectories arrive.
- Gripper is controlled directly via `franka_gripper/move` and `franka_gripper/grasp` action clients.
- Dynamically reconfigurable via ROS parameters: `server_url`, `instruction`, `request_rate_hz`, image preprocessing (`center_crop`, `resize_to`), planning parameters (`move_group_name`, `planning_frame`, `ee_frame`, `cartesian_step_size`, `cartesian_avoid_collisions`), delta safety limits (`max_translation_delta`, `max_rotation_delta`).
- Uses threading locks so at most one planning+execution cycle is in-flight at a time.

**Config files:**
- `config/servo_deploy.yaml` — `policy_client_node` parameters (server URL, planning settings, delta limits, gripper settings).

**Launch file** (`launch/servo_deploy.launch.py`):
Conditionally starts the robot (`ros2_control` + `fr3_arm_controller` + state broadcaster), the RealSense camera, `move_group` (OMPL planning plugin + kinematics + planning scene), `policy_client_node`, and `policy_server`. All toggleable via launch arguments (`start_robot`, `start_camera`, `start_policy_client`, `start_policy_server`).

### `motion_plan` — MoveIt RRT Planner Plugin (C++17, `ament_cmake`)

A MoveIt `planning_interface::PlannerManager` plugin loaded by `move_group` at runtime. Registered via `motion_plan_plugin.xml` and exported with `PLUGINLIB_EXPORT_CLASS`.

**Class hierarchy:**
- **`RRTPlannerManager`** (`rrt_planner_manager.hpp/cpp`) — Plugin entry point. Provides two algorithm IDs to MoveIt: `RRTBaseline` and `RRTImproved`. Reads per-group planner configs from the MoveIt planner configuration map with a 5-level fallback chain: `group[planner_id]` → `planner_id` → `group[normalized_id]` → `normalized_id` → `group`.
- **`RRTPlanningContext`** (`rrt_planning_context.hpp/cpp`) — Per-planning-request instance. Resolves goal constraints (joint, pose+orientation via IK, or kinematic constraint set with random sampling fallback), runs the RRT solver, and post-processes the solution with `TimeOptimalTrajectoryGeneration` for time parameterization.
- **`RRTCore`** (`rrt_core.hpp/cpp`) — Templated generic RRT solver. Callbacks for state validation, goal checking, and goal sampling are passed as template arguments (type-erased via lambdas in practice), making it agnostic to the collision/constraint environment. Supports goal biasing, adaptive step sizing (clearance-based), and path smoothing via random shortcut pairs.

**Key parameters** in `config/rrt_planning.yaml`: range, goal_bias, max_iterations, goal_tolerance, collision_check_resolution, simplify_path, smoothing_iterations, adaptive_step_size, adaptive_clearance_near/far.

**Key files:**
- `motion_plan_plugin.xml` — pluginlib descriptor
- `CMakeLists.txt` — builds a single shared library `libmotion_plan.so`
- `launch/fr3_sensor_moveit.launch.py` — unified launch file that can switch between OMPL and the custom RRT planner via `planner:=rrt`, and optionally launches RealSense + handeye TF publisher

### `handeye_calibration` — Hand-Eye Calibration & Pixel-to-Robot (Python, `ament_python`)

Six executables defined in `setup.py` entry points, each with a wrapper script in `scripts/` and a main module in `handeye_calibration/`.

**Key modules:**
- **`aruco_handeye_calibrator.py`** — Core hand-eye solver node. Offline mode reads images from disk + robot poses from CSV. Computes `AX = XB` using OpenCV's `calibrateHandEye` with 5 methods (Tsai, Park, Horaud, Andreff, Daniilidis) plus a direct average fallback for eye-to-hand. Optional RANSAC outlier rejection on the hand-eye result. Outputs `handeye_results.csv`.
- **`aruco_camera_calibrator.py`** — Camera intrinsic calibration using detected boards.
- **`pixel_to_robot.py`** — Interactive click-to-grasp pipeline: user clicks on an OpenCV image window → depth image → camera-frame 3D point → TF lookup to base frame → MoveIt `GetMotionPlan` service call → trajectory execution via `FollowJointTrajectory` action → optional auto-grasp (open gripper → close with force). Uses a `MultiThreadedExecutor` for concurrent ROS spinning and OpenCV GUI.
- **`board_detection.py`** — Unified board detection supporting chessboard, single_aruco, charuco, and aruco_grid. Each board type has its own path through `detect_calibration_points()` and `estimate_board_pose()`.
- **`calibration_config.py`** — Dataclasses for `BoardConfig`, `IntrinsicsConfig`, `SamplePaths`. Handles preset resolution, intrinsics file resolution with multi-source fallback (`camera_info` → `calibrated` → `file` → `auto` → `manual`), and sample directory layout convention.
- **`grasp_logic.py`** — Auto-grasp gating: classifies grasp outcomes (failure/empty/success) and decides whether auto-grasp should trigger after a successful motion plan execution.
- **`sample_collector.py`** — Interactive sample collection: press 's' to simultaneously save a camera image and record the current robot end-effector pose.
- **`handeye_tf_publisher.py`** — Publishes a static TF from the saved calibration result CSV (typically `fr3_link8` → `camera_link`).
- **`target_cloud_filter.py`** — Filters a point cloud to isolate a target region.

**Sample directory convention:** `samples/{eye_in_hand|eye_to_hand}/{board_type}/` containing `img/` (images), `poses.csv` (robot poses), and calibration outputs.

### External (non-ROS) directories

- **`src/openvla`** — OpenVLA model training/evaluation/finetuning code (Prismatic VLA framework). Not built by colcon. The `policy_server` OpenVLA backend loads trained checkpoints from this directory tree by path.
- **`src/anygrasp_sdk`** — AnyGrasp grasp detection SDK with prebuilt `.so` files for multiple Python versions. Requires license registration. Contains `grasp_detection/` (GSNet), `grasp_tracking/`, and `pointnet2/` subdirectories.

## Coding Conventions

- C++: `snake_case` filenames and `CamelCase` class names, `-Wall -Wextra -Wpedantic`; plugins use `PLUGINLIB_EXPORT_CLASS`
- Python: `snake_case.py`, 4-space indent, explicit `main()` entry points; ROS nodes use `rclpy.spin(node)`
- Launch files: `*.launch.py` with `generate_launch_description()`
- ROS 2 Humble distro; run commands from workspace root `/home/young/ros2_ws`
- Never edit `build/`, `install/`, `log/`, or vendor sources unless the task explicitly requires it
