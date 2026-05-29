# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 Humble (realtime kernel) colcon workspace for a Franka FR3 arm with RealSense D435i camera. Two in-house packages (`motion_plan`, `handeye_calibration`) plus vendor packages (`franka_ros`, `realsense-ros`) and an ignored `IsaacLab` checkout (`COLCON_IGNORE`).

Always source the workspace overlay before running nodes: `source install/setup.bash`.

## Build & Test

```bash
colcon build                                  # full workspace
colcon build --packages-select motion_plan    # single C++ package
colcon build --packages-up-to handeye_calibration  # package + deps
source install/setup.bash                     # overlay after build
colcon test --packages-select motion_plan handeye_calibration
colcon test-result --verbose                  # inspect failures
```

## Architecture

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

## Coding Conventions

- C++: `snake_case` filenames and `CamelCase` class names, `motion_plan/` include prefix, `-Wall -Wextra -Wpedantic`
- Python: `snake_case.py`, 4-space indent, explicit `main()` entry points
- Launch files: `*.launch.py` with `generate_launch_description()`
- ROS 2 Humble distro; run commands from workspace root `/home/young/ros2_ws`
- Never edit `build/`, `install/`, `log/`, or vendor sources unless the task explicitly requires it
