# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 Humble (realtime kernel) colcon workspace for a Franka FR3 arm with RealSense D435i camera. Seven in-house ROS packages (`motion_plan`, `handeye_calibration`, `franka_policy_runtime`, `policy_server`, `franka_telep`, `franka_policy_controller`, `fr3_visual_servo`) plus vendor packages (`franka_ros`, `realsense-ros`), one ROS 1 package (`serl_franka_controllers` — cannot be built with colcon), and external non-ROS directories (`third_party/`, `openvla`, `anygrasp_sdk`, `LeRobot-Anything-U-Arm`, `IsaacLab`, `RLinf` — all ignored via `COLCON_IGNORE` or not in `package.xml`).

Always source the workspace overlay before running nodes: `source install/setup.bash`.

**Important:** If you use a conda environment (e.g. `isaaclab` for robomimic backends, `openvla` for OpenVLA/AnyGrasp), activate the conda env **before** sourcing `install/setup.bash`. Otherwise the conda env's Python won't see the workspace packages. The `policy_server.launch.py` uses `sys.executable` (the Python running the launch file) to spawn the server subprocess, so the correct conda env + workspace PYTHONPATH must be active at launch time.

## Build & Test

Always use `--symlink-install` so Python changes take effect without rebuilding. The `install/` directory is tracked in git — branch switches don't require rebuilding unless C++ packages or vendor packages changed.

```bash
colcon build --symlink-install                                              # full workspace
colcon build --symlink-install --packages-select franka_policy_runtime      # Python-only package
colcon build --packages-select policy_server                                # Python-only package
colcon build --packages-select handeye_calibration                          # Python-only package
colcon build --packages-select franka_telep                                 # mixed C++/Python package
colcon build --packages-select franka_policy_controller                     # C++ package
colcon build --packages-select motion_plan                                  # C++ package
colcon build --packages-select fr3_visual_servo                             # Python package (YOLO deps)
colcon build --packages-up-to franka_policy_runtime                         # package + deps
source install/setup.bash                                                   # overlay after build
colcon test --packages-select franka_policy_runtime
colcon test --packages-select policy_server
colcon test --packages-select handeye_calibration
colcon test --packages-select franka_telep
colcon test-result --verbose                                                # inspect failures
```

`franka_policy_runtime`, `policy_server`, `handeye_calibration`, `fr3_visual_servo`, and `franka_telep` (Python portion) are pure Python packages — no C++ compilation needed. `franka_telep` also builds a C++ controller library. `franka_policy_controller` and `motion_plan` are C++ (`ament_cmake`). `motion_plan`, `franka_policy_controller`, and `fr3_visual_servo` have no automated tests. `serl_franka_controllers` is a ROS 1 (catkin) package — it cannot be built in this workspace.

For fast Python-only test iteration without a full build (single file):
```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_action_test.py -q
PYTHONPATH=src/policy_server pytest src/policy_server/test/test_app_dummy_backend.py -q
PYTHONPATH=src/handeye_calibration pytest src/handeye_calibration/test/test_handeye_pipeline_math.py -q
```

Or all tests in a package:
```bash
PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/ -q
PYTHONPATH=src/policy_server pytest src/policy_server/test/ -q
PYTHONPATH=src/handeye_calibration pytest src/handeye_calibration/test/ -q
PYTHONPATH=src/franka_telep pytest src/franka_telep/test/ -q
```

## Architecture

### Policy Control Pipeline

The policy runtime supports three control modes set via the `control_mode` parameter (default `"cartesian_delta"`):

**Mode 1 — `cartesian_delta`** (default): No IK needed. Policy action → Cartesian delta Twist published to `/policy/cartesian_delta` → `TwistIKController` (C++ ros2_control plugin) performs KDL IK natively in the realtime loop with PD impedance → effort commands to joints. Lowest latency path.

```
Camera image → PolicyRuntimeBase subclass
                 → Observer assembles observation
                 → HTTP POST /act → policy_server (FastAPI + uvicorn)
                      → backend.predict_payload() → 7D action array [dx,dy,dz,ax,ay,az,gripper]
                 ← JSON response
               → split_policy_action() + policy_action_to_cartesian_delta()
               → publish Twist to /policy/cartesian_delta
                 → TwistIKController (realtime KDL IK + PD impedance)
```

**Mode 2 — `joint_position`**: Policy action interpreted as absolute joint positions (7-DoF). Published as JointState to `/policy/joint_target` → `TwistIKController` (joint_position mode) applies impedance control.

**Mode 3 — `trajectory`** (legacy): MoveIt IK path. Still available but requires MoveIt move_group to be running.

```
Camera image → PolicyRuntimeBase subclass
                 → Observer → HTTP /act → policy_server → 7D action array
               → apply_tcp_delta() in base frame (axis-angle or RPY format)
               → MoveIt GetPositionIK (/compute_ik) → joint positions
               → send_goal_async(FollowJointTrajectory) → joint_trajectory_controller
```

Key design decisions:
- **Template method pattern.** `PolicyRuntimeBase` in `runtimes/base_node.py` contains ALL shared logic (~760 lines): subscriptions, inference, all three control mode dispatch, IK (trajectory mode), gripper, timing. Subclasses only override `_declare_parameters()` and `_create_observer()`, plus `_unnorm_key` and `_rotation_format` properties.
- **Policy actions are 7D.** [dx, dy, dz, ax, ay, az, gripper]. The `cartesian_delta` and `trajectory` modes treat the first 6 as Cartesian TCP deltas; `joint_position` mode treats the first 7 as absolute joint positions.
- **Two rotation delta formats**: `"axis_angle"` (IsaacLab convention, default) and `"rpy"` (OpenVLA convention), set via `_rotation_format` property on the runtime subclass.
- **Single-step control loop.** On each `_control_tick()`, the runtime observes, requests one action, dispatches to the selected control mode, then schedules the next tick. In `trajectory` mode it waits for `_trajectory_result_cb`; in `cartesian_delta`/`joint_position` mode it fires on a configurable interval (`command_interval_sec`, default 0.5s).
- **Gripper** is handled by the runtime node via `franka_gripper/move` action, integrating the 7th action dimension as a binary open/close decision (for policy runtimes) or width command (for AnyGrasp).
- **TCP pose source** is configurable via `tcp_pose_source`: `"tf"` (TF tree), `"current_pose"` (Franka state broadcaster topic), or `"franka_current_pose"` / `"franka_state_broadcaster"` (Franka state broadcaster with TF fallback).
- **`run_node(node_cls, *, args, num_threads)`** is the shared entry-point utility in `runtimes/base_node.py`. Every `main()` calls `run_node(TheirClass)`.

### `franka_policy_runtime` — Policy Runtime Bridge (Python, `ament_python`)

The central node that bridges policy inference to the robot controller.

**Node hierarchy** (template method):
- **`runtimes/base_node.py`** — `PolicyRuntimeBase(Node)`: all shared logic (~760 lines). Declares common parameters, creates subscriptions, runs the single-step control loop (observe → infer → dispatch to control mode). Supports three `control_mode` values: `cartesian_delta` (publish Twist), `joint_position` (publish JointState), `trajectory` (MoveIt IK → FollowJointTrajectory). Uses `MultiThreadedExecutor` (2 threads) with `ReentrantCallbackGroup` for control and IK. Manages gripper, TCP pose acquisition (TF or Franka state broadcaster, with fallback), per-cycle timing instrumentation. Also provides `run_node()` utility.
- **`runtimes/vla_node.py`** — `VLAPolicyRuntime(PolicyRuntimeBase)`: declares `instruction` + `unnorm_key` params, creates `OpenVLAObserver`, overrides `_rotation_format` to `"rpy"`. Entry point: `vla_policy_runtime`.
- **`runtimes/bc_cube_stack_node.py`** — `BCCubeStackPolicyRuntime(PolicyRuntimeBase)`: declares `object_pose_provider` + `object_target_color` + `object_camera_frame` + `object_min_pixels` params, creates `IsaacLabStackBCObserver` (with `ColorCubeObjectPoseProvider` + `ColorCubeStackObjectProvider` when `object_pose_provider == "color_cube"`). Entry point: `bc_cube_stack_runtime`.
- **`runtimes/anygrasp_node.py`** — `AnyGraspRuntime(PolicyRuntimeBase)`: **not a learned policy** — runs a one-shot RGB-D grasp pipeline using the AnyGrasp SDK backend. Overrides the entire control loop with a phase machine: `waiting → opening → approaching → grasping → closing → retreating → done`. Supports manual target selection via `cv2.selectROI`. Declares camera/depth params (`camera_frame`, `sensor_name`, `depth_scale`, `approach_distance`, `grasp_to_tcp_rotvec`, `manual_target_selection`, `execute_grasp`, `repeat_grasps`). Entry point: `anygrasp_runtime`.
- **`runtimes/action_test_node.py`** — `ActionTesterRuntime(PolicyRuntimeBase)`: **testing/diagnostics node** — no policy server, no sensors. Replays a hard-coded single-dimension test sequence (+dx, -dx, +dy, …, gripper open/close) through the full IK→trajectory pipeline, measures settled TCP position against target, and logs per-step error with OK/FAIL flag. Optionally writes CSV. Entry point: `action_test`.

**Observer package** (`observers/`):
- **`base.py`** — `BaseObserver` (thread-safe sensor sink), `BackendObservation` dataclass, `ObjectPoseProvider` type alias, utility functions (`image_msg_to_array`, `depth_msg_to_array`, `camera_info_to_k`, `estimate_object_pose_in_eef`).
- **`openvla.py`** — `OpenVLAObserver(BaseObserver)`: image + instruction observation for OpenVLA.
- **`bc_isaaclab.py`** — `IsaacLabStackBCObserver(BaseObserver)`: structured robot-state terms observation (joint positions/velocities, TCP pose, gripper position, last action, object poses).
- **`color_cube.py`** — `ColorCubeObjectPoseProvider` and `ColorCubeStackObjectProvider`: color-based cube detection for the BC stack task.
- **`anygrasp.py`** — `AnyGraspObserver(BaseObserver)`: RGB-D + camera intrinsics observation. Supports `set_target_bbox()` / `clear_target_bbox()` for ROI-based grasp selection.

**Utilities** (`utils/pose_math.py`):
Combined module (formerly `reference.py`, `runtime_config.py`, `tcp_pose.py`, `motion_conversion.py`). Provides:
- Quaternion operations (`quat_multiply_xyzw`, `rotate_vector_xyzw`, `compose_pose_xyzw`, `invert_pose_xyzw`)
- Policy action helpers (`validate_action`, `split_policy_action`, `apply_tcp_delta` with `axis_angle`/`rpy`, `step_toward_pose`)
- Trajectory/gripper helpers (`make_joint_trajectory`, `gripper_width_from_binary_action`)
- AnyGrasp helpers (`anygrasp_action_to_base_poses`)
- `DummyObserver` — no-sensor observer for testing
- Constants (`FR3_JOINT_NAMES`)

**Config:** `config/franka_policy_runtime.yaml` — shared runtime parameters (policy_url, topics, frames, trajectory_action, ik_service, move_group_name, control_period_sec, trajectory_duration_sec, action_scale, gripper settings, joint_names). Per-policy/per-runtime configs: `anygrasp_runtime.yaml`, `action_test.yaml`.

**Launch file hierarchy** (base → per-policy):
- `robot_base.launch.py` — Pure robot stack: robot_state_publisher + ros2_control (joint_trajectory_controller + joint_state_broadcaster + franka_robot_state_broadcaster) + MoveIt move_group + joint_state_publisher + Franka gripper. **No sensors, no inference, no RViz.** Other launches include this via `IncludeLaunchDescription` and append their own cameras + inference.
- `vla_policy.launch.py` — robot_base + eye-to-hand RealSense (color only, depth disabled) + handeye TF + policy_server (OpenVLA) + `vla_policy_runtime` node. Args: instruction, unnorm_key.
- `bc_cube_stack.launch.py` — robot_base + eye-to-hand RealSense (color + depth) + handeye TF + policy_server (bc_isaaclab_stack) + `bc_cube_stack_runtime` node. Args: object_pose_provider, object_target_color, object_camera_frame, object_min_pixels.
- `anygrasp.launch.py` — robot_base + eye-to-hand RealSense (color + aligned depth) + handeye TF + policy_server (`anygrasp` backend) + `anygrasp_runtime` node. Args: `execute_grasp` (default false = perception dry-run only), `policy_python_executable` (conda env with AnyGrasp CUDA deps).
- `action_test.launch.py` — robot_base + `action_test_runtime` node. **No sensors, no policy server, no handeye.** Args: `step_interval_sec`, `action_scale`, `tolerance_pos_m`, `csv_output_path`.

The ros2_control configuration (`franka_bringup_policy_controllers.yaml`) lives in `config/` and configures the standard `joint_trajectory_controller`, `joint_state_broadcaster`, and `franka_robot_state_broadcaster` controllers.

### `policy_server` — HTTP Inference Server (Python, `ament_python`)

Serves learned policy models over HTTP. Runs as a standalone uvicorn subprocess (not a ROS node), so it can use GPU memory without interfering with the realtime control loop.

**Entry point:** `policy_server.server:main` — CLI (`--config`, `--backend`, `--host`, `--port`), loads config via `load_config()`, creates backend via `create_backend()`, serves FastAPI app via uvicorn.

**Backend plugin system** (`policy_server/backends/`):
- **`base.py`** — `BasePolicyBackend(ABC)`: `predict_payload(payload)` is the sole abstract method. `predict(image, instruction, unnorm_key)` is a non-abstract convenience method (default raises `NotImplementedError`). `_decode_image_from_payload()` static helper for JPEG→numpy decoding shared by image backends. `__init_subclass__` auto-registers every subclass by its `backend_type` class attribute into `_registry`.
- **`factory.py`** — `create_backend(config)`: looks up `config["type"]` in `BasePolicyBackend._registry`. Imports all backend modules (triggers registration), then does a simple dict lookup. No hardcoded if/elif chain.
- **`config.py`** — `default_config()`: collects per-backend defaults from each registered backend's `default_config()` static method. `merge_config()` / `load_config()` for YAML deep-merge.
- **`openvla.py`** — `OpenVLABackend`: loads OpenVLA via HuggingFace `AutoModelForVision2Seq`. 4-bit quantization default. Implements both `predict_payload()` and `predict()`.
- **`bc_isaaclab_stack.py`** — `BCIsaacLabStackBackend`: structured-terms backend for robomimic BC checkpoints. Validates required_terms shape, formats observation dict, runs policy. Lazy-loads robomimic at first inference.
- **`anygrasp.py`** — `AnyGraspBackend`: RGB-D grasp detection using the AnyGrasp SDK (`gsnet`). Requires `sdk_root` (defaults to `src/anygrasp_sdk`) and checkpoint. Returns absolute camera-frame grasp as `[x, y, z, rx, ry, rz, width]` (axis-angle). Supports `target_bbox` for ROI filtering. Lazy-loads model on first inference.
- **`dummy.py`** — `DummyBackend`: returns a fixed configured action. For testing/dry-run.
- **`python_plugin.py`** — `PythonPluginBackend`: generic `module:ClassName` loader. Escape hatch for custom backends without server changes.

**HTTP API** (FastAPI in `app.py`):
- `GET /health` → `{"ok": true, "backend_type": "..."}`
- `GET /metadata` → per-backend info dict
- `POST /act` → accepts JSON with `image_b64` (JPEG base64), `instruction` (string), `unnorm_key`, `terms` (dict of named arrays), `images_b64` (multi-camera), `depth_npy_b64` or `depth`, `camera_matrix`, `target_bbox`; delegates to `backend.predict_payload()`; returns `{"action": [...]}`.

### `motion_plan` — MoveIt RRT Planner Plugin (C++17, `ament_cmake`)

A MoveIt `planning_interface::PlannerManager` plugin loaded by `move_group` at runtime. Provides `RRTBaseline` and `RRTImproved` algorithm IDs. Templated `RRTCore` solver with goal biasing, adaptive step sizing (clearance-based), and random shortcut path smoothing. Post-processes solutions with `TimeOptimalTrajectoryGeneration`.

**Key files:** `rrt_planner_manager.hpp/cpp` (plugin entry), `rrt_planning_context.hpp/cpp` (per-request instance), `rrt_core.hpp/cpp` (generic solver), `motion_plan_plugin.xml` (pluginlib descriptor).

**Launch:** `fr3_sensor_moveit.launch.py` — full MoveIt + RealSense octomap + hand-eye TF stack. Select planner via `planner:=ompl` (default) or `planner:=rrt`.

**Config:** `config/rrt_planning.yaml` — per-algorithm parameters.

### `handeye_calibration` — Hand-Eye Calibration & Pixel-to-Robot (Python, `ament_python`)

Six console scripts for camera calibration, ArUco-based hand-eye solving (`AX=XB` via OpenCV with 5 methods + RANSAC), interactive sample collection, pixel-to-robot click-to-grasp, hand-eye TF publishing, and point cloud filtering.

Scripts are installed to `lib/handeye_calibration/` via `data_files` (ROS 2 launch `Node` looks for executables there). Sample convention: `samples/{eye_in_hand|eye_to_hand}/{board_type}/`.

**Key modules:** `board_detection.py` (ArUco/chessboard), `calibration_config.py` (`CalibrationConfig`), `grasp_logic.py` (pixel+depth → grasp pose).

### `franka_telep` — UArm-to-FR3 Teleoperation Bridge (mixed C++/Python, `ament_cmake`)

Teleoperation system that reads Zhonglin/uArm servo angles and drives the FR3 arm via a joint-impedance follower controller or JTC trajectory commands. Also includes an OpenVLA RLDS dataset recording pipeline.

**C++ controller plugin** (`src/uarm_follower_controller.cpp`):
- `UarmFollowerController` — ros2_control `ControllerInterface` plugin loaded as `franka_telep/TeleopFollowerController`. Runs a 1 kHz realtime joint-impedance loop: subscribes to `/uarm_leader/joint_states` via a `RealtimeBuffer`, computes PD effort commands with configurable gains. Used for smooth, low-latency teleop.

**Python nodes** (7 console_scripts):
- `zhonglin_servo_reader` — Serial reader for the Zhonglin/uArm servo bus. Publishes raw servo angles at up to 120 Hz.
- `uarm_leader_publisher` — Filters raw servo angles, computes joint-space mapping (index remapping, sign/scale, limits), publishes `/uarm_leader/joint_states` at 100 Hz. Handles gripper via `franka_gripper` action. Supports home-offset latching.
- `franka_teleop` — Alternative teleop path: uses `FollowJointTrajectory` action (JTC) instead of the impedance controller. Includes critically-damped filter (`cd_omega`), velocity limiting, deadband filtering.
- `franka_home_initializer` — Sends the FR3 to a home pose via `fr3_arm_controller` JTC, then publishes `/franka_teleop/home_ready`.
- `urdf_joint_state` — Standalone URDF preview node for testing servo→joint mapping without hardware.
- `openvla_dataset_recorder` — Records synchronized (image, joint_state, TCP pose, gripper) episodes during teleop demonstrations. Outputs RLDS-compatible TFDS datasets for OpenVLA fine-tuning.
- `episode_replay` — Replays a previously recorded RLDS episode on the real robot. Sends recorded joint positions as JTC goals at the original episode rate. Supports URDF-only preview mode for dry-run testing.

**Key modules:**
- `franka_mapping.py` — Servo-to-joint mapping math: `map_servo_offsets_to_joints()`, `map_gripper_offset_to_width()`, joint limits constants (`FR3_JOINT_NAMES`, `FR3_READY_JOINTS`, `FR3_LOWER_LIMITS`, `FR3_UPPER_LIMITS`).
- `openvla_dataset.py` — RLDS dataset conversion: `OpenVLAEpisodeWriter`, image processing (`center_crop_resize_rgb`), state/action encoding for OpenVLA format.
- `zhonglin_protocol.py` — Low-level serial protocol for Zhonglin servo communication.

**Launch files:**
- `uarm_teleop_fr3.launch.py` — Full uArm→FR3 teleop stack: zhonglin_servo_reader + uarm_leader_publisher + robot_state_publisher + ros2_control (follower_controller for impedance tracking or fr3_arm_controller for JTC) + joint/franka state broadcasters + franka_home_initializer (optional) + gripper. Args: `robot_ip`, `use_home_init`, `servo_port`.
- `uarm_openvla_collect.launch.py` — Wraps `uarm_teleop_fr3.launch.py` + eye-to-hand RealSense camera + `openvla_dataset_recorder`. For collecting teleop demos. Args: `instruction`, `dataset_root`, `dataset_name`, `sample_rate_hz`, `auto_start`.
- `episode_replay.launch.py` — Replays a recorded episode on the real robot via `fr3_arm_controller` JTC. Args: `episode_path`, `robot_ip`, `sample_rate_hz`.
- `episode_replay_preview.launch.py` — URDF-only episode replay (no real robot). Args: `episode_path`.
- `fr3_urdf_preview.launch.py` — URDF-only preview with `urdf_joint_state` (no real robot, no cameras).

**Config:** `config/franka_telep.yaml` — all node parameters (servo reader, teleop, home init, urdf preview, leader publisher, dataset recorder). `config/uarm_teleop_controllers.yaml` — ros2_control config for `fr3_arm_controller` (JTC) + `follower_controller` (impedance) + broadcasters.

**Test:** `test/test_mapping.py` (mapping math), `test/test_openvla_dataset.py`, `test/test_urdf_preview.py`.

### `franka_policy_controller` — Cartesian Delta IK & Joint Position Controller (C++, `ament_cmake`)

A ros2_control `ControllerInterface` plugin (`franka_policy_controller/TwistIKController`) that receives policy commands and converts them to joint effort via impedance control. Two command modes (`command_mode` param):

- **`cartesian_delta`** — Subscribes to `geometry_msgs/Twist` on `command_topic`. Uses KDL forward kinematics + IK velocity/position solvers to compute target joint positions from a Cartesian delta. Applies joint impedance (PD) with configurable Kp/Kd gains. Clips translation/rotation steps per cycle (`max_translation_step`, `max_rotation_step`) and torque rate.
- **`joint_position`** — Subscribes to `sensor_msgs/JointState` on `joint_command_topic`. Direct joint-space target with impedance control. Clips per-cycle joint deltas (`max_joint_delta`).

Builds the KDL kinematic chain from the URDF at configure time. Uses `franka_semantic_components::FrankaRobotModel` for state interfaces.

**Config:** No dedicated YAML — parameters are set inline in launch files or ros2_control YAML.

**Note:** This is an alternative to the MoveIt IK path used by `franka_policy_runtime`. The policy runtime uses MoveIt's `/compute_ik` service; this controller does IK natively in the realtime loop. They serve different control architectures.

### `fr3_visual_servo` — YOLO-Based Visual Servoing (Python, `ament_python`)

Independent package for Position-Based Visual Servoing (PBVS) using a YOLO model and RealSense D435. Does not depend on any other in-house packages.

**Nodes** (installed as data_files to `lib/fr3_visual_servo/`, use `#!/usr/bin/env python3` shebang for conda compatibility):
- **`yolo_d435_target_node`** — Subscribes to D435 color, aligned depth, and camera info. Runs YOLO inference (`best.pt` weights) to detect objects. Publishes filtered 3D target pose to `/fr3_visual_servo/target_pose`.
- **`fr3_pbvs_servo`** — Subscribes to `/fr3_visual_servo/target_pose`, transforms into `fr3_link0`, computes conservative PBVS linear velocity, publishes `geometry_msgs/TwistStamped` to `/servo_node/delta_twist_cmds` for MoveIt Servo.
- **`yolo_test_node`** — Standalone YOLO inference test node (no robot needed). Displays annotated image with bounding boxes.

**Launch files:**
- `fr3_pbvs_bringup.launch.py` — Full PBVS stack: RealSense camera + YOLO target detection + PBVS servo node. For running visual servoing on the real robot.
- `fr3_servo_only.launch.py` — Servo node only (assumes camera and target detection running separately).
- `fr3_visual_servo.launch.py` — Camera + target detection only (no servo commands). Useful for testing/perception debugging.
- `pbvs_dry_run.launch.py` — Dry-run mode: publishes synthetic target poses for testing servo logic without camera/hardware.
- `yolo_test.launch.py` — YOLO inference test with annotated image display.

**Prerequisites:** MoveIt Servo must be running and accepting `TwistStamped` on `/servo_node/delta_twist_cmds`. Camera extrinsic TF must be published: eye-in-hand (`fr3_hand_tcp → camera_color_optical_frame`) or eye-to-hand (`fr3_link0 → camera_color_optical_frame`).

**Config:** `config/` directory with YAML parameter files. YOLO weights at `weights/best.pt`.

### `serl_franka_controllers` — SERL Cartesian Impedance & Joint Position (ROS 1, `catkin`)

**Cannot be built with colcon** — this is a ROS 1 Noetic package from [rail-berkeley/serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers). Provides Cartesian impedance and joint position controllers for Franka Emika robots, used in the SERL reinforcement learning framework. Depends on `libfranka`, `franka_ros` (ROS 1), and `franka_hw`. Included in this workspace for reference only.

### External (non-ROS) directories

- **`src/openvla`** — OpenVLA model training/evaluation/finetuning code (Prismatic VLA framework). Not built by colcon.
- **`src/anygrasp_sdk`** — AnyGrasp grasp detection SDK with prebuilt `.so` files. Requires license registration.
- **`src/LeRobot-Anything-U-Arm`** — LeRobot community arm collection. Ignored by colcon (`COLCON_IGNORE`).
- **`src/IsaacLab`** — Ignored by colcon (`COLCON_IGNORE`).
- **`src/RLinf`** — RL training/inference framework for embodied agents (VLA training). Has its own `CLAUDE.md` and `AGENTS.md`. Ignored by colcon (`COLCON_IGNORE`). Supports single-machine (Docker or `requirements/install.sh`) and multi-node Ray clusters. Includes OpenVLA evaluation pipelines. See its `CLAUDE.md` for setup and configuration.
- **`src/third_party/`** — Contains `franka_ros2_teleop`, `franka_spacemouse`, `LeRobot-Anything-U-Arm` (all with `COLCON_IGNORE`). Reference/experimental code, not built.

## Coding Conventions

- C++: `snake_case` filenames, `CamelCase` class names, `-Wall -Wextra -Wpedantic`; plugins use `PLUGINLIB_EXPORT_CLASS`; ros2_control controllers use `controller_interface::ControllerInterface` with `pluginlib` export
- Python: `snake_case.py`, 4-space indent, explicit `main()` entry points; ROS nodes use `MultiThreadedExecutor` with `run_node()` from `runtimes/base_node.py`
- Mixed C++/Python packages (`franka_telep`): use `ament_cmake` build type with `ament_cmake_python` + `ament_python_install_package()`. Python packages declared in `setup.py`; C++ libraries via `add_library()` with `pluginlib` export
- Launch files: `*.launch.py` with `generate_launch_description()`; use `LaunchDescription(description=...)` only when ROS distro ≥ Iron (NOT in Humble); `DeclareLaunchArgument(description=...)` IS supported in Humble
- New backends: create a file in `policy_server/backends/`, subclass `BasePolicyBackend` with a unique `backend_type` class attribute, implement `predict_payload()`, add `default_config()` static method. Import in `factory.py`. No changes to `config.py` or `factory.py` logic needed
- New policy runtime: subclass `PolicyRuntimeBase`, override `_declare_parameters()` + `_create_observer()`, add entry point in `setup.py`, create launch file that includes `robot_base.launch.py`
- New ros2_control controller: subclass `controller_interface::ControllerInterface`, add a `<class>` entry in the package's `.xml` plugin descriptor, register in `CMakeLists.txt` with `pluginlib_export_plugin_description_file()`
- ROS 2 Humble distro; run commands from workspace root `/home/young/ros2_ws`
- Never edit `build/`, `install/`, `log/`, or vendor sources (`franka_ros`, `realsense-ros`) unless explicitly asked
- `serl_franka_controllers` is ROS 1 (catkin) — do not attempt to build it with colcon

## Robot Operation

### Home Position

The FR3 fourth joint (J4) has been mechanically adjusted down to -2.95 rad:

```python
FR3_READY_JOINTS = [-0.0059, -1.4723, 0.0059, -2.95, -0.0028, 1.6555, 0.8048]
```

This value lives in `franka_mapping.py` (`FR3_READY_JOINTS`), `config/franka_telep.yaml` (under `franka_home_initializer`, `franka_teleop`, `urdf_preview`, `uarm_leader_publisher`), and must be consistent across all four nodes.

### Starting the Real Robot

```bash
# Cartesian delta mode (default — TwistIKController performs IK in realtime loop):
ros2 launch franka_policy_runtime robot_base.launch.py \
  robot_ip:=172.16.0.2 \
  load_gripper:=true \
  controller_mode:=cartesian_delta

# Joint position mode:
ros2 launch franka_policy_runtime robot_base.launch.py \
  robot_ip:=172.16.0.2 \
  load_gripper:=true \
  controller_mode:=joint_position
```

When `controller_mode:=cartesian_delta` or `controller_mode:=joint_position`, the `TwistIKController` ros2_control plugin is loaded. The `franka_bringup_policy_controllers.yaml` config must match.

### uArm Teleop + Data Collection

Full teleop + recording launch:

```bash
ros2 launch franka_telep uarm_openvla_collect.launch.py \
  robot_ip:=172.16.0.2 \
  instruction:="pick up the red block"
```

Episode control during recording:

```bash
# Save current episode and start a new one:
ros2 topic pub --once /franka_teleop/dataset_recording \
  std_msgs/msg/Bool "{data: false}"

# Set instruction for next episode:
ros2 topic pub --once /franka_teleop/dataset_instruction \
  std_msgs/msg/String "{data: 'place the block'}"
```

Data saved to `~/franka_openvla_data/franka_teleop/raw/episode_XXXXXX/`.

Convert to TFDS/RLDS:

```bash
export FRANKA_TELEOP_RAW_DIR=~/franka_openvla_data/franka_teleop/raw
cd install/franka_telep/share/franka_telep/openvla_rlds/franka_teleop_dataset
tfds build --overwrite --data_dir ~/tensorflow_datasets
```

Training data fields: `observation.image [256,256,3]`, `observation.state [8]`, `action [7]`, `language_instruction`.

### Known Issues

- **Gripper state**: Do not rely on the merged `/joint_states` for gripper position. Subscribe directly to `/fr3_gripper/joint_states` instead.
- **RealSense D435i on USB 2.1**: Falls back to 640×480×15 FPS. Use `--show-args` to verify actual parameters.
- **Controller mode YAML mismatch**: `controller_mode:=cartesian_delta` requires the YAML to configure `TwistIKController` with `command_mode: cartesian_delta`; similarly `joint_position` requires `command_mode: joint_position`.
- **`anygrasp_runtime` conda env**: Requires the `openvla` conda environment (for AnyGrasp CUDA deps). Pass `policy_python_executable` to the launch file to use the correct Python.
