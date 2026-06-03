# Action Dimension Test Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ActionTesterRuntime node that sends single-dimension action deltas through the full IK→trajectory pipeline and records TCP outcomes for validation.

**Architecture:** Subclass PolicyRuntimeBase, override `_create_observer` (DummyObserver), `_request_policy` (pop from test sequence), and `_control_tick` (intercept pre/target/post TCP recording). The base handles IK client, trajectory client, TF listener, and gripper setup.

**Tech Stack:** Python, ROS 2 Humble, numpy (no new dependencies)

---

### Task 1: Create the ActionTesterRuntime node

**Files:**
- Create: `src/franka_policy_runtime/franka_policy_runtime/action_test_node.py`

- [ ] **Step 1: Write the node**

```python
"""Action dimension test runtime — validates each action dim through the pipeline."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK

from franka_policy_runtime.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.observers.base import BackendObservation, BaseObserver
from franka_policy_runtime.reference import apply_tcp_delta, split_policy_action

_DIM_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def _action_dim_label(action: np.ndarray) -> str:
    """Human-readable label for which dimension(s) are non-zero in the action."""
    arr = np.asarray(action, dtype=float)
    parts: list[str] = []
    for i, name in enumerate(_DIM_LABELS):
        val = arr[i]
        if abs(val) > 1e-9:
            if name == "gripper":
                parts.append("gripper_open" if val > 0 else "gripper_close")
            else:
                sign = "+" if val > 0 else "-"
                parts.append(f"{sign}{name}")
    return ",".join(parts) if parts else "zero"


def _quat_angle_diff_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Angular distance in radians between two xyzw quaternions."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    dot = float(np.dot(a_norm, b_norm))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


class ActionTesterRuntime(PolicyRuntimeBase):
    """Test node that sends single-dimension deltas and records TCP outcomes.

    Inherits IK / trajectory / gripper plumbing from PolicyRuntimeBase.
    Overrides _create_observer, _request_policy, and _control_tick to
    inject test actions and record per-step results.
    """

    def __init__(self, node_name: str = "action_test_runtime") -> None:
        # --- call super().__init__() which calls _declare_parameters,
        # _create_observer, _create_subscriptions, and creates the oneshot
        # timer.  We override _create_observer so a DummyObserver is used.
        super().__init__(node_name=node_name)

        self._step_index = 0
        self._results: list[dict] = []
        self._pending_step_data: dict | None = None

        # param accessors
        self._test_sequence: list[list[float]] = list(
            self.get_parameter("test_sequence").value
        )
        self._step_interval_sec: float = float(
            self.get_parameter("step_interval_sec").value
        )
        self._tolerance_pos_m: float = float(
            self.get_parameter("tolerance_pos_m").value
        )
        self._csv_output_path: str = str(
            self.get_parameter("csv_output_path").value
        )
        self._action_scale_val: float = float(
            self.get_parameter("action_scale").value
        )

        self.get_logger().info(
            f"ActionTesterRuntime ready — {len(self._test_sequence)} steps configured"
        )

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "test_sequence",
            [
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ],
        )
        self.declare_parameter("step_interval_sec", 2.0)
        self.declare_parameter("tolerance_pos_m", 0.01)
        self.declare_parameter("csv_output_path", "")

    def _create_observer(self) -> BaseObserver:
        return _DummyObserver(joint_names=self._joint_names)

    @property
    def _unnorm_key(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Policy request (test sequence)
    # ------------------------------------------------------------------

    def _request_policy(self, observation) -> np.ndarray:
        if self._step_index >= len(self._test_sequence):
            return np.zeros((1, 7), dtype=float)
        action = np.asarray(self._test_sequence[self._step_index], dtype=float)
        return action.reshape(1, -1)

    # ------------------------------------------------------------------
    # Control tick (overridden for recording)
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        if self._goal_active:
            return

        if self._step_index >= len(self._test_sequence):
            self._log_summary_and_shutdown()
            return

        self._goal_active = True

        try:
            # ---- 1. get current TCP via TF ----
            tcp_result = self._update_observer_tcp_pose()
            if tcp_result is None:
                self.get_logger().warn("TF unavailable, skipping step")
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return
            pre_position, pre_quat = tcp_result

            # ---- 2. get action from test sequence ----
            observation = self._observer.observe()
            actions = self._request_policy(observation)
            if actions is None or actions.shape[0] == 0:
                self._log_summary_and_shutdown()
                return

            action = actions[0]
            self._observer.update_last_action(action)

            # ---- 3. gripper ----
            self._handle_gripper(action)

            # ---- 4. compute target via apply_tcp_delta ----
            target_position, target_quat = apply_tcp_delta(
                pre_position, pre_quat, action,
                action_scale=self._action_scale_val,
                rotation_format=self._rotation_format,
            )

            # ---- 5. IK ----
            current_joints = self._observer.latest_joint_positions()
            if current_joints is None:
                self.get_logger().warn("No joint state, skipping step")
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return

            target_joints = self._compute_ik(current_joints, target_position, target_quat)
            if target_joints is None:
                self.get_logger().warn(
                    f"IK failed for step {self._step_index} "
                    f"({_action_dim_label(action)}), skipping"
                )
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return

            # ---- 6. stash pre/target for post-step comparison ----
            self._pending_step_data = {
                "step": self._step_index,
                "dim_label": _action_dim_label(action),
                "action": action.tolist(),
                "pre_position": pre_position.tolist(),
                "pre_quat_xyzw": pre_quat.tolist(),
                "target_position": target_position.tolist(),
                "target_quat_xyzw": target_quat.tolist(),
            }

            self._send_trajectory_goal(target_joints)
            # result callback records post_tcp and continues

        except Exception:
            self._goal_active = False
            raise

    def _trajectory_result_cb(self, future):
        """Wraps base result callback to record post_tcp."""
        try:
            _ = future.result()
        except Exception:
            pass

        # ---- record post-TCP ----
        if self._pending_step_data is not None:
            tcp_result = self._update_observer_tcp_pose()
            if tcp_result is not None:
                post_position, post_quat = tcp_result
                step_data = self._pending_step_data
                pos_err = float(np.linalg.norm(
                    np.array(post_position)
                    - np.array(step_data["target_position"])
                ))
                step_data["post_position"] = post_position.tolist()
                step_data["post_quat_xyzw"] = post_quat.tolist()
                step_data["target_error_pos_m"] = pos_err
                step_data["ok"] = pos_err < self._tolerance_pos_m
                self._results.append(step_data)

                self.get_logger().info(
                    f"[{step_data['dim_label']:>14s}] "
                    f"pre=[{step_data['pre_position'][0]:.3f},"
                    f"{step_data['pre_position'][1]:.3f},"
                    f"{step_data['pre_position'][2]:.3f}] "
                    f"target=[{step_data['target_position'][0]:.3f},"
                    f"{step_data['target_position'][1]:.3f},"
                    f"{step_data['target_position'][2]:.3f}] "
                    f"post=[{post_position[0]:.3f},{post_position[1]:.3f},{post_position[2]:.3f}] "
                    f"err={pos_err:.4f}m {'OK' if step_data['ok'] else 'FAIL'}"
                )
            self._pending_step_data = None

        self._goal_active = False
        self._active_goal_handle = None
        self._step_index += 1
        if self._step_index >= len(self._test_sequence):
            self._log_summary_and_shutdown()
        else:
            self._schedule_next()

    def _schedule_next(self):
        """Schedule the next step after the configured interval."""
        self.create_timer(
            self._step_interval_sec,
            self._control_tick,
            oneshot=True,
        )

    def _log_summary_and_shutdown(self):
        """Log summary table and optionally write CSV, then shutdown."""
        if not self._results:
            self.get_logger().info("No results to report")
            self._shutdown()
            return

        self.get_logger().info("=" * 60)
        self.get_logger().info("Action Test Summary")
        self.get_logger().info("=" * 60)
        errors = [r["target_error_pos_m"] for r in self._results]
        n_ok = sum(1 for r in self._results if r["ok"])
        self.get_logger().info(
            f"Steps: {len(self._results)}  OK: {n_ok}  FAIL: {len(self._results) - n_ok}"
        )
        self.get_logger().info(
            f"Pos error — mean: {np.mean(errors):.4f}  max: {np.max(errors):.4f}  "
            f"p95: {np.percentile(errors, 95):.4f}  "
            f"tolerance: {self._tolerance_pos_m:.4f}"
        )
        self.get_logger().info("-" * 60)
        for r in self._results:
            self.get_logger().info(
                f"  [{r['dim_label']:>14s}] "
                f"err={r['target_error_pos_m']:.4f}m "
                f"{'OK' if r['ok'] else 'FAIL'}"
            )
        self.get_logger().info("=" * 60)

        if self._csv_output_path:
            self._write_csv()

        self._shutdown()

    def _write_csv(self):
        """Write results to CSV file."""
        path = Path(self._csv_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step", "dim_label", "action",
            "pre_position", "pre_quat_xyzw",
            "target_position", "target_quat_xyzw",
            "post_position", "post_quat_xyzw",
            "target_error_pos_m", "ok",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                row = {k: r[k] for k in fieldnames}
                writer.writerow(row)
        self.get_logger().info(f"Results written to {path}")

    def _shutdown(self):
        """Trigger a controlled shutdown."""
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class _DummyObserver(BaseObserver):
    """Observer that always reports ready with an empty payload."""

    def observe(self) -> BackendObservation:
        return BackendObservation(ready=True, payload={})


def main(args=None) -> None:
    run_node(ActionTesterRuntime, args=args)
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('src/franka_policy_runtime/franka_policy_runtime/action_test_node.py', doraise=True)"`
Expected: no output (success)

---

### Task 2: Add entry point in setup.py

**Files:**
- Modify: `src/franka_policy_runtime/setup.py`

- [ ] **Step 1: Add console_scripts entry**

In `setup.py`, add the entry point:

```python
entry_points={
    "console_scripts": [
        "vla_policy_runtime = franka_policy_runtime.vla_node:main",
        "bc_cube_stack_runtime = franka_policy_runtime.bc_cube_stack_node:main",
        "action_test = franka_policy_runtime.action_test_node:main",
    ],
},
```

- [ ] **Step 2: Build and verify entry point exists**

Run: `cd /home/young/ros2_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select franka_policy_runtime 2>&1 | tail -5`
Expected: `Finished <<< franka_policy_runtime`

Then verify: `ls install/franka_policy_runtime/lib/franka_policy_runtime/action_test`
Expected: file exists (symlink)

---

### Task 3: Create config file

**Files:**
- Create: `src/franka_policy_runtime/config/action_test.yaml`

- [ ] **Step 1: Write the config**

```yaml
action_test_runtime:
  ros__parameters:
    test_sequence:
      - [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      - [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      - [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
      - [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0]
      - [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]
      - [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
      - [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]
      - [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
      - [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]
      - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    step_interval_sec: 2.0
    tolerance_pos_m: 0.01
    csv_output_path: ""
    action_scale: 0.5
    trajectory_action: /joint_trajectory_controller/follow_joint_trajectory
    command_frame: fr3_link0
    tcp_frame: fr3_hand_tcp
    move_group_name: fr3_arm
    ik_service: /compute_ik
    trajectory_duration_sec: 0.5
    joint_state_topic: /joint_states
    gripper_move_action: /franka_gripper/move
    gripper_min_width: 0.0
    gripper_max_width: 0.08
    gripper_initial_width: 0.04
    gripper_speed: 0.05
    gripper_deadband: 0.002
    joint_names:
      - fr3_joint1
      - fr3_joint2
      - fr3_joint3
      - fr3_joint4
      - fr3_joint5
      - fr3_joint6
      - fr3_joint7
```

- [ ] **Step 2: Commit**

```bash
git add src/franka_policy_runtime/config/action_test.yaml
git commit -m "feat: add action_test config with default sequence"
```

---

### Task 4: Create launch file

**Files:**
- Create: `src/franka_policy_runtime/launch/action_test.launch.py`

- [ ] **Step 1: Write the launch file**

```python
"""Action dimension test launch.

Robot base + action_test_runtime node.  No sensors, no policy server,
no handeye — pure robot stack with the test runtime injecting hard-coded
actions.

Usage:
    ros2 launch franka_policy_runtime action_test.launch.py
    ros2 launch franka_policy_runtime action_test.launch.py \
        use_fake_hardware:=true load_gripper:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2",
                              description="FR3 robot IP address (use 192.168.0.100 for fake hardware)."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false",
                              description="Run mock hardware interfaces instead of a real FR3."),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Include Franka gripper in robot description and launch driver."),
        DeclareLaunchArgument("step_interval_sec", default_value="2.0",
                              description="Wait time between test steps (seconds)."),
        DeclareLaunchArgument("action_scale", default_value="0.5",
                              description="Multiplier applied to action delta before IK."),
        DeclareLaunchArgument("tolerance_pos_m", default_value="0.01",
                              description="Max acceptable position error for OK flag."),
        DeclareLaunchArgument("csv_output_path", default_value="",
                              description="If set, write results CSV to this path."),
    ]

    robot_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "launch",
                "robot_base.launch.py",
            ])
        ]),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "load_gripper": LaunchConfiguration("load_gripper"),
        }.items(),
    )

    action_test = Node(
        package="franka_policy_runtime",
        executable="action_test",
        name="action_test_runtime",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("franka_policy_runtime"),
                "config",
                "action_test.yaml",
            ]),
            {
                "step_interval_sec": LaunchConfiguration("step_interval_sec"),
                "action_scale": LaunchConfiguration("action_scale"),
                "tolerance_pos_m": LaunchConfiguration("tolerance_pos_m"),
                "csv_output_path": LaunchConfiguration("csv_output_path"),
            },
        ],
    )

    return LaunchDescription(args + [robot_base, action_test])
```

- [ ] **Step 2: Commit**

```bash
git add src/franka_policy_runtime/launch/action_test.launch.py
git commit -m "feat: add action_test launch file"
```

---

### Task 5: Write tests

**Files:**
- Create: `src/franka_policy_runtime/test/test_action_test.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for ActionTesterRuntime helpers."""
import numpy as np
import pytest

from franka_policy_runtime.action_test_node import _DummyObserver, _action_dim_label
from franka_policy_runtime.observers.base import BackendObservation


def test_dummy_observer_always_ready():
    observer = _DummyObserver(joint_names=["fr3_joint1"])
    result = observer.observe()
    assert isinstance(result, BackendObservation)
    assert result.ready is True
    assert result.payload == {}


def test_dummy_observer_inherits_base_methods():
    observer = _DummyObserver(joint_names=["fr3_joint1", "fr3_joint2"])
    assert observer.latest_joint_positions() is None

    from geometry_msgs.msg import JointState
    msg = JointState()
    msg.name = ["fr3_joint1", "fr3_joint2"]
    msg.position = [0.1, 0.2]
    msg.velocity = [0.0, 0.0]
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
def test_action_dim_label(action, expected):
    assert _action_dim_label(np.array(action, dtype=float)) == expected


def test_action_dim_label_multi_dim():
    action = np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    assert _action_dim_label(action) == "+dx,+dy"
```

- [ ] **Step 2: Run the tests**

Run: `PYTHONPATH=src/franka_policy_runtime pytest src/franka_policy_runtime/test/test_action_test.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add src/franka_policy_runtime/test/test_action_test.py src/franka_policy_runtime/action_test_node.py src/franka_policy_runtime/setup.py
git commit -m "feat: add ActionTesterRuntime node with tests"
```
