"""ActionTesterRuntime node.

Sends single-dimension action deltas through the full IK-to-trajectory
pipeline and records TCP outcomes for validation.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

from franka_policy_runtime.runtimes.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.observers.base import BaseObserver
from franka_policy_runtime.utils.pose_math import (
    DummyObserver,
    action_dim_label,
    apply_tcp_delta,
)

_DEFAULT_TEST_SEQUENCE = [
    [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # +dx
    [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -dx
    [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],   # +dy
    [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0],  # -dy
    [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],   # +dz
    [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0],  # -dz
    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],    # +rx
    [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0],   # -rx
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],    # +ry
    [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],   # -ry
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],    # +rz
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],   # -rz
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],    # gripper_open
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],   # gripper_close
]
_DEFAULT_TEST_SEQUENCE_FLAT = [
    value for action in _DEFAULT_TEST_SEQUENCE for value in action
]


class ActionTesterRuntime(PolicyRuntimeBase):
    """Policy runtime that replays a test sequence and measures TCP outcomes.

    Each step sends a single-dimension action delta through
    IK -> trajectory -> TF, then records the resulting TCP pose
    and computes position error against the target.  Results are
    logged and optionally written to CSV.
    """

    def __init__(self, node_name: str = "action_tester_runtime") -> None:
        super().__init__(node_name=node_name)

        self._test_sequence = self._load_test_sequence()
        self._step_interval_sec: float = float(
            self.get_parameter("step_interval_sec").value
        )
        self._tolerance_pos_m: float = float(
            self.get_parameter("tolerance_pos_m").value
        )
        self._max_ik_retries_per_step: int = int(
            self.get_parameter("max_ik_retries_per_step").value
        )
        self._settle_before_measure_sec: float = float(
            self.get_parameter("settle_before_measure_sec").value
        )
        self._csv_output_path: str = str(
            self.get_parameter("csv_output_path").value
        )
        self._action_scale_val: float = float(
            self.get_parameter("action_scale").value
        )

        self._step_index: int = 0
        self._results: list[dict] = []
        self._pending_step_data: dict | None = None
        self._retry_timer = None
        self._settle_timer = None
        self._measure_after_ns: int | None = None
        self._ik_retries_for_step = 0
        self._control_tick_lock = threading.Lock()
        self._next_control_time = 0.0

        self.get_logger().info(
            f"ActionTesterRuntime ready: {len(self._test_sequence)} steps, "
            f"interval={self._step_interval_sec:.1f}s, "
            f"tolerance={self._tolerance_pos_m:.4f}m"
        )

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("test_sequence_flat", _DEFAULT_TEST_SEQUENCE_FLAT)
        self.declare_parameter("step_interval_sec", 2.0)
        self.declare_parameter("tolerance_pos_m", 0.01)
        self.declare_parameter("max_ik_retries_per_step", 5)
        self.declare_parameter("settle_before_measure_sec", 1.0)
        self.declare_parameter("csv_output_path", "")

    def _create_observer(self) -> BaseObserver:
        return DummyObserver(joint_names=self._joint_names)

    def _load_test_sequence(self) -> list[list[float]]:
        flat = list(self.get_parameter("test_sequence_flat").value)
        if len(flat) % 7 != 0:
            raise ValueError(
                "test_sequence_flat length must be a multiple of 7, "
                f"got {len(flat)}"
            )
        return [
            [float(value) for value in flat[index:index + 7]]
            for index in range(0, len(flat), 7)
        ]

    @property
    def _unnorm_key(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Policy request (overridden — reads from test sequence)
    # ------------------------------------------------------------------

    def _request_policy(self, observation) -> np.ndarray:
        if self._step_index >= len(self._test_sequence):
            return np.zeros((1, 7), dtype=float)
        action = np.asarray(self._test_sequence[self._step_index], dtype=float)
        return action.reshape(1, 7)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        if not self._control_tick_lock.acquire(blocking=False):
            return

        try:
            if time.monotonic() < self._next_control_time:
                return

            if self._goal_active:
                return

            if self._step_index >= len(self._test_sequence):
                self._log_summary_and_shutdown()
                return

            if not self._runtime_ready():
                self._schedule_next()
                return

            self._goal_active = True

            # (1) Get current TCP pose
            tcp_pose = self._update_observer_tcp_pose()
            if tcp_pose is None:
                self.get_logger().warning(
                    f"TF lookup failed at step {self._step_index}; skipping"
                )
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return
            pre_position, pre_quat = tcp_pose

            # (2) Get next action from test sequence
            action = np.asarray(
                self._test_sequence[self._step_index], dtype=float
            )

            # (3) Handle gripper
            self._handle_gripper(action)

            # (4) Compute target pose via TCP delta
            target_position, target_quat = apply_tcp_delta(
                pre_position,
                pre_quat,
                action,
                action_scale=self._action_scale_val,
                rotation_format=self._rotation_format,
            )

            # (5) Run IK
            current_joints = self._observer.latest_joint_positions()
            if current_joints is None:
                self.get_logger().warning(
                    f"No joint positions at step {self._step_index}; skipping"
                )
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return

            target_joints = self._compute_ik(
                current_joints, target_position, target_quat
            )
            if target_joints is None:
                self._goal_active = False
                if self._retry_or_skip_after_ik_failure():
                    self._step_index += 1
                self._schedule_next()
                return
            self._ik_retries_for_step = 0

            # (6) Stash pre/target data for post-trajectory comparison
            self._pending_step_data = {
                "step": self._step_index,
                "dim_label": action_dim_label(action),
                "action": action.tolist(),
                "pre_position": pre_position.tolist(),
                "pre_quat_xyzw": pre_quat.tolist(),
                "target_position": target_position.tolist(),
                "target_quat_xyzw": target_quat.tolist(),
            }

            # (7) Send trajectory goal
            self._send_trajectory_goal(target_joints)
            if not self._goal_active:
                self._step_index += 1
                self._schedule_next()

        except Exception as exc:
            self.get_logger().warning(
                f"Unexpected error at step {self._step_index}: {exc}; skipping"
            )
            self._goal_active = False
            self._step_index += 1
            self._schedule_next()
        finally:
            self._control_tick_lock.release()

    def _trajectory_goal_response_cb(self, future) -> None:
        """Handle trajectory goal response — retry on failure/rejection."""
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warning(
                f"Trajectory goal failed at step {self._step_index}: {exc}"
            )
            self._goal_active = False
            self._active_goal_handle = None
            self._schedule_next()
            return
        if not goal_handle.accepted:
            self.get_logger().warning(
                f"Trajectory goal rejected at step {self._step_index}"
            )
            self._goal_active = False
            self._active_goal_handle = None
            self._schedule_next()
            return
        self._active_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            self._trajectory_result_cb
        )

    def _trajectory_result_cb(self, future) -> None:
        """Wait for the robot to settle before measuring TCP error."""
        try:
            _ = future.result()
        except Exception:
            pass

        self._active_goal_handle = None

        if self._settle_before_measure_sec <= 0.0:
            self._measure_after_ns = None
            self._record_post_trajectory_result()
            return

        if self._settle_timer is not None:
            self._settle_timer.cancel()

        self._measure_after_ns = (
            self.get_clock().now().nanoseconds
            + int(self._settle_before_measure_sec * 1_000_000_000)
        )
        self.get_logger().info(
            f"Waiting {self._settle_before_measure_sec:.2f}s before measuring TCP error"
        )

        def _timer_cb() -> None:
            if self._settle_timer is not None:
                self._settle_timer.cancel()
                self._settle_timer = None
            self._record_post_trajectory_result()

        self._settle_timer = self.create_timer(
            self._settle_before_measure_sec,
            _timer_cb,
            callback_group=self._control_callback_group,
        )

    def _record_post_trajectory_result(self) -> None:
        """Record post-trajectory TCP and compare against target."""
        try:
            if not self._tcp_pose_is_fresh_for_measurement():
                self._schedule_measure_retry()
                return

            if self._pending_step_data is not None:
                pending = self._pending_step_data
                self._pending_step_data = None

                # Record settled TCP from the configured TCP pose source.
                tcp_pose = self._update_observer_tcp_pose()
                if tcp_pose is not None:
                    post_position, post_quat = tcp_pose
                else:
                    post_position = np.full(3, float("nan"))
                    post_quat = np.full(4, float("nan"))

                target_position = np.asarray(pending["target_position"], dtype=float)
                pos_err = float(np.linalg.norm(post_position - target_position))
                ok = pos_err <= self._tolerance_pos_m

                result = {
                    "step": pending["step"],
                    "dim_label": pending["dim_label"],
                    "action": pending["action"],
                    "pre_position": pending["pre_position"],
                    "pre_quat_xyzw": pending["pre_quat_xyzw"],
                    "target_position": pending["target_position"],
                    "target_quat_xyzw": pending["target_quat_xyzw"],
                    "post_position": post_position.tolist(),
                    "post_quat_xyzw": post_quat.tolist(),
                    "target_error_pos_m": pos_err,
                    "ok": ok,
                }
                self._results.append(result)

                status = "OK" if ok else "FAIL"
                self.get_logger().info(
                    f"[{pending['step']:2d}] {pending['dim_label']:20s}  "
                    f"err={pos_err:.4f}m  {status}"
                )
        finally:
            if self._pending_step_data is not None:
                return
            self._measure_after_ns = None
            self._goal_active = False
            self._step_index += 1
            self._ik_retries_for_step = 0
            if self._step_index >= len(self._test_sequence):
                self._log_summary_and_shutdown()
            else:
                self._schedule_next()

    def _tcp_pose_is_fresh_for_measurement(self) -> bool:
        source = str(self.get_parameter("tcp_pose_source").value).lower()
        if source not in {"current_pose", "franka_current_pose", "franka_state_broadcaster"}:
            return True
        if self._measure_after_ns is None:
            return True
        received_ns = self._latest_current_pose_received_ns
        return received_ns is not None and received_ns >= self._measure_after_ns

    def _schedule_measure_retry(self) -> None:
        if self._settle_timer is not None:
            self._settle_timer.cancel()

        def _timer_cb() -> None:
            if self._settle_timer is not None:
                self._settle_timer.cancel()
                self._settle_timer = None
            self._record_post_trajectory_result()

        self._settle_timer = self.create_timer(
            0.05,
            _timer_cb,
            callback_group=self._control_callback_group,
        )

    def _retry_or_skip_after_ik_failure(self) -> bool:
        self._ik_retries_for_step += 1
        if self._ik_retries_for_step <= self._max_ik_retries_per_step:
            self.get_logger().warning(
                f"IK failed at step {self._step_index}; retry "
                f"{self._ik_retries_for_step}/{self._max_ik_retries_per_step}"
            )
            return False
        self.get_logger().warning(
            f"IK failed at step {self._step_index}; skipping after "
            f"{self._max_ik_retries_per_step} retries"
        )
        self._ik_retries_for_step = 0
        return True

    def _schedule_next(self) -> None:
        """Schedule the next control tick after step_interval_sec."""
        self._next_control_time = time.monotonic() + self._step_interval_sec
        if self._retry_timer is not None:
            self._retry_timer.cancel()

        def _timer_cb() -> None:
            if self._retry_timer is not None:
                self._retry_timer.cancel()
                self._retry_timer = None
            self._next_control_time = 0.0
            self._control_tick()

        self._retry_timer = self.create_timer(
            self._step_interval_sec,
            _timer_cb,
            callback_group=self._control_callback_group,
        )

    def _runtime_ready(self) -> bool:
        if not self._ik_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warning(
                "Waiting for MoveIt IK service", throttle_duration_sec=2.0)
            return False
        if not self._trajectory_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warning(
                "Waiting for trajectory action server", throttle_duration_sec=2.0)
            return False
        if self._observer.latest_joint_positions() is None:
            self.get_logger().warning(
                "Waiting for joint states", throttle_duration_sec=2.0)
            return False
        if self._update_observer_tcp_pose() is None:
            self.get_logger().warning(
                "Waiting for TCP pose", throttle_duration_sec=2.0)
            return False
        return True

    # ------------------------------------------------------------------
    # Summary / CSV / Shutdown
    # ------------------------------------------------------------------

    def _log_summary_and_shutdown(self) -> None:
        """Log a summary table of results and optionally write CSV."""
        n = len(self._results)
        if n == 0:
            self.get_logger().info("No test results recorded.")
            self._shutdown()
            return

        n_ok = sum(1 for r in self._results if r["ok"])
        n_fail = n - n_ok
        errors = [
            r["target_error_pos_m"]
            for r in self._results
            if not math.isnan(r["target_error_pos_m"])
        ]
        mean_err = float(np.mean(errors)) if errors else float("nan")
        max_err = float(np.max(errors)) if errors else float("nan")
        p95_err = (
            float(np.percentile(errors, 95)) if errors else float("nan")
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Action Tester Summary")
        self.get_logger().info(
            f"  Steps: {n} total  |  {n_ok} OK  |  {n_fail} FAIL"
        )
        self.get_logger().info(
            f"  Position errors: mean={mean_err:.4f}m  "
            f"max={max_err:.4f}m  p95={p95_err:.4f}m"
        )
        for r in self._results:
            status = "OK" if r["ok"] else "FAIL"
            self.get_logger().info(
                f"  {r['dim_label']:20s}  err={r['target_error_pos_m']:.4f}m  {status}"
            )
        self.get_logger().info("=" * 60)

        if self._csv_output_path:
            self._write_csv()

        self._shutdown()

    def _write_csv(self) -> None:
        """Write _results to a CSV file."""
        fieldnames = [
            "step",
            "dim_label",
            "action",
            "pre_position",
            "pre_quat_xyzw",
            "target_position",
            "target_quat_xyzw",
            "post_position",
            "post_quat_xyzw",
            "target_error_pos_m",
            "ok",
        ]
        path = Path(self._csv_output_path)
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._results)
            self.get_logger().info(f"Results written to {path}")
        except OSError as exc:
            self.get_logger().error(f"Failed to write CSV: {exc}")

    def _shutdown(self) -> None:
        """Destroy the node and shut down rclpy."""
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> None:
    run_node(ActionTesterRuntime, args=args)
