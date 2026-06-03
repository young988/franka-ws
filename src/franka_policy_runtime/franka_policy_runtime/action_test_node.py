"""ActionTesterRuntime node.

Sends single-dimension action deltas through the full IK-to-trajectory
pipeline and records TCP outcomes for validation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import rclpy

from franka_policy_runtime.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.reference import (
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


class ActionTesterRuntime(PolicyRuntimeBase):
    """Policy runtime that replays a test sequence and measures TCP outcomes.

    Each step sends a single-dimension action delta through
    IK -> trajectory -> TF, then records the resulting TCP pose
    and computes position error against the target.  Results are
    logged and optionally written to CSV.
    """

    def __init__(self, node_name: str = "action_tester_runtime") -> None:
        super().__init__(node_name=node_name)

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

        self._step_index: int = 0
        self._results: list[dict] = []
        self._pending_step_data: dict | None = None

        self.get_logger().info(
            f"ActionTesterRuntime ready: {len(self._test_sequence)} steps, "
            f"interval={self._step_interval_sec:.1f}s, "
            f"tolerance={self._tolerance_pos_m:.4f}m"
        )

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("test_sequence", _DEFAULT_TEST_SEQUENCE)
        self.declare_parameter("step_interval_sec", 2.0)
        self.declare_parameter("tolerance_pos_m", 0.01)
        self.declare_parameter("csv_output_path", "")

    def _create_observer(self) -> BaseObserver:
        return DummyObserver(joint_names=self._joint_names)

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
        if self._goal_active:
            return

        if self._step_index >= len(self._test_sequence):
            self._log_summary_and_shutdown()
            return

        self._goal_active = True

        try:
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
                self.get_logger().warning(
                    f"IK failed at step {self._step_index}; skipping"
                )
                self._goal_active = False
                self._step_index += 1
                self._schedule_next()
                return

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

        except Exception:
            self.get_logger().warning(
                f"Unexpected error at step {self._step_index}; skipping"
            )
            self._goal_active = False
            self._step_index += 1
            self._schedule_next()

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
        """Record post-trajectory TCP and compare against target."""
        try:
            _ = future.result()
        except Exception:
            pass

        self._goal_active = False
        self._active_goal_handle = None

        try:
            if self._pending_step_data is not None:
                pending = self._pending_step_data
                self._pending_step_data = None

                # Record post-trajectory TCP from TF
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
            self._step_index += 1
            if self._step_index >= len(self._test_sequence):
                self._log_summary_and_shutdown()
            else:
                self._schedule_next()

    def _schedule_next(self) -> None:
        """Schedule the next control tick after step_interval_sec."""
        self.create_timer(
            self._step_interval_sec,
            self._control_tick,
            callback_group=self._control_callback_group,
            oneshot=True,
        )

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
