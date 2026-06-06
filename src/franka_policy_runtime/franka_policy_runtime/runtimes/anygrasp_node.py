"""AnyGrasp one-shot RGB-D grasp execution runtime."""

from __future__ import annotations

import time

import numpy as np
from franka_msgs.action import Move

from franka_policy_runtime.observers import AnyGraspObserver
from franka_policy_runtime.runtimes.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.utils.pose_math import anygrasp_action_to_base_poses, make_joint_trajectory


class AnyGraspRuntime(PolicyRuntimeBase):
    """Infer one grasp and execute open, approach, close, and retreat phases."""

    def __init__(self) -> None:
        self._phase = "waiting"
        self._completed_at = 0.0
        self._pregrasp_position: np.ndarray | None = None
        self._grasp_position: np.ndarray | None = None
        self._grasp_quat: np.ndarray | None = None
        self._requested_gripper_width = 0.0
        self._selection_cancelled = False
        super().__init__(node_name="anygrasp_runtime")

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        self.declare_parameter("camera_frame", "eye_to_hand_camera_color_optical_frame")
        self.declare_parameter("sensor_name", "eye_to_hand")
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("color_order", "rgb")
        self.declare_parameter("approach_distance", 0.10)
        self.declare_parameter("grasp_to_tcp_rotvec", [0.0, 1.5707963267948966, 0.0])
        self.declare_parameter("grasp_width_offset", 0.005)
        self.declare_parameter("manual_target_selection", True)
        self.declare_parameter("reselect_target_each_grasp", True)
        self.declare_parameter("roi_window_name", "AnyGrasp target selection")
        self.declare_parameter("execute_grasp", False)
        self.declare_parameter("repeat_grasps", False)
        self.declare_parameter("repeat_delay_sec", 2.0)

    def _create_observer(self):
        return AnyGraspObserver(
            joint_names=self._joint_names,
            sensor_name=str(self.get_parameter("sensor_name").value),
            depth_scale=float(self.get_parameter("depth_scale").value),
            color_order=str(self.get_parameter("color_order").value),
        )

    @property
    def _rotation_format(self) -> str:
        return "axis_angle"

    def _control_tick(self) -> None:
        if self._goal_active:
            return
        if self._phase == "done":
            if not bool(self.get_parameter("repeat_grasps").value):
                return
            elapsed = time.monotonic() - self._completed_at
            if elapsed < float(self.get_parameter("repeat_delay_sec").value):
                return
            if bool(self.get_parameter("reselect_target_each_grasp").value):
                self._observer.clear_target_bbox()
            self._phase = "waiting"
        if self._selection_cancelled:
            return

        self._goal_active = True
        try:
            observation = self._observer.observe()
            if not observation.ready:
                self._goal_active = False
                return
            if (bool(self.get_parameter("manual_target_selection").value)
                and self._observer.target_bbox() is None):
                if not self._select_target_bbox(observation.payload["image"]):
                    self._selection_cancelled = True
                    self._goal_active = False
                    return
                observation = self._observer.observe()
            if self._observer.latest_joint_positions() is None:
                self._goal_active = False
                return

            try:
                action = self._request_policy(observation)[0]
                self._observer.update_last_action(action)
                self._prepare_grasp(action)
            except Exception as exc:
                self._goal_active = False
                return

            if not bool(self.get_parameter("execute_grasp").value):
                self._goal_active = False
                self._phase = "done"
                self._completed_at = time.monotonic()
                return

            open_width = float(self.get_parameter("gripper_max_width").value)
            self._phase = "opening"
            self._send_gripper_width(open_width, self._on_gripper_opened)
        except Exception:
            self._goal_active = False
            self._phase = "waiting"
            raise

    def _select_target_bbox(self, rgb_image: np.ndarray) -> bool:
        import cv2
        window_name = str(self.get_parameter("roi_window_name").value)
        bgr_image = cv2.cvtColor(np.asarray(rgb_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        try:
            bbox = cv2.selectROI(window_name, bgr_image, showCrosshair=True)
        finally:
            cv2.destroyWindow(window_name)
        x, y, width, height = (int(v) for v in bbox)
        if width <= 0 or height <= 0:
            return False
        self._observer.set_target_bbox((x, y, width, height))
        return True

    def _prepare_grasp(self, action: np.ndarray) -> None:
        tcp_pose = self._current_tcp_pose()
        if tcp_pose is None:
            raise RuntimeError("TCP pose unavailable for grasp preparation")
        current_position, current_quat = tcp_pose
        approach_dist = float(self.get_parameter("approach_distance").value)
        rotvec = np.asarray(self.get_parameter("grasp_to_tcp_rotvec").value, dtype=np.float64)

        pregrasp, grasp, quat, width = anygrasp_action_to_base_poses(
            action,
            current_position, current_quat,
            approach_distance=approach_dist,
            grasp_to_tcp_rotvec=rotvec,
        )
        self._pregrasp_position = pregrasp
        self._grasp_position = grasp
        self._grasp_quat = quat
        self._requested_gripper_width = width

    def _on_gripper_opened(self, future) -> None:
        # future from gripper send_goal_async
        current_joints = self._observer.latest_joint_positions()
        if current_joints is None:
            self._goal_active = False
            return
        target_joints = self._compute_ik(current_joints, self._pregrasp_position, self._grasp_quat)
        if target_joints is None:
            self._goal_active = False
            return
        self._phase = "approaching"
        self._move_to_pose(target_joints, self._on_pregrasp_reached)

    def _on_pregrasp_reached(self) -> None:
        current_joints = self._observer.latest_joint_positions()
        if current_joints is None:
            self._goal_active = False
            return
        target_joints = self._compute_ik(current_joints, self._grasp_position, self._grasp_quat)
        if target_joints is None:
            self._goal_active = False
            return
        self._phase = "grasping"
        self._move_to_pose(target_joints, self._on_grasp_reached)

    def _on_grasp_reached(self) -> None:
        self._phase = "closing"
        self._send_gripper_width(self._requested_gripper_width, self._on_gripper_closed)

    def _on_gripper_closed(self, future) -> None:
        self._phase = "retreating"
        self._move_to_pose(self._pregrasp_position, self._grasp_quat, self._on_retreat_finished)

    def _on_retreat_finished(self) -> None:
        self._phase = "done"
        self._completed_at = time.monotonic()
        self._goal_active = False

    def _move_to_pose(self, joint_positions, callback) -> None:
        from control_msgs.action import FollowJointTrajectory
        msg = make_joint_trajectory(
            self._joint_names, joint_positions,
            float(self.get_parameter("trajectory_duration_sec").value),
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = msg
        send_future = self._trajectory_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._on_move_goal_sent(f, callback)
        )

    def _on_move_goal_sent(self, future, callback) -> None:
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                callback()
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda f: callback())
        except Exception:
            callback()

    def _send_gripper_width(self, width: float, on_complete) -> None:
        goal = Move.Goal()
        goal.width = width
        goal.speed = float(self.get_parameter("gripper_speed").value)
        send_future = self._gripper_client.send_goal_async(goal)
        send_future.add_done_callback(on_complete)


def main(args=None) -> None:
    run_node(AnyGraspRuntime, args=args)
