"""Color-cube object pose providers for IsaacLab BC stack tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from franka_policy_runtime.observers.base import _depth_to_meters, _quat_xyzw_to_matrix, BaseObserver


class ColorCubeObjectPoseProvider:
    """Estimate obj2ee from a pure-color cube in RGB-D images."""

    def __init__(
        self,
        target_color: str = "red",
        camera_name: str = "eye_to_hand",
        camera_frame: str = "eye_to_hand_camera_color_optical_frame",
        tcp_frame: str = "fr3_hand_tcp",
        min_pixels: int = 30,
        min_channel: int = 80,
        dominance: float = 1.5,
    ) -> None:
        self.target_color = str(target_color).lower()
        self.camera_name = str(camera_name)
        self.camera_frame = str(camera_frame)
        self.tcp_frame = str(tcp_frame)
        self.min_pixels = int(min_pixels)
        self.min_channel = int(min_channel)
        self.dominance = float(dominance)

    def __call__(self, observer: "BaseObserver") -> np.ndarray | None:
        image, depth, k_matrix = observer.sensor_snapshot(self.camera_name)
        if image is None or depth is None or k_matrix is None or observer.tf_buffer is None:
            return None
        mask = self._color_mask(image)
        if int(np.count_nonzero(mask)) < self.min_pixels:
            return None
        point_camera = self._point_from_mask_depth(mask, depth, k_matrix)
        if point_camera is None:
            return None
        point_eef = self._transform_point(observer.tf_buffer, point_camera)
        if point_eef is None:
            return None
        return np.array([*point_eef.tolist(), 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _color_mask(self, image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return np.zeros(rgb.shape[:2], dtype=bool)
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        if self.target_color == "red":
            return (red >= self.min_channel) & (red >= green * self.dominance) & (red >= blue * self.dominance)
        if self.target_color == "green":
            return (green >= self.min_channel) & (green >= red * self.dominance) & (green >= blue * self.dominance)
        if self.target_color == "blue":
            return (blue >= self.min_channel) & (blue >= red * self.dominance) & (blue >= green * self.dominance)
        raise ValueError(f"unsupported target_color: {self.target_color}")

    def _point_from_mask_depth(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        k_matrix: np.ndarray,
    ) -> np.ndarray | None:
        depth_m = _depth_to_meters(depth)
        valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid):
            return None
        v_coords, u_coords = np.nonzero(valid)
        z_values = depth_m[v_coords, u_coords]
        fx = float(k_matrix[0, 0])
        fy = float(k_matrix[1, 1])
        cx = float(k_matrix[0, 2])
        cy = float(k_matrix[1, 2])
        x_values = (u_coords.astype(np.float64) - cx) * z_values / fx
        y_values = (v_coords.astype(np.float64) - cy) * z_values / fy
        return np.array([
            float(np.median(x_values)),
            float(np.median(y_values)),
            float(np.median(z_values)),
        ], dtype=np.float64)

    def _transform_point(self, tf_buffer: Any, point_camera: np.ndarray) -> np.ndarray | None:
        try:
            from rclpy.time import Time

            lookup_time = Time()
        except ModuleNotFoundError:
            lookup_time = object()
        try:
            transform = tf_buffer.lookup_transform(self.tcp_frame, self.camera_frame, lookup_time, None)
        except TypeError:
            try:
                transform = tf_buffer.lookup_transform(self.tcp_frame, self.camera_frame, lookup_time)
            except Exception:
                return None
        except Exception:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = _quat_xyzw_to_matrix(np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64))
        trans = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        return rot @ np.asarray(point_camera, dtype=np.float64) + trans


class ColorCubeStackObjectProvider:
    """Estimate IsaacLab stack ``object`` observation from blue/red/green cubes."""

    # IsaacLab Franka stack maps cube_1=blue, cube_2=red, cube_3=green.
    cube_colors = ("blue", "red", "green")

    def __init__(
        self,
        camera_name: str = "eye_to_hand",
        camera_frame: str = "eye_to_hand_camera_color_optical_frame",
        base_frame: str = "fr3_link0",
        min_pixels: int = 30,
        min_channel: int = 80,
        dominance: float = 1.5,
    ) -> None:
        self.camera_name = str(camera_name)
        self.camera_frame = str(camera_frame)
        self.base_frame = str(base_frame)
        self.min_pixels = int(min_pixels)
        self._detectors = {
            color: ColorCubeObjectPoseProvider(
                target_color=color,
                camera_name=camera_name,
                camera_frame=camera_frame,
                tcp_frame=base_frame,
                min_pixels=min_pixels,
                min_channel=min_channel,
                dominance=dominance,
            )
            for color in self.cube_colors
        }

    def __call__(self, observer: "BaseObserver") -> np.ndarray | None:
        with observer._lock:
            tcp_position = None if observer._tcp_position is None else observer._tcp_position.copy()
        if tcp_position is None:
            return None

        cube_positions: list[np.ndarray] = []
        for color in self.cube_colors:
            detector = self._detectors[color]
            image, depth, k_matrix = observer.sensor_snapshot(self.camera_name)
            if image is None or depth is None or k_matrix is None or observer.tf_buffer is None:
                return None
            mask = detector._color_mask(image)
            if int(np.count_nonzero(mask)) < self.min_pixels:
                return None
            point_camera = detector._point_from_mask_depth(mask, depth, k_matrix)
            if point_camera is None:
                return None
            point_base = detector._transform_point(observer.tf_buffer, point_camera)
            if point_base is None:
                return None
            cube_positions.append(point_base)

        cube_1_pos, cube_2_pos, cube_3_pos = cube_positions
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        gripper_to_cube_1 = cube_1_pos - tcp_position
        gripper_to_cube_2 = cube_2_pos - tcp_position
        gripper_to_cube_3 = cube_3_pos - tcp_position
        cube_1_to_2 = cube_1_pos - cube_2_pos
        cube_2_to_3 = cube_2_pos - cube_3_pos
        cube_1_to_3 = cube_1_pos - cube_3_pos
        return np.concatenate(
            (
                cube_1_pos,
                identity_quat,
                cube_2_pos,
                identity_quat,
                cube_3_pos,
                identity_quat,
                gripper_to_cube_1,
                gripper_to_cube_2,
                gripper_to_cube_3,
                cube_1_to_2,
                cube_2_to_3,
                cube_1_to_3,
            )
        ).astype(np.float64)
