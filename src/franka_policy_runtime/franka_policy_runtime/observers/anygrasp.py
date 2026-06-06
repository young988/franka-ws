"""RGB-D observation provider for the AnyGrasp backend."""

from __future__ import annotations

import numpy as np

from franka_policy_runtime.observers.base import BackendObservation, BaseObserver


class AnyGraspObserver(BaseObserver):
    """Publish aligned RGB, depth, and camera intrinsics as one observation."""

    def __init__(
        self,
        joint_names: list[str] | None = None,
        *,
        sensor_name: str = "eye_to_hand",
        depth_scale: float = 1000.0,
        color_order: str = "rgb",
    ) -> None:
        super().__init__(joint_names)
        self._sensor_name = str(sensor_name)
        self._depth_scale = float(depth_scale)
        self._color_order = str(color_order).lower()
        self._target_bbox: tuple[int, int, int, int] | None = None
        if self._depth_scale <= 0.0:
            raise ValueError("depth_scale must be positive")
        if self._color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be 'rgb' or 'bgr'")

    def set_target_bbox(self, bbox: tuple[int, int, int, int]) -> None:
        x, y, width, height = (int(value) for value in bbox)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("target bbox must be [x>=0, y>=0, width>0, height>0]")
        with self._lock:
            self._target_bbox = (x, y, width, height)

    def clear_target_bbox(self) -> None:
        with self._lock:
            self._target_bbox = None

    def target_bbox(self) -> tuple[int, int, int, int] | None:
        with self._lock:
            return self._target_bbox

    def observe(self) -> BackendObservation:
        image, depth, camera_matrix = self.sensor_snapshot(self._sensor_name)
        if image is None or depth is None or camera_matrix is None:
            return BackendObservation(ready=False)
        if image.shape[:2] != depth.shape:
            return BackendObservation(ready=False)
        if self._color_order == "bgr":
            image = np.ascontiguousarray(image[..., ::-1])
        payload = {
            "image": image,
            "depth": depth,
            "camera_matrix": camera_matrix.tolist(),
            "depth_scale": self._depth_scale,
        }
        target_bbox = self.target_bbox()
        if target_bbox is not None:
            payload["target_bbox"] = list(target_bbox)
        return BackendObservation(ready=True, payload=payload)
