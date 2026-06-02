"""BC cube stack policy runtime node (IsaacLab robomimic checkpoint)."""

from franka_policy_runtime.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.observers import (
    ColorCubeObjectPoseProvider,
    ColorCubeStackObjectProvider,
    IsaacLabStackBCObserver,
)


class BCCubeStackPolicyRuntime(PolicyRuntimeBase):
    """Policy runtime for IsaacLab stack BC checkpoints (structured obs)."""

    def __init__(self) -> None:
        super().__init__(node_name="bc_cube_stack_runtime")

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("object_pose_provider", "color_cube")
        self.declare_parameter("object_target_color", "red")
        self.declare_parameter("object_camera_frame", "eye_to_hand_camera_color_optical_frame")
        self.declare_parameter("object_min_pixels", 30)

    def _create_observer(self):
        pose_provider = None
        object_provider = None
        if str(self.get_parameter("object_pose_provider").value).lower() == "color_cube":
            pose_provider = ColorCubeObjectPoseProvider(
                target_color=str(self.get_parameter("object_target_color").value),
                camera_frame=str(self.get_parameter("object_camera_frame").value),
                tcp_frame=str(self.get_parameter("policy_tcp_frame").value),
                min_pixels=int(self.get_parameter("object_min_pixels").value),
            )
            object_provider = ColorCubeStackObjectProvider(
                camera_frame=str(self.get_parameter("object_camera_frame").value),
                base_frame=str(self.get_parameter("command_frame").value),
                min_pixels=int(self.get_parameter("object_min_pixels").value),
            )
        return IsaacLabStackBCObserver(
            self._joint_names,
            object_pose_provider=pose_provider,
            object_provider=object_provider,
        )


def main(args=None) -> None:
    run_node(BCCubeStackPolicyRuntime, args=args)
