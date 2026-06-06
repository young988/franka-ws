"""VLA (OpenVLA) policy runtime node."""

from franka_policy_runtime.runtimes.base_node import PolicyRuntimeBase, run_node
from franka_policy_runtime.observers import OpenVLAObserver


class VLAPolicyRuntime(PolicyRuntimeBase):
    """Policy runtime for OpenVLA models (image + instruction → action)."""

    def __init__(self) -> None:
        super().__init__(node_name="vla_policy_runtime")

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("instruction", "pick up the cube")
        self.declare_parameter("unnorm_key", "fractal")

    def _create_observer(self):
        return OpenVLAObserver(
            joint_names=self._joint_names,
            instruction=str(self.get_parameter("instruction").value),
        )

    @property
    def _unnorm_key(self) -> str:
        return str(self.get_parameter("unnorm_key").value)

    @property
    def _rotation_format(self) -> str:
        return "rpy"


def main(args=None) -> None:
    run_node(VLAPolicyRuntime, args=args)
