from pathlib import Path


_BASE = Path(__file__).parents[1] / "franka_policy_runtime" / "base_node.py"
_VLA = Path(__file__).parents[1] / "franka_policy_runtime" / "vla_node.py"
_BC = Path(__file__).parents[1] / "franka_policy_runtime" / "bc_cube_stack_node.py"


def _read(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def test_policy_runtime_uses_multithreaded_executor():
    source = _BASE.read_text(encoding="utf-8")
    assert "MultiThreadedExecutor" in source
    assert "executor.spin()" in source


def test_policy_runtime_delegates_arm_control_to_cartesian_backend():
    source = _BASE.read_text(encoding="utf-8")
    assert "CartesianPoseBackend" in source
    assert "self._cartesian_backend.ingest_action(action)" in source
    assert "self._cartesian_backend.step_commanded_pose()" in source
    assert "GetPositionIK" not in source
    assert "make_joint_trajectory" not in source


def test_policy_runtime_publishes_pose_stamped_commands():
    source = _BASE.read_text(encoding="utf-8")
    assert "PoseStamped" in source
    assert 'self.declare_parameter("cartesian_command_topic"' in source
    assert "self._cartesian_command_pub" in source


def test_policy_runtime_still_delegates_observation_to_observer():
    source = _read(_BASE, _VLA, _BC)
    assert "from franka_policy_runtime.observers.base import BaseObserver" in source
    assert "self._observer" in source
    assert "OpenVLAObserver" in source
    assert "IsaacLabStackBCObserver" in source


def test_policy_runtime_refreshes_tcp_pose_before_observing_for_inference():
    source = _BASE.read_text(encoding="utf-8")
    inference_loop = source[source.index("    def _inference_loop"):]
    refresh_index = inference_loop.index("self._update_observer_tcp_pose()")
    observe_index = inference_loop.index("observation = self._observer.observe()")
    assert refresh_index < observe_index
