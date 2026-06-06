from pathlib import Path


_BASE = Path(__file__).parents[1] / "franka_policy_runtime" / "runtimes" / "base_node.py"
_VLA = Path(__file__).parents[1] / "franka_policy_runtime" / "runtimes" / "vla_node.py"
_BC = Path(__file__).parents[1] / "franka_policy_runtime" / "runtimes" / "bc_cube_stack_node.py"


def _read(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def test_policy_runtime_uses_multithreaded_executor_for_blocking_ik_wait():
    source = _BASE.read_text(encoding="utf-8")

    assert "MultiThreadedExecutor" in source
    assert "executor.spin()" in source
    assert "rclpy.spin(node)" not in source


def test_policy_runtime_separates_ik_wait_from_default_callback_group():
    source = _BASE.read_text(encoding="utf-8")

    assert "ReentrantCallbackGroup" in source
    assert "self._ik_callback_group" in source
    assert "callback_group=self._ik_callback_group" in source
    assert "callback_group=self._control_callback_group" in source


def test_policy_runtime_delegates_observation_to_observer():
    source = _read(_BASE, _VLA, _BC)

    # Base imports the observer interface; subclasses import implementations.
    assert "from franka_policy_runtime.observers.base import BaseObserver" in source
    assert "self._observer" in source
    assert "OpenVLAObserver" in source
    assert "IsaacLabStackBCObserver" in source
    assert "RLObserver" not in source
    # Old monolithic patterns no longer exist.
    assert "from franka_policy_runtime.observers import VLAObserver" not in source
    assert "self._latest_image" not in source


def test_policy_runtime_subscribes_to_instruction_updates():
    source = _BASE.read_text(encoding="utf-8")

    assert "from std_msgs.msg import String" in source
    assert 'self.declare_parameter("instruction_topic", "~/instruction")' in source
    assert "self._instruction_cb" in source
    assert "self._observer.update_instruction(msg)" in source
    assert "payload = self._payload_from_observation(observation)" in source


def test_policy_runtime_refreshes_tcp_pose_before_observing_for_inference():
    source = _BASE.read_text(encoding="utf-8")
    control_tick = source[source.index("    def _control_tick"):]
    refresh_index = control_tick.index("self._update_observer_tcp_pose()")
    observe_index = control_tick.index("observation = self._observer.observe()")

    assert refresh_index < observe_index


def test_policy_runtime_prefers_franka_robot_state_for_tcp_pose():
    source = _BASE.read_text(encoding="utf-8")

    assert "from franka_msgs.msg import FrankaRobotState" in source
    assert 'self.declare_parameter("tcp_pose_source", "franka_state")' in source
    assert 'self.declare_parameter("franka_state_topic", "/franka_robot_state_broadcaster/robot_state")' in source
    assert "self._franka_state_cb" in source
    assert "msg.o_t_ee.pose" in source
    assert "self._latest_franka_state_pose" in source
    update_tcp = source[source.index("    def _update_observer_tcp_pose"):]
    assert "_tcp_pose_from_franka_state" in update_tcp
    assert "_lookup_tcp_pose_from_tf" in update_tcp
