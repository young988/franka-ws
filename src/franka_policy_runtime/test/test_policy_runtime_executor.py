from pathlib import Path


SOURCE = Path(__file__).parents[1] / "franka_policy_runtime" / "policy_runtime_node.py"


def test_policy_runtime_uses_multithreaded_executor_for_blocking_ik_wait():
    source = SOURCE.read_text(encoding="utf-8")

    assert "MultiThreadedExecutor" in source
    assert "executor.spin()" in source
    assert "rclpy.spin(node)" not in source


def test_policy_runtime_separates_ik_wait_from_default_callback_group():
    source = SOURCE.read_text(encoding="utf-8")

    assert "ReentrantCallbackGroup" in source
    assert "self._ik_callback_group" in source
    assert "callback_group=self._ik_callback_group" in source
    assert "callback_group=self._control_callback_group" in source


def test_policy_runtime_delegates_observation_to_observer():
    source = SOURCE.read_text(encoding="utf-8")

    assert "from franka_policy_runtime.observers import" in source
    assert "observer_type" in source
    assert "self._observer" in source
    assert "VLAObserver" in source
    assert "RLObserver" in source
    assert "self._latest_image" not in source


def test_policy_runtime_subscribes_to_instruction_updates():
    source = SOURCE.read_text(encoding="utf-8")

    assert "from std_msgs.msg import String" in source
    assert 'self.declare_parameter("instruction_topic", "~/instruction")' in source
    assert "self._instruction_cb" in source
    assert "self._observer.update_instruction(msg)" in source
    assert '"instruction": observation.instruction' in source
