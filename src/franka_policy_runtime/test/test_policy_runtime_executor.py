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
