import importlib.util
import sys
import types
from pathlib import Path


def _load_launch_module():
    sys.modules["launch"] = types.SimpleNamespace(LaunchDescription=object)
    sys.modules["launch.actions"] = types.SimpleNamespace(
        DeclareLaunchArgument=object,
        ExecuteProcess=object,
    )
    sys.modules["launch.substitutions"] = types.SimpleNamespace(
        LaunchConfiguration=object,
        PathJoinSubstitution=object,
    )
    sys.modules["launch_ros.substitutions"] = types.SimpleNamespace(FindPackageShare=object)
    launch_path = Path(__file__).parents[1] / "launch" / "policy_server.launch.py"
    spec = importlib.util.spec_from_file_location("policy_server_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_python_executable_prefers_active_conda_env(monkeypatch, tmp_path):
    conda_prefix = tmp_path / "isaaclab"
    conda_python = conda_prefix / "bin" / "python"
    conda_python.parent.mkdir(parents=True)
    conda_python.write_text("")
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))

    module = _load_launch_module()

    assert module._default_python_executable() == str(conda_python)
