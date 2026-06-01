import numpy as np

from policy_server.backends.factory import create_backend
from policy_server.config import default_config, merge_config


def test_default_config_uses_4bit_openvla_backend():
    config = default_config()

    assert config["backend"]["type"] == "openvla"
    assert config["backend"]["openvla"]["load_in_4bit"] is True
    assert config["backend"]["openvla"]["load_in_8bit"] is False


def test_merge_config_keeps_4bit_defaults_when_overriding_server_port():
    config = merge_config({"server": {"port": 9000}})

    assert config["server"]["port"] == 9000
    assert config["backend"]["openvla"]["load_in_4bit"] is True
    assert config["backend"]["openvla"]["load_in_8bit"] is False


def test_dummy_backend_returns_valid_7d_action():
    backend = create_backend({
        "type": "dummy",
        "dummy": {"action": [0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0]},
    })

    action = backend.predict(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        instruction="move the block",
        unnorm_key=None,
    )

    assert action.shape == (7,)
    assert action.dtype == np.float64
    assert action.tolist() == [0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0]
