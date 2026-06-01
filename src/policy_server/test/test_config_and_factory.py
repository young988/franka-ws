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

def test_factory_does_not_expose_generic_rl_backend():
    try:
        create_backend({"type": "rl"})
    except ValueError as exc:
        assert "unknown policy backend type: rl" in str(exc)
    else:
        raise AssertionError("generic rl backend should not exist")


def test_bc_isaaclab_stack_backend_accepts_required_terms_payload():
    backend = create_backend({
        "type": "bc_isaaclab_stack",
        "bc_isaaclab_stack": {
            "required_terms": ["eef_pos", "eef_quat", "gripper_pos", "object"],
            "checkpoint_path": "",
            "fallback_action": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    })

    action = backend.predict_payload({
        "terms": {
            "eef_pos": [0.0, 0.0, 0.0],
            "eef_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_pos": [0.0, 0.0],
            "object": [0.0] * 39,
        },
    })

    assert action.tolist() == [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_bc_isaaclab_stack_default_points_to_selected_checkpoint():
    config = default_config()["backend"]["bc_isaaclab_stack"]

    assert config["checkpoint_path"].endswith("bc_cube_stack/models/model_epoch_2000.pth")
    assert config["required_terms"] == ["eef_pos", "eef_quat", "gripper_pos", "object"]
    assert config["term_shapes"] == {
        "eef_pos": [3],
        "eef_quat": [4],
        "gripper_pos": [2],
        "object": [39],
    }
