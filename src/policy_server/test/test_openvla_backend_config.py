import pytest

from policy_server.backends.openvla import OpenVLABackend, get_openvla_prompt


def test_openvla_prompt_uses_current_format_by_default():
    assert get_openvla_prompt("Move The Block", "openvla/openvla-7b") == (
        "In: What action should the robot take to move the block?\nOut:"
    )


def test_openvla_backend_rejects_8bit_when_local_runtime_requires_4bit():
    with pytest.raises(ValueError, match="4-bit"):
        OpenVLABackend({"load_in_4bit": False, "load_in_8bit": True}, lazy_load=True)


def test_openvla_backend_can_be_constructed_without_loading_model():
    backend = OpenVLABackend(
        {
            "openvla_path": "openvla/openvla-7b",
            "load_in_4bit": True,
            "load_in_8bit": False,
        },
        lazy_load=True,
    )

    metadata = backend.metadata()
    assert metadata["backend_type"] == "openvla"
    assert metadata["quantization"] == "4bit"
