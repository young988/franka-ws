import numpy as np
import pytest

pytest.importorskip("fastapi")
from policy_server.app import create_app
from policy_server.backends.base import BasePolicyBackend
from policy_server.backends.dummy import DummyBackend


def _endpoint(app, path, method):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


def test_health_and_metadata_report_dummy_backend():
    app = create_app(DummyBackend({"action": [0, 0, 0, 0, 0, 0, 1]}))

    assert _endpoint(app, "/health", "GET")() == {"ok": True, "backend_type": "dummy"}
    metadata = _endpoint(app, "/metadata", "GET")()
    assert metadata["backend_type"] == "dummy"
    assert metadata["action_dim"] == 7


def test_act_accepts_plain_image_payload_and_returns_action_object():
    app = create_app(DummyBackend({"action": [0.1, 0, 0, 0, 0, 0, -1]}))
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    response = _endpoint(app, "/act", "POST")(
        {"image": image.tolist(), "instruction": "pick up the cube"}
    )

    assert response == {"action": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]}


def test_act_decodes_payload_for_backend_without_schema_routing():
    class RecordingBackend(BasePolicyBackend):
        backend_type = "recording"

        def __init__(self):
            self.observation = None

        def predict_payload(self, payload):
            self.observation = payload
            return np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

    import base64
    import io

    from PIL import Image as PILImage

    def encode(image):
        buf = io.BytesIO()
        PILImage.fromarray(image).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    backend = RecordingBackend()
    app = create_app(backend)
    eye_to_hand = np.zeros((4, 4, 3), dtype=np.uint8)
    eye_in_hand = np.full((4, 4, 3), 128, dtype=np.uint8)

    response = _endpoint(app, "/act", "POST")({
        "images_b64": {
            "eye_to_hand": encode(eye_to_hand),
            "eye_in_hand": encode(eye_in_hand),
        },
        "image_b64": encode(eye_to_hand),
        "instruction": "stack the cubes",
        "unnorm_key": "bridge_orig",
        "terms": {"joint_pos": [0.0] * 7},
        "availability": {"joint_pos": True},
    })

    assert response == {"action": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    assert sorted(backend.observation["images_b64"]) == ["eye_in_hand", "eye_to_hand"]
    assert backend.observation["terms"]["joint_pos"] == [0.0] * 7
    assert backend.observation["instruction"] == "stack the cubes"
