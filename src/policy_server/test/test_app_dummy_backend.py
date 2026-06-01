import numpy as np
import pytest

pytest.importorskip("fastapi")
from policy_server.app import create_app
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
