"""FastAPI app for policy inference."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException

from policy_server.backends.base import BasePolicyBackend, validate_action


def create_app(backend: BasePolicyBackend) -> FastAPI:
    app = FastAPI(title="Franka Policy Server")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "backend_type": backend.backend_type}

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        return backend.metadata()

    @app.post("/act")
    def act(payload: dict[str, Any]) -> dict[str, list[float]]:
        try:
            if "encoded" in payload:
                payload = json.loads(payload["encoded"])
            action = validate_action(backend.predict_payload(payload))
            return {"action": action.astype(float).tolist()}
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="policy inference failed") from exc

    return app
