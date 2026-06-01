"""CLI entry point for the policy server."""

from __future__ import annotations

import argparse

import uvicorn

from policy_server.app import create_app
from policy_server.backends.factory import create_backend
from policy_server.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config["backend"]["type"] = args.backend
    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port

    backend = create_backend(config["backend"])
    app = create_app(backend)
    uvicorn.run(
        app,
        host=str(config["server"]["host"]),
        port=int(config["server"]["port"]),
        log_level=str(config["server"]["log_level"]),
    )


if __name__ == "__main__":
    main()
