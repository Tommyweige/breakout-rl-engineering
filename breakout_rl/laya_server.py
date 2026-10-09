"""Loopback-only RGB decision bridge for the local browser demo."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import numpy as np

from breakout_rl.laya_vision_agent import LayaVisionPolicy

RGB_BYTES = 210 * 160 * 3
ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT")


class LayaDecisionService:
    def __init__(self, agent, runtime: dict):
        self.runtime = runtime
        self.decisions = 0
        self.frame = np.zeros((210, 160, 3), dtype=np.uint8)
        self.policy = LayaVisionPolicy(agent, lambda: self.frame, ACTION_NAMES)

    def predict_rgb(self, payload: bytes) -> dict:
        if len(payload) != RGB_BYTES:
            raise ValueError("Expected exactly one 160 x 210 uint8 RGB frame")
        self.frame = np.frombuffer(payload, dtype=np.uint8).reshape(210, 160, 3)
        action = self.policy.select_action(None, rng=None)
        decision = self.policy.decisions[-1]
        probabilities = decision["probabilities"]
        result = {"actionIndex": action, "action": ACTION_NAMES[action],
                  "probabilities": [probabilities[name] for name in ACTION_NAMES] if probabilities is not None else None,
                  "inferenceMs": decision["decision_seconds"] * 1000, **self.runtime}
        self.decisions += 1
        self.policy.decisions.clear()
        self.policy.model_latencies.clear()
        return result


def create_laya_server(service: LayaDecisionService, port: int = 8766) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/api/laya/health":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            self.reply(200, {"ready": True, "decisions": service.decisions, **service.runtime})

        def do_POST(self):
            if self.path != "/api/laya/predict":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1"):
                    self.reply(403, {"error": "Local browser origin required"})
                    return
            if self.headers.get("Content-Type") != "application/octet-stream" or self.headers.get("Content-Length") != str(RGB_BYTES):
                self.reply(400, {"error": "Expected a complete uint8 RGB frame"})
                return
            try:
                self.connection.settimeout(5)
                result = service.predict_rgb(self.rfile.read(RGB_BYTES))
                self.reply(200, result)
            except (ValueError, TimeoutError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                self.reply(500, {"error": f"Laya inference failed: {error}"})

        def log_message(self, format, *args):
            # Keep per-frame requests out of the console; runtime facts are in /health.
            pass

    return HTTPServer(("127.0.0.1", port), Handler)
