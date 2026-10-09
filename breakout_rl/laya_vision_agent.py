"""Optional frozen Laya policy; the decision boundary receives RGB pixels only."""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import distribution
from typing import Any, Callable, Sequence

import numpy as np

LAYA_CODE_REVISION = "9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c"
LAYA_MODEL_ID = "thaitea/laya-vision"
LAYA_MODEL_REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
ACTION_DESCRIPTIONS = {
    "NOOP": "Do nothing.",
    "FIRE": "Launch the ball.",
    "RIGHT": "Move the paddle right.",
    "LEFT": "Move the paddle left.",
}


def load_laya_agent(device: str = "cuda") -> Any:
    """Require the audited upstream code and exact checkpoint revision."""
    import laya

    source = distribution("laya").read_text("direct_url.json")
    installed_revision = json.loads(source or "{}").get("vcs_info", {}).get("commit_id")
    if installed_revision != LAYA_CODE_REVISION:
        raise RuntimeError("Install Laya from the pinned Git commit in requirements-laya.txt")
    agent = laya.load_vlm(LAYA_MODEL_ID, revision=LAYA_MODEL_REVISION, device=device)
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    return agent


class LayaVisionPolicy:
    policy_type = "laya-vision"

    def __init__(self, agent: Any, render_rgb: Callable, action_names: Sequence[str]):
        self.agent = agent
        self.render_rgb = render_rgb
        self.action_names = tuple(action_names)
        if len(self.action_names) != 4 or set(self.action_names) != set(ACTION_DESCRIPTIONS):
            raise ValueError("Laya requires exactly NOOP/FIRE/RIGHT/LEFT minimal actions")
        self.questions = {"move": {
            "type": "choice",
            "instructions": "Play Atari Breakout. Keep the ball in play and break the bricks. Which action should you take now?",
            "criteria": {name: ACTION_DESCRIPTIONS[name] for name in self.action_names},
        }}
        self.decisions: list[dict] = []
        self.model_latencies: list[float] = []
        self._model_depth = 0
        self._pending_model_times: list[tuple[Any, Any]] = []

    def _model_start(self, module, inputs):
        if self._model_depth == 0:
            if getattr(getattr(self.agent, "device", None), "type", None) == "cuda":
                import torch
                self._model_started = torch.cuda.Event(enable_timing=True)
                self._model_started.record()
            else:
                self._model_started = time.perf_counter()
        self._model_depth += 1

    def _model_end(self, module, inputs, output):
        self._model_depth -= 1
        if self._model_depth == 0:
            if isinstance(self._model_started, float):
                self._pending_model_times.append((self._model_started, time.perf_counter()))
            else:
                import torch
                ended = torch.cuda.Event(enable_timing=True)
                ended.record()
                self._pending_model_times.append((self._model_started, ended))

    def _synchronize(self):
        device = getattr(self.agent, "device", None)
        if getattr(device, "type", None) == "cuda":
            import torch
            torch.cuda.synchronize(device)

    def select_action(self, observation, *, rng) -> int:
        del observation, rng
        from PIL import Image

        self._synchronize()
        started = time.perf_counter()
        frame = self.render_rgb()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected one H x W x 3 uint8 ALE RGB frame")
        image = Image.fromarray(frame.copy())
        converted = time.perf_counter()
        # Hooks observe the supported predict API without replacing its preprocessing.
        handles = []
        if hasattr(self.agent, "model"):
            # predict encodes images outside model.forward; include both image modules.
            modules = [self.agent.model.encoder.vision_model,
                       self.agent.model.encoder.connector, self.agent.model]
            for module in modules:
                handles.extend([module.register_forward_pre_hook(self._model_start),
                                module.register_forward_hook(self._model_end)])
        model_start_index = len(self.model_latencies)
        self._pending_model_times.clear()
        self._model_depth = 0
        try:
            response = self.agent.predict({"image": image}, self.questions, strict=True)
        finally:
            for handle in handles:
                handle.remove()
        self._synchronize()
        self.model_latencies.extend(
            end - start if isinstance(start, float) else start.elapsed_time(end) / 1000
            for start, end in self._pending_model_times
        )
        ended = time.perf_counter()
        try:
            answer = response["answers"]["move"]
            choice = answer["choice"]
        except (KeyError, TypeError) as error:
            raise ValueError("Laya response is missing a move choice") from error
        if answer.get("type") != "choice" or not isinstance(choice, str) or choice not in self.action_names or answer.get("truncated"):
            raise ValueError(f"Invalid Laya choice: {answer!r}")
        probabilities = answer.get("probabilities")
        entropy = None
        if probabilities is not None:
            if not isinstance(probabilities, dict) or set(probabilities) != set(self.action_names):
                raise ValueError("Invalid Laya choice probability mapping")
            values = np.array([probabilities[name] for name in self.action_names], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or abs(values.sum() - 1) > 0.001:
                raise ValueError("Invalid Laya choice probabilities")
            if values.max() - values[self.action_names.index(choice)] > 0.00011:
                raise ValueError("Laya choice is not greedy within probability rounding precision")
            entropy = -sum(float(p) * math.log(float(p)) for p in values if p > 0)
        action = self.action_names.index(choice)
        ended = time.perf_counter()
        model_seconds = sum(self.model_latencies[model_start_index:]) if handles else None
        self.decisions.append({
            "requested_action": action, "choice": choice, "probabilities": probabilities,
            "confidence": answer.get("confidence"), "entropy_nats": entropy,
            "rgb_conversion_seconds": converted - started,
            "model_seconds": model_seconds,
            "predict_non_model_seconds": ended - converted - (model_seconds or 0),
            "decision_seconds": ended - started,
        })
        return action
