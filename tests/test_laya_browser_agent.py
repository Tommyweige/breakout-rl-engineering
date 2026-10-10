"""Fixed inputs must be reused; RGB pixels and decisions must never be cached."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from breakout_rl.laya_browser_agent import LayaBrowserAgent
from breakout_rl.laya_vision_agent import LayaVisionPolicy


@pytest.fixture
def setup(monkeypatch):
    calls = []
    questions = LayaVisionPolicy(None, None, ["NOOP", "FIRE", "RIGHT", "LEFT"]).questions

    def build(*args, **kwargs):
        calls.append("build")
        return {"markers": [1, 2, 3, 4], "truncation": {}}

    def collate(*args, **kwargs):
        calls.append("collate")
        return {key: torch.ones(1, 4, dtype=torch.long) for key in
                ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "option_span")}

    upstream = SimpleNamespace(QTYPES={"choice": 0}, build_vlm_inputs=build, collate_vlm=collate,
                               truncation_answer=lambda *args: {}, truncation_error=lambda *args: ValueError("truncated"),
                               confidence_from_probs=lambda *args: 0.1,
                               vlm_prefix=lambda *args: {"raw_images": torch.zeros(1, 3, 210, 160, dtype=torch.uint8)})
    monkeypatch.setitem(sys.modules, "laya.vlm", upstream)

    class Model:
        def encode_raw_images(self, pixels):
            return pixels[0, 0, 0, 0].float() / 255

        def __call__(self, *args, image_hidden_states, **kwargs):
            value = image_hidden_states
            return torch.stack((value * 0, value * 0, value, 1 - value))[None], None

    agent = SimpleNamespace(device=torch.device("cpu"), model=Model(), cfg={}, prep=None,
                            processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
                            _to_internal=lambda q: {"t": "choice"}, _checkpoint_temperature=lambda *args: 1.0)
    return agent, questions, calls, upstream


def test_template_cached_once_but_each_prediction_uses_current_rgb(setup):
    agent, questions, calls, _ = setup
    fast = LayaBrowserAgent(agent, questions)
    tensor_ids = [id(t) for t in fast.args]
    for value, expected in ((0, "LEFT"), (255, "RIGHT"), (0, "LEFT")):
        frame = Image.fromarray(np.full((210, 160, 3), value, dtype=np.uint8))
        answer = fast.predict({"image": frame}, questions)["answers"]["move"]
        assert answer["choice"] == expected
        assert abs(sum(answer["probabilities"].values()) - 1) <= .001
    assert calls == ["build", "collate"]
    assert [id(t) for t in fast.args] == tensor_ids
    assert fast.mode == "fixed-eager"


def test_rejects_changed_question_extra_state_and_invalid_pixels(setup):
    agent, questions, _, _ = setup
    fast = LayaBrowserAgent(agent, questions)
    state = {"image": Image.fromarray(np.zeros((210, 160, 3), dtype=np.uint8))}
    changed = deepcopy(questions)
    changed["move"]["instructions"] = "changed"
    for s, q, strict in ((state, changed, True), ({**state, "text": "extra"}, questions, True),
                         (state, questions, False)):
        with pytest.raises(ValueError, match="fixed"):
            fast.predict(s, q, strict=strict)
    for rgb in (np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((210, 160, 3), dtype=np.float32)):
        with pytest.raises(ValueError, match="RGB"):
            fast.predict({"image": rgb}, questions)


def test_refuses_truncated_template(setup):
    agent, questions, _, upstream = setup
    upstream.truncation_answer = lambda *args: {"instructions": True}
    with pytest.raises(ValueError, match="truncated"):
        LayaBrowserAgent(agent, questions)


def test_browser_lightweight_policy_does_not_force_diagnostic_synchronization(setup, monkeypatch):
    agent, questions, _, _ = setup
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    policy = LayaVisionPolicy(LayaBrowserAgent(agent, questions), lambda: frame,
                             ["NOOP", "FIRE", "RIGHT", "LEFT"], diagnostics=False)
    monkeypatch.setattr(policy, "_synchronize", lambda: pytest.fail("diagnostic synchronization"))
    assert policy.select_action(None, rng=None) == 3
    assert policy.decisions[-1]["decision_seconds"] >= 0
