"""Laya's public RGB decision and shared evaluation boundaries."""

import numpy as np
import pytest
import os

from breakout_rl.laya_vision_agent import LayaVisionPolicy


class MockLaya:
    def __init__(self, choice="RIGHT", probabilities=None):
        self.choice = choice
        self.probabilities = probabilities
        self.states = []

    def predict(self, state, questions, **kwargs):
        self.states.append(state)
        assert kwargs["strict"] is True
        answer = {"type": "choice", "choice": self.choice}
        if self.probabilities is not None:
            answer["probabilities"] = self.probabilities
        return {"answers": {"move": answer}}


def test_rgb_frame_preserves_channels_and_maps_actual_action_order():
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[0, 0] = [255, 20, 5]
    agent = MockLaya()
    policy = LayaVisionPolicy(agent, lambda: frame, ["LEFT", "RIGHT", "FIRE", "NOOP"])
    assert policy.select_action(None, rng=np.random.default_rng(101)) == 1
    assert tuple(np.asarray(agent.states[0]["image"])[0, 0]) == (255, 20, 5)
    assert set(agent.states[0]) == {"image"}


@pytest.mark.parametrize("frame", [np.zeros((4, 84, 84), dtype=np.uint8), np.zeros((210, 160, 3))])
def test_rejects_non_rgb_or_non_uint8(frame):
    policy = LayaVisionPolicy(MockLaya(), lambda: frame, ["NOOP", "FIRE", "RIGHT", "LEFT"])
    with pytest.raises(ValueError, match="RGB"):
        policy.select_action(None, rng=np.random.default_rng(101))


@pytest.mark.parametrize("choice", [None, "UP", 2])
def test_rejects_invalid_choice(choice):
    policy = LayaVisionPolicy(MockLaya(choice), lambda: np.zeros((210, 160, 3), dtype=np.uint8), ["NOOP", "FIRE", "RIGHT", "LEFT"])
    with pytest.raises(ValueError, match="choice"):
        policy.select_action(None, rng=np.random.default_rng(101))


def test_rejects_invalid_minimal_action_mapping():
    with pytest.raises(ValueError, match="action"):
        LayaVisionPolicy(MockLaya(), lambda: None, ["NOOP", "FIRE", "RIGHT", "UP"])


def test_rejects_choice_that_is_not_greedy():
    agent = MockLaya("LEFT", {"NOOP": 0.1, "FIRE": 0.1, "RIGHT": 0.7, "LEFT": 0.1})
    policy = LayaVisionPolicy(agent, lambda: np.zeros((210, 160, 3), dtype=np.uint8), ["NOOP", "FIRE", "RIGHT", "LEFT"])
    with pytest.raises(ValueError, match="greedy"):
        policy.select_action(None, rng=np.random.default_rng(101))


def test_nested_model_work_is_counted_once():
    import torch
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(vision_model=torch.nn.Identity(), connector=torch.nn.Identity())

        def forward(self, value):
            return self.encoder.connector(self.encoder.vision_model(value))

    class Agent(MockLaya):
        model = Model()

        def predict(self, state, questions, **kwargs):
            self.model(torch.ones(1))
            return super().predict(state, questions, **kwargs)

    policy = LayaVisionPolicy(Agent(), lambda: np.zeros((210, 160, 3), dtype=np.uint8), ["NOOP", "FIRE", "RIGHT", "LEFT"])
    policy.select_action(None, rng=np.random.default_rng(101))
    assert len(policy.model_latencies) == 1
    assert 0 < policy.decisions[0]["model_seconds"] <= policy.decisions[0]["decision_seconds"]


def test_greedy_ties_preserve_upstream_choice_before_probability_rounding():
    # Upstream rounds probabilities to four decimals after choosing its argmax.
    agent = MockLaya("RIGHT", {"NOOP": 0.25, "FIRE": 0.25, "RIGHT": 0.25, "LEFT": 0.25})
    policy = LayaVisionPolicy(agent, lambda: np.zeros((210, 160, 3), dtype=np.uint8), ["NOOP", "FIRE", "RIGHT", "LEFT"])
    assert [policy.select_action(None, rng=np.random.default_rng(101)) for _ in range(2)] == [2, 2]


def test_shared_evaluator_accounts_for_fire_override():
    import gymnasium as gym
    from breakout_rl.evaluation import evaluate_policy

    class FireEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        action_space = gym.spaces.Discrete(4)

        def get_action_meanings(self):
            return ["NOOP", "FIRE", "RIGHT", "LEFT"]

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def render(self):
            return np.zeros((210, 160, 3), dtype=np.uint8)

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {
                "fire_reset_auto_fire": True, "fire_reset_executed_action": 1,
                "fire_reset_reason": "initial_serve",
            }

    trace = []
    result = evaluate_policy(None, episodes=1, seeds=[101], device="cpu", env_factory=FireEnv,
                             policy_factory=lambda env: LayaVisionPolicy(MockLaya(), env.render, env.get_action_meanings()),
                             step_callback=trace.append)
    assert result.policy_type == "laya-vision"
    assert result.episodes[0].requested_action_distribution["RIGHT"] == 1
    assert result.episodes[0].executed_action_distribution["FIRE"] == 1
    assert trace[0]["requested_action"] == 2
    assert trace[0]["executed_action"] == 1


def test_real_ale_pilot_seed_subset_remains_explicit():
    from breakout_env import make_breakout_env
    from breakout_rl.evaluation import evaluate_policy
    from breakout_rl.evaluation_contract import load_evaluation_contract, breakout_environment_kwargs

    contract = load_evaluation_contract("configs/eval/breakout_contract_v2.json")
    result = evaluate_policy(None, episodes=1, seeds=[101], device="cpu",
                             env_factory=lambda: make_breakout_env(render_mode="rgb_array", **breakout_environment_kwargs(contract)),
                             policy_factory=lambda env: LayaVisionPolicy(MockLaya("NOOP"), env.render, env.unwrapped.get_action_meanings()),
                             metadata={"evaluation_contract_path": "configs/eval/breakout_contract_v2.json",
                                       "evaluation_contract": contract.to_dict(), "evaluation_seed_subset": [101]})
    assert result.contract_provenance["runtime_binding_validated"] is True
    assert result.evaluation_seeds == (101,)


def test_real_laya_ale_decision_when_optional_dependencies_are_available():
    if os.environ.get("LAYA_RUN_INTEGRATION") != "1":
        pytest.skip("Set LAYA_RUN_INTEGRATION=1 with cached optional model weights")
    pytest.importorskip("laya")
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import try_to_load_from_cache
    from breakout_rl.laya_vision_agent import LAYA_MODEL_ID, LAYA_MODEL_REVISION, load_laya_agent
    if not isinstance(try_to_load_from_cache(LAYA_MODEL_ID, "vlm_agent_config.json", revision=LAYA_MODEL_REVISION), str):
        pytest.skip("Pinned optional checkpoint is not cached")
    from breakout_env import make_breakout_env
    from breakout_rl.evaluation_contract import load_evaluation_contract, breakout_environment_kwargs
    agent = load_laya_agent("cpu")
    env = make_breakout_env(render_mode="rgb_array", **breakout_environment_kwargs(load_evaluation_contract("configs/eval/breakout_contract_v2.json")))
    try:
        observation, _ = env.reset(seed=101)
        policy = LayaVisionPolicy(agent, env.render, env.unwrapped.get_action_meanings())
        assert env.action_space.contains(policy.select_action(observation, rng=np.random.default_rng(101)))
    finally:
        env.close()
