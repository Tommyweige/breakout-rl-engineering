"""Reusable data, model, and tensor utilities for the Breakout RL project."""

from typing import Any

__all__ = [
    "ReplayBuffer",
    "ReplayTensorBatch",
    "TransitionBatch",
    "estimate_replay_memory_bytes",
    "LinearEpsilonSchedule",
    "compute_dqn_targets",
    "DQNConfig",
    "DQNTrainer",
    "DQNTrainingStepResult",
    "hard_update",
    "observation_to_tensor",
    "replay_batch_to_tensors",
    "select_epsilon_greedy_action",
    "should_update_target",
    "dqn_training_step",
    "DQNPolicy",
    "EvaluationConfig",
    "EvaluationResult",
    "RandomPolicy",
    "evaluate_policy",
    "DQNPolicy",
    "EvaluationConfig",
    "EvaluationResult",
    "RandomPolicy",
    "evaluate_policy",
]


def __getattr__(name: str) -> Any:
    """Load optional tensor-heavy helpers only when a caller requests them."""

    if name in {"ReplayBuffer", "TransitionBatch", "estimate_replay_memory_bytes"}:
        from breakout_rl.replay import (
            ReplayBuffer,
            TransitionBatch,
            estimate_replay_memory_bytes,
        )

        return {
            "ReplayBuffer": ReplayBuffer,
            "TransitionBatch": TransitionBatch,
            "estimate_replay_memory_bytes": estimate_replay_memory_bytes,
        }[name]

    if name in {"ReplayTensorBatch", "replay_batch_to_tensors"}:
        from breakout_rl.replay_tensors import (
            ReplayTensorBatch,
            replay_batch_to_tensors,
        )

        return {
            "ReplayTensorBatch": ReplayTensorBatch,
            "replay_batch_to_tensors": replay_batch_to_tensors,
        }[name]

    if name == "observation_to_tensor":
        from breakout_rl.tensors import observation_to_tensor

        return observation_to_tensor

    if name in {"LinearEpsilonSchedule", "select_epsilon_greedy_action"}:
        from breakout_rl.exploration import (
            LinearEpsilonSchedule,
            select_epsilon_greedy_action,
        )

        return {
            "LinearEpsilonSchedule": LinearEpsilonSchedule,
            "select_epsilon_greedy_action": select_epsilon_greedy_action,
        }[name]

    if name in {"compute_dqn_targets", "hard_update", "should_update_target"}:
        from breakout_rl.targets import (
            compute_dqn_targets,
            hard_update,
            should_update_target,
        )

        return {
            "compute_dqn_targets": compute_dqn_targets,
            "hard_update": hard_update,
            "should_update_target": should_update_target,
        }[name]

    if name in {
        "DQNConfig",
        "DQNTrainer",
        "DQNTrainingStepResult",
        "dqn_training_step",
    }:
        from breakout_rl.training import (
            DQNConfig,
            DQNTrainer,
            DQNTrainingStepResult,
            dqn_training_step,
        )

        return {
            "DQNConfig": DQNConfig,
            "DQNTrainer": DQNTrainer,
            "DQNTrainingStepResult": DQNTrainingStepResult,
            "dqn_training_step": dqn_training_step,
        }[name]

    if name in {
        "DQNPolicy",
        "EvaluationConfig",
        "EvaluationResult",
        "RandomPolicy",
        "evaluate_policy",
    }:
        from breakout_rl.evaluation import (
            DQNPolicy,
            EvaluationConfig,
            EvaluationResult,
            RandomPolicy,
            evaluate_policy,
        )

        return {
            "DQNPolicy": DQNPolicy,
            "EvaluationConfig": EvaluationConfig,
            "EvaluationResult": EvaluationResult,
            "RandomPolicy": RandomPolicy,
            "evaluate_policy": evaluate_policy,
        }[name]

    if name in {
        "DQNPolicy",
        "EvaluationConfig",
        "EvaluationResult",
        "RandomPolicy",
        "evaluate_policy",
    }:
        from breakout_rl.evaluation import (
            DQNPolicy,
            EvaluationConfig,
            EvaluationResult,
            RandomPolicy,
            evaluate_policy,
        )

        return {
            "DQNPolicy": DQNPolicy,
            "EvaluationConfig": EvaluationConfig,
            "EvaluationResult": EvaluationResult,
            "RandomPolicy": RandomPolicy,
            "evaluate_policy": evaluate_policy,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
