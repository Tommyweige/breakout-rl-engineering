"""Training configuration, update, logging, and trainer entry points.

The trainer imports PyTorch. Keep it lazy so CSV-only diagnostics and plotting
can run in a clean process without loading the training runtime first.
"""

from breakout_rl.training.config import DQNConfig, SUPPORTED_ALGORITHMS
from breakout_rl.training.backend_manifest import (
    load_day16_backend_manifest,
    validate_day16_backend_manifest,
)
from breakout_rl.training.metrics import METRIC_FIELDS, MetricsLogger

__all__ = [
    "DQNConfig",
    "SUPPORTED_ALGORITHMS",
    "load_day16_backend_manifest",
    "DQNTrainer",
    "DQNTrainingStepResult",
    "METRIC_FIELDS",
    "MetricsLogger",
    "NonFiniteTrainingError",
    "TrainingStepCallback",
    "TrainingStepSnapshot",
    "dqn_training_step",
    "resolve_device",
    "seed_everything",
    "validate_day16_backend_manifest",
    "ACTION_SELECTION_BATCH_SEMANTICS",
    "STRICT_ACTION_SELECTION_PARITY_RULE",
    "VectorScheduleEventKind",
    "VectorizedDQNTrainer",
    "VectorizedTrainingStepCallback",
    "VectorizedTrainingStepSnapshot",
    "crossed_transition_boundaries",
    "strict_action_selection_parity_satisfied",
]


def __getattr__(name: str):
    if name not in {
        "DQNTrainer",
        "DQNTrainingStepResult",
        "NonFiniteTrainingError",
        "TrainingStepCallback",
        "TrainingStepSnapshot",
        "dqn_training_step",
        "resolve_device",
        "seed_everything",
        "VectorScheduleEventKind",
        "VectorizedDQNTrainer",
        "VectorizedTrainingStepCallback",
        "VectorizedTrainingStepSnapshot",
        "crossed_transition_boundaries",
        "ACTION_SELECTION_BATCH_SEMANTICS",
        "STRICT_ACTION_SELECTION_PARITY_RULE",
        "strict_action_selection_parity_satisfied",
    }:
        raise AttributeError(name)

    if name in {
        "VectorScheduleEventKind",
        "VectorizedDQNTrainer",
        "VectorizedTrainingStepCallback",
        "VectorizedTrainingStepSnapshot",
        "crossed_transition_boundaries",
        "ACTION_SELECTION_BATCH_SEMANTICS",
        "STRICT_ACTION_SELECTION_PARITY_RULE",
        "strict_action_selection_parity_satisfied",
    }:
        from breakout_rl.training import vectorized

        value = getattr(vectorized, name)
    else:
        from breakout_rl.training import dqn_trainer

        value = getattr(dqn_trainer, name)
    globals()[name] = value
    return value
