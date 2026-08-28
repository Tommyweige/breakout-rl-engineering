"""Training configuration, update, logging, and trainer entry points.

The trainer imports PyTorch. Keep it lazy so CSV-only diagnostics and plotting
can run in a clean process without loading the training runtime first.
"""

from breakout_rl.training.config import DQNConfig
from breakout_rl.training.metrics import METRIC_FIELDS, MetricsLogger

__all__ = [
    "DQNConfig",
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
    "VectorScheduleEventKind",
    "VectorizedDQNTrainer",
    "VectorizedTrainingStepCallback",
    "VectorizedTrainingStepSnapshot",
    "crossed_transition_boundaries",
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
    }:
        raise AttributeError(name)

    if name in {
        "VectorScheduleEventKind",
        "VectorizedDQNTrainer",
        "VectorizedTrainingStepCallback",
        "VectorizedTrainingStepSnapshot",
        "crossed_transition_boundaries",
    }:
        from breakout_rl.training import vectorized

        value = getattr(vectorized, name)
    else:
        from breakout_rl.training import dqn_trainer

        value = getattr(dqn_trainer, name)
    globals()[name] = value
    return value
