"""Training configuration, logging, and trainer entry points.

PyTorch-heavy trainer modules stay lazy so lightweight configuration and metrics
utilities can be imported without loading the full training runtime.
"""

from breakout_rl.training.config import DQNConfig, SUPPORTED_ALGORITHMS, normalize_algorithm
from breakout_rl.training.metrics import METRIC_FIELDS, MetricsLogger
from breakout_rl.training.reward_shaping import (
    LIFE_LOSS_INFO_KEY,
    reward_design_metadata,
    shape_training_reward,
    validate_life_loss_penalty,
)
from breakout_rl.training.survival import (
    compute_episode_survival_metrics,
    life_losses_per_1000_steps,
)

__all__ = [
    "DQNConfig",
    "SUPPORTED_ALGORITHMS",
    "normalize_algorithm",
    "DQNTrainer",
    "DQNTrainingStepResult",
    "METRIC_FIELDS",
    "MetricsLogger",
    "LIFE_LOSS_INFO_KEY",
    "reward_design_metadata",
    "shape_training_reward",
    "validate_life_loss_penalty",
    "compute_episode_survival_metrics",
    "life_losses_per_1000_steps",
    "NonFiniteTrainingError",
    "TrainingStepCallback",
    "TrainingStepSnapshot",
    "dqn_training_step",
    "resolve_device",
    "seed_everything",
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
    vectorized_names = {
        "VectorScheduleEventKind",
        "VectorizedDQNTrainer",
        "VectorizedTrainingStepCallback",
        "VectorizedTrainingStepSnapshot",
        "crossed_transition_boundaries",
        "ACTION_SELECTION_BATCH_SEMANTICS",
        "STRICT_ACTION_SELECTION_PARITY_RULE",
        "strict_action_selection_parity_satisfied",
    }
    trainer_names = {
        "DQNTrainer",
        "DQNTrainingStepResult",
        "NonFiniteTrainingError",
        "TrainingStepCallback",
        "TrainingStepSnapshot",
        "dqn_training_step",
        "resolve_device",
        "seed_everything",
    }
    if name in vectorized_names:
        from breakout_rl.training import vectorized
        value = getattr(vectorized, name)
    elif name in trainer_names:
        from breakout_rl.training import dqn_trainer
        value = getattr(dqn_trainer, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
