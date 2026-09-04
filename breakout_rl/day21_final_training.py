"""Protocol, health gates, and selection helpers for Day 21 final training.

The module keeps the long-training decision process machine-readable.  It does
not start a training process; the CLI runner owns environment lifecycle and
persists the runtime evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.day20_comparison import (
    build_day20_report,
    load_day20_config,
    relative_path,
    write_json,
)
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.evaluation_artifacts import ACTION_DISTRIBUTION_SEMANTICS
from breakout_rl.training.backend_manifest import load_day16_backend_manifest
from breakout_rl.training.config import DQNConfig


DAY21_SCHEMA_VERSION = 1
DAY21_EXPERIMENT_ID = "day21-final-long-training"
DAY21_WINNER_FAMILY_ID = "dueling_double_dqn"
DAY21_ALGORITHM = "double_dqn"
DAY21_ARCHITECTURE = "dueling"
DAY21_TRAINING_SEEDS: tuple[int, int, int] = (1011, 2022, 3033)
DAY21_STAGE_TARGETS: dict[str, int] = {
    "stage_a_1m": 1_000_000,
    "stage_b_2_5m": 2_500_000,
    "stage_c_5m": 5_000_000,
}
DAY21_STAGE_ORDER: tuple[str, ...] = tuple(DAY21_STAGE_TARGETS)
DAY21_SELECTION_SEEDS: tuple[int, int, int] = (101, 202, 303)
DAY21_HOLDOUT_GROUP_SEEDS: tuple[int, int, int] = (404, 505, 606)
DAY21_EPISODES_PER_GROUP = 5
DAY21_MAX_STAGE_B_CANDIDATES = 2
DAY21_MAX_STAGE_C_CANDIDATES = 1
DAY21_DEFAULT_CUDA_HEADROOM_BYTES = 1_073_741_824
DAY21_REQUIRED_TRAINING_METRICS: tuple[str, ...] = (
    "loss",
    "q_mean",
    "q_max",
    "target_mean",
    "td_error_mean_abs",
    "gradient_norm",
    "sps",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: expected a JSON object")
    return dict(payload)


def _repository_root(source: Path) -> Path:
    for candidate in (source.parent, *source.parents):
        if (candidate / "breakout_rl").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return source.parent


def _resolve_reference(
    value: Any,
    *,
    source: Path,
    repository_root: Path,
) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    for option in (repository_root / candidate, source.parent / candidate):
        if option.is_file():
            return option.resolve()
    return (repository_root / candidate).resolve()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _unique_ints(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    parsed = tuple(_positive_int(item, name=name) for item in value)
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique values")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _artifact_reference(path: Path, *, start: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative_path(path, start=start),
        "sha256": sha256_file(path),
    }


def manifest_reference(path: str | Path, *, start: str | Path) -> str:
    """Return a portable manifest path, falling back to absolute on another drive."""

    try:
        return relative_path(path, start=start)
    except ValueError:
        return str(Path(path).resolve())


def _source_value(payload: Mapping[str, Any], name: str) -> Any:
    source = payload.get("source_of_truth")
    if not isinstance(source, Mapping):
        raise ValueError("source_of_truth must be an object")
    if name not in source:
        raise ValueError(f"source_of_truth is missing {name}")
    return source[name]


def _evaluation_settings(
    value: Any,
    *,
    name: str,
    expected_seeds: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], int, float, bool]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    seeds = _unique_ints(value.get("seeds"), name=f"{name}.seeds")
    if expected_seeds is not None and seeds != expected_seeds:
        raise ValueError(f"{name}.seeds must be {list(expected_seeds)}")
    episodes = _positive_int(
        value.get("episodes_per_seed", value.get("episodes_per_group")),
        name=f"{name}.episodes_per_seed",
    )
    epsilon = _finite_float(value.get("epsilon"), name=f"{name}.epsilon")
    if epsilon != 0.0:
        raise ValueError(f"{name}.epsilon must be zero")
    raw_reward = value.get("raw_reward")
    if raw_reward is not True:
        raise ValueError(f"{name}.raw_reward must be true")
    return seeds, episodes, epsilon, True


@dataclass(frozen=True)
class Day21FinalTrainingConfig:
    """Validated Day 21 final-training protocol."""

    source_path: Path
    repository_root: Path
    experiment_id: str
    day20_config_path: Path
    day20_selection_path: Path
    backend_manifest_path: Path
    contract_path: Path
    evaluation_config_path: Path
    day20_selection: Mapping[str, Any]
    backend_manifest: Mapping[str, Any]
    contract: BreakoutEvaluationContractV2
    evaluation_config: Mapping[str, Any]
    backend_config: DQNConfig
    winner_family_id: str
    algorithm: str
    architecture: str
    training_seeds: tuple[int, ...]
    day20_training_seeds: tuple[int, ...]
    stage_targets: Mapping[str, int]
    selection_seeds: tuple[int, ...]
    selection_episodes_per_seed: int
    holdout_group_seeds: tuple[int, ...]
    holdout_episodes_per_group: int
    requested_device: str
    precision: str
    sequential: bool
    continuous_run: bool
    require_cuda: bool
    cuda_headroom_bytes: int
    selection_rule: Mapping[str, Any]
    health_rule: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def backend_id(self) -> str | None:
        value = self.backend_manifest.get("backend_id")
        return None if value is None else str(value)

    @property
    def selection_concrete_seeds(self) -> tuple[int, ...]:
        return expand_concrete_episode_seeds(
            self.selection_seeds,
            episodes_per_seed=self.selection_episodes_per_seed,
        )

    @property
    def holdout_concrete_seeds(self) -> tuple[int, ...]:
        return expand_concrete_episode_seeds(
            self.holdout_group_seeds,
            episodes_per_seed=self.holdout_episodes_per_group,
        )

    @property
    def contract_provenance(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract.contract_id,
            "contract_path": relative_path(
                self.contract_path,
                start=self.repository_root,
            ),
            "contract_sha256": sha256_file(self.contract_path),
            "semantics": self.contract.to_dict(),
        }

    @property
    def source_hashes(self) -> dict[str, str]:
        return {
            "day20_config": sha256_file(self.day20_config_path),
            "day20_selection": sha256_file(self.day20_selection_path),
            "backend_manifest": sha256_file(self.backend_manifest_path),
            "contract": sha256_file(self.contract_path),
            "evaluation_config": sha256_file(self.evaluation_config_path),
        }

    def training_config(self, seed: int) -> DQNConfig:
        """Return the frozen Day 20 backend with only seed/budget overridden."""

        return self.backend_config.with_overrides(
            algorithm=self.algorithm,
            architecture=self.architecture,
            seed=int(seed),
            total_steps=int(self.stage_targets["stage_c_5m"]),
            checkpoint_interval=100_000,
            contract_id=self.contract.contract_id,
            contract_path=relative_path(
                self.contract_path,
                start=self.repository_root,
            ),
        )

    def protocol(self) -> dict[str, Any]:
        return {
            "winner_family_id": self.winner_family_id,
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "training_seeds": list(self.training_seeds),
            "day20_training_seeds": list(self.day20_training_seeds),
            "stage_targets": dict(self.stage_targets),
            "stage_order": list(DAY21_STAGE_ORDER),
            "max_stage_b_candidates": DAY21_MAX_STAGE_B_CANDIDATES,
            "max_stage_c_candidates": DAY21_MAX_STAGE_C_CANDIDATES,
            "stage_c_policy": dict(
                self.raw.get("execution", {}).get("stage_c_policy", {})
            ),
            "selection_seeds": list(self.selection_seeds),
            "selection_concrete_seeds": list(self.selection_concrete_seeds),
            "selection_episodes_per_seed": self.selection_episodes_per_seed,
            "holdout_group_seeds": list(self.holdout_group_seeds),
            "holdout_concrete_seeds": list(self.holdout_concrete_seeds),
            "holdout_episodes_per_group": self.holdout_episodes_per_group,
            "evaluation_order": self.raw.get("final_holdout", {}).get(
                "evaluation_order"
            ),
            "holdout_seed_scope": (
                "independent holdout seeds; Contract v2 environment semantics "
                "are unchanged and selection concrete seeds remain canonical"
            ),
            "evaluation_epsilon": 0.0,
            "raw_reward": True,
            "requested_device": self.requested_device,
            "precision": self.precision,
            "sequential": self.sequential,
            "continuous_run": self.continuous_run,
            "require_cuda": self.require_cuda,
            "actual_transition_definition": (
                "global_step is accepted environment transitions, not vector "
                "iterations, optimizer updates, or raw Atari frames"
            ),
        }


def _validate_day20_selection(
    selection_path: Path,
    *,
    expected_manifest_path: Path,
) -> dict[str, Any]:
    report = _read_json(selection_path)
    if report.get("manifest_status") != "completed":
        raise ValueError("Day 20 selection report is not completed")
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Day 20 selection report is missing selection")
    if selection.get("status") != "complete":
        raise ValueError("Day 20 selection is not complete")
    if selection.get("final_training_family") != DAY21_WINNER_FAMILY_ID:
        raise ValueError(
            "Day 20 selected family does not match the frozen Day 21 winner"
        )
    if selection.get("deployment_candidate") != DAY21_WINNER_FAMILY_ID:
        raise ValueError("Day 20 deployment candidate is not the Day 21 winner")
    if report.get("selection_horizon") != 1_000_000:
        raise ValueError("Day 20 selection must come from the completed 1M extension")
    report_manifest = report.get("manifest")
    if isinstance(report_manifest, str):
        resolved_report_manifest = _resolve_reference(
            report_manifest,
            source=selection_path,
            repository_root=_repository_root(expected_manifest_path),
        )
        if resolved_report_manifest != expected_manifest_path.resolve():
            raise ValueError("Day 20 selection report points at an unexpected manifest")
    # The report is itself the frozen machine-readable decision.  Its embedded
    # manifest hash may be stale when the Day 20 runner rewrote only its
    # timestamp after report generation; retain both values as provenance but
    # do not discard the already-complete selection for that derived-only drift.
    current_manifest_sha256 = sha256_file(expected_manifest_path)
    hash_matches = report.get("manifest_sha256") == current_manifest_sha256
    selection_matches_current_manifest = hash_matches
    if not hash_matches:
        current_report = build_day20_report(
            expected_manifest_path,
            config=load_day20_config(
                "configs/comparisons/dqn-family/manifest.json",
                repository_root=_repository_root(expected_manifest_path),
                require_probe_states=False,
            ),
        )
        current_selection = current_report.get("selection")
        selection_matches_current_manifest = isinstance(current_selection, Mapping) and all(
            current_selection.get(field) == selection.get(field)
            for field in (
                "status",
                "final_training_family",
                "deployment_candidate",
                "winner_quality_mean",
                "runner_up_quality_mean",
                "selection_horizon",
            )
        )
        if not selection_matches_current_manifest:
            raise ValueError(
                "Day 20 selection report is stale and its selection differs from the current manifest"
            )
    report["day21_manifest_hash_check"] = {
        "reported_manifest_sha256": report.get("manifest_sha256"),
        "current_manifest_sha256": current_manifest_sha256,
        "match": hash_matches,
        "recomputed_selection_match": selection_matches_current_manifest,
    }
    return report


def load_day21_config(
    path: str | Path = "configs/final-training/manifest.json",
    *,
    repository_root: str | Path | None = None,
) -> Day21FinalTrainingConfig:
    """Load and validate the frozen Day 21 protocol from JSON."""

    source = Path(path).resolve()
    payload = _read_json(source)
    root = Path(repository_root).resolve() if repository_root else _repository_root(source)
    if payload.get("schema_version") != DAY21_SCHEMA_VERSION:
        raise ValueError("Day 21 config has an unsupported schema_version")
    if payload.get("day") != 21:
        raise ValueError("Day 21 config must have day=21")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Day 21 experiment_id must be a non-empty string")
    if experiment_id.strip() != DAY21_EXPERIMENT_ID:
        raise ValueError(f"Day 21 experiment_id must be {DAY21_EXPERIMENT_ID!r}")

    day20_config_path = _resolve_reference(
        _source_value(payload, "day20_config"),
        source=source,
        repository_root=root,
    )
    day20_selection_path = _resolve_reference(
        _source_value(payload, "day20_selection"),
        source=source,
        repository_root=root,
    )
    backend_manifest_path = _resolve_reference(
        _source_value(payload, "backend_manifest"),
        source=source,
        repository_root=root,
    )
    contract_path = _resolve_reference(
        _source_value(payload, "contract"),
        source=source,
        repository_root=root,
    )
    evaluation_config_path = _resolve_reference(
        _source_value(payload, "evaluation_config"),
        source=source,
        repository_root=root,
    )

    day20_config = load_day20_config(
        day20_config_path,
        repository_root=root,
        require_probe_states=False,
    )
    expected_day20_manifest = root / "experiments/day20-dqn-family/manifest.json"
    day20_selection = _validate_day20_selection(
        day20_selection_path,
        expected_manifest_path=expected_day20_manifest,
    )
    backend_manifest = load_day16_backend_manifest(
        backend_manifest_path,
        repository_root=root,
        verify_evidence_files=False,
    )
    trainer = backend_manifest.get("trainer")
    if not isinstance(trainer, Mapping) or not isinstance(trainer.get("config"), Mapping):
        raise ValueError("Day 16 backend manifest is missing trainer.config")
    backend_config = DQNConfig.from_dict(trainer["config"])
    required_backend = {
        "num_envs": 2,
        "replay_backend": "gpu",
        "replay_transfer": "direct",
        "device": "cuda",
        "precision": "float32",
        "cpu_threads": 2,
        "batch_size": 32,
        "replay_capacity": 10_000,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "train_frequency": 4,
        "learning_starts": 1000,
        "target_update_interval": 500,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 10_000,
        "gradient_clip_norm": 10.0,
        "reward_clip": True,
        "strict_action_selection_parity": True,
        "diagnostics_interval": 100,
        "metrics_flush_interval": 500,
    }
    frozen_backend_fields = payload.get("frozen_backend_fields")
    if isinstance(frozen_backend_fields, (str, bytes)) or not isinstance(
        frozen_backend_fields,
        Sequence,
    ):
        raise ValueError("frozen_backend_fields must be a sequence")
    expected_frozen_fields = set(required_backend).difference({"device"})
    if set(str(field) for field in frozen_backend_fields) != expected_frozen_fields:
        raise ValueError(
            "frozen_backend_fields must enumerate every frozen backend control"
        )
    observed_backend = backend_config.to_dict()
    backend_diffs = {
        field: {"expected": expected, "observed": observed_backend.get(field)}
        for field, expected in required_backend.items()
        if observed_backend.get(field) != expected
    }
    if backend_diffs:
        raise ValueError(f"Day 21 backend differs from Day 16 canonical controls: {backend_diffs}")

    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    evaluation_config = _read_json(evaluation_config_path)
    evaluation_seeds = _unique_ints(
        evaluation_config.get("seeds"),
        name="evaluation.seeds",
    )
    evaluation_episodes = _positive_int(
        evaluation_config.get("episodes_per_seed"),
        name="evaluation.episodes_per_seed",
    )
    evaluation_epsilon = _finite_float(
        evaluation_config.get("epsilon"),
        name="evaluation.epsilon",
    )
    if evaluation_seeds != DAY21_SELECTION_SEEDS:
        raise ValueError("Day 21 selection seeds must reuse [101, 202, 303]")
    if contract.concrete_episode_seeds != expand_concrete_episode_seeds(
        evaluation_seeds,
        episodes_per_seed=evaluation_episodes,
    ):
        raise ValueError("selection evaluation seeds do not match Contract v2")
    if evaluation_epsilon != 0.0 or evaluation_epsilon != contract.evaluation_epsilon:
        raise ValueError("Day 21 selection evaluation requires Contract v2 epsilon=0")

    winner = payload.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("winner must be an object")
    if winner.get("family_id") != DAY21_WINNER_FAMILY_ID:
        raise ValueError("winner.family_id must be dueling_double_dqn")
    if winner.get("algorithm") != DAY21_ALGORITHM:
        raise ValueError("winner.algorithm must be double_dqn")
    if winner.get("architecture") != DAY21_ARCHITECTURE:
        raise ValueError("winner.architecture must be dueling")
    if winner.get("hidden_dim") != 512:
        raise ValueError("winner.hidden_dim must remain the Day 20 512-unit model")
    if day20_selection["selection"].get("final_training_family") != winner.get("family_id"):
        raise ValueError("Day 21 winner disagrees with Day 20 selection report")

    raw_seeds = _unique_ints(payload.get("training_seeds"), name="training_seeds")
    if raw_seeds != DAY21_TRAINING_SEEDS:
        raise ValueError("Day 21 training_seeds must be [1011, 2022, 3033]")
    if not set(raw_seeds).isdisjoint(day20_config.training_seeds):
        raise ValueError("Day 21 training seeds must be fresh relative to Day 20")

    raw_targets = payload.get("milestones")
    if not isinstance(raw_targets, Mapping):
        raise ValueError("milestones must be an object")
    stage_targets = {
        stage: _positive_int(raw_targets.get(stage), name=f"milestones.{stage}")
        for stage in DAY21_STAGE_ORDER
    }
    if stage_targets != DAY21_STAGE_TARGETS:
        raise ValueError("Day 21 milestones must be 1M, 2.5M, and 5M")
    if tuple(stage_targets[stage] for stage in DAY21_STAGE_ORDER) != tuple(
        sorted(stage_targets.values())
    ):
        raise ValueError("Day 21 milestone targets must be increasing")

    selection_seeds, selection_episodes, _selection_epsilon, _ = _evaluation_settings(
        payload.get("selection_evaluation"),
        name="selection_evaluation",
        expected_seeds=DAY21_SELECTION_SEEDS,
    )
    holdout = payload.get("final_holdout")
    if not isinstance(holdout, Mapping):
        raise ValueError("final_holdout must be an object")
    holdout_seeds, holdout_episodes, _holdout_epsilon, _ = _evaluation_settings(
        {
            **dict(holdout),
            "seeds": holdout.get("group_seeds"),
            "episodes_per_seed": holdout.get("episodes_per_group"),
        },
        name="final_holdout",
    )
    if holdout_seeds != DAY21_HOLDOUT_GROUP_SEEDS:
        raise ValueError("final_holdout.group_seeds must be [404, 505, 606]")
    selection_concrete = expand_concrete_episode_seeds(
        selection_seeds,
        episodes_per_seed=selection_episodes,
    )
    holdout_concrete = expand_concrete_episode_seeds(
        holdout_seeds,
        episodes_per_seed=holdout_episodes,
    )
    if set(selection_concrete).intersection(holdout_concrete):
        raise ValueError("selection and final holdout concrete seeds must not overlap")

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object")
    requested_device = str(execution.get("requested_device", "")).strip().lower()
    if requested_device != "cuda":
        raise ValueError("Day 21 formal training must request exactly cuda")
    precision = str(execution.get("precision", "")).strip().lower()
    if precision not in {"float32", "fp32"}:
        raise ValueError("Day 21 formal training must use float32")
    if execution.get("sequential") is not True:
        raise ValueError("Day 21 formal runs must be sequential")
    if execution.get("continuous_run") is not True:
        raise ValueError("Day 21 must prefer continuous in-process milestone runs")
    if execution.get("require_cuda") is not True:
        raise ValueError("Day 21 formal training must require CUDA")
    stage_c_policy = execution.get("stage_c_policy")
    if not isinstance(stage_c_policy, Mapping):
        raise ValueError("execution.stage_c_policy must be an object")
    primary_trigger = stage_c_policy.get("primary_trigger")
    if not isinstance(primary_trigger, str) or not primary_trigger.strip():
        raise ValueError("execution.stage_c_policy.primary_trigger must be non-empty")
    trigger_evidence = stage_c_policy.get("trigger_evidence")
    if not isinstance(trigger_evidence, Mapping):
        raise ValueError("execution.stage_c_policy.trigger_evidence must be an object")
    if trigger_evidence.get("training_seed") != 2022:
        raise ValueError(
            "execution.stage_c_policy.trigger_evidence.training_seed must be 2022"
        )
    one_million_mean = _finite_float(
        trigger_evidence.get("stage_a_1m_mean_return"),
        name="execution.stage_c_policy.trigger_evidence.stage_a_1m_mean_return",
    )
    two_point_five_million_mean = _finite_float(
        trigger_evidence.get("stage_b_2_5m_mean_return"),
        name="execution.stage_c_policy.trigger_evidence.stage_b_2_5m_mean_return",
    )
    improvement = _finite_float(
        trigger_evidence.get("mean_return_improvement"),
        name="execution.stage_c_policy.trigger_evidence.mean_return_improvement",
    )
    if not math.isclose(
        improvement,
        two_point_five_million_mean - one_million_mean,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "execution.stage_c_policy.trigger_evidence.mean_return_improvement "
            "must equal the 1M-to-2.5M improvement"
        )
    if not isinstance(stage_c_policy.get("user_requested_5m"), bool):
        raise ValueError("execution.stage_c_policy.user_requested_5m must be boolean")
    if stage_c_policy.get("request_is_supplemental_provenance") is not True:
        raise ValueError(
            "execution.stage_c_policy.request_is_supplemental_provenance must be true"
        )
    if holdout.get("evaluation_order") != "selection → final freeze → final holdout":
        raise ValueError(
            "final_holdout.evaluation_order must preserve selection → final freeze → final holdout"
        )
    headroom = _positive_int(
        execution.get("cuda_headroom_bytes", DAY21_DEFAULT_CUDA_HEADROOM_BYTES),
        name="execution.cuda_headroom_bytes",
    )
    selection_rule = payload.get("selection_rule")
    health_rule = payload.get("health_rule")
    if not isinstance(selection_rule, Mapping) or not isinstance(health_rule, Mapping):
        raise ValueError("selection_rule and health_rule must be objects")
    return Day21FinalTrainingConfig(
        source_path=source,
        repository_root=root,
        experiment_id=experiment_id.strip(),
        day20_config_path=day20_config_path,
        day20_selection_path=day20_selection_path,
        backend_manifest_path=backend_manifest_path,
        contract_path=contract_path,
        evaluation_config_path=evaluation_config_path,
        day20_selection=day20_selection,
        backend_manifest=backend_manifest,
        contract=contract,
        evaluation_config={
            **evaluation_config,
            "seeds": list(evaluation_seeds),
            "episodes_per_seed": evaluation_episodes,
            "epsilon": evaluation_epsilon,
        },
        backend_config=backend_config,
        winner_family_id=DAY21_WINNER_FAMILY_ID,
        algorithm=DAY21_ALGORITHM,
        architecture=DAY21_ARCHITECTURE,
        training_seeds=raw_seeds,
        day20_training_seeds=tuple(day20_config.training_seeds),
        stage_targets=stage_targets,
        selection_seeds=selection_seeds,
        selection_episodes_per_seed=selection_episodes,
        holdout_group_seeds=holdout_seeds,
        holdout_episodes_per_group=holdout_episodes,
        requested_device=requested_device,
        precision="float32",
        sequential=True,
        continuous_run=True,
        require_cuda=True,
        cuda_headroom_bytes=headroom,
        selection_rule=dict(selection_rule),
        health_rule=dict(health_rule),
        raw=payload,
    )


def build_day21_manifest(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: str | Path,
    runs_root: str | Path = "runs",
    evaluations_root: str | Path = "evaluations",
    evidence_root: str | Path = "assets/day21",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create the planned runtime manifest before starting any CUDA work."""

    destination = Path(manifest_path).resolve()
    root = config.repository_root
    runs_base = Path(runs_root).resolve() / config.experiment_id
    evaluations_base = Path(evaluations_root).resolve() / config.experiment_id
    evidence_base = Path(evidence_root).resolve()
    entries: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        run_dir = runs_base / f"seed{seed}"
        stages: dict[str, dict[str, Any]] = {}
        for stage, target in config.stage_targets.items():
            evaluation_dir = evaluations_base / f"seed{seed}" / f"step-{target:08d}"
            compact_metrics = (
                evidence_base
                / "evidence-runs"
                / f"seed{seed}"
                / f"metrics-{stage}.csv"
            )
            stage_summary = evidence_base / "evidence-runs" / f"seed{seed}" / f"summary-{stage}.json"
            stages[stage] = {
                "target_transitions": target,
                "status": "pending",
                "eligible": False,
                "checkpoint": None,
                "training": {
                    "metrics_path": manifest_reference(
                        run_dir / "metrics.csv",
                        start=destination.parent,
                    ),
                    "compact_metrics_path": manifest_reference(
                        compact_metrics,
                        start=destination.parent,
                    ),
                    "summary_path": manifest_reference(
                        stage_summary,
                        start=destination.parent,
                    ),
                    "summary": None,
                    "runtime": None,
                    "resume_exact": None,
                },
                "health": None,
                "evaluation": {
                    "directory": manifest_reference(
                        evaluation_dir,
                        start=destination.parent,
                    ),
                    "results": None,
                    "episodes": None,
                    "summary": None,
                    "status": "locked" if stage != "stage_a_1m" else "pending",
                    "phase": "selection",
                    "seeds": list(config.selection_seeds),
                    "concrete_episode_seeds": list(config.selection_concrete_seeds),
                    "episodes_per_seed": config.selection_episodes_per_seed,
                },
                "error": None,
            }
        entries.append(
            {
                "run_id": f"{config.experiment_id}-seed{seed}",
                "training_seed": seed,
                "algorithm": config.algorithm,
                "architecture": config.architecture,
                "run_dir": manifest_reference(run_dir, start=destination.parent),
                "status": "pending",
                "fresh_training_seed": True,
                "stages": stages,
            }
        )
    source_of_truth = {
        "config": _artifact_reference(config.source_path, start=root),
        "day20_config": _artifact_reference(config.day20_config_path, start=root),
        "day20_selection": _artifact_reference(config.day20_selection_path, start=root),
        "backend_manifest": _artifact_reference(config.backend_manifest_path, start=root),
        "contract": {
            **_artifact_reference(config.contract_path, start=root),
            **config.contract_provenance,
        },
        "evaluation_config": _artifact_reference(config.evaluation_config_path, start=root),
    }
    return {
        "schema_version": DAY21_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "status": "planned",
        "source_of_truth": source_of_truth,
        "protocol": config.protocol(),
        "winner": {
            "family_id": config.winner_family_id,
            "algorithm": config.algorithm,
            "architecture": config.architecture,
            "day20_selection": config.day20_selection.get("selection"),
        },
        "selection_rule": dict(config.selection_rule),
        "health_rule": dict(config.health_rule),
        "selection_decisions": {
            "stage_a_1m": None,
            "stage_b_2_5m": None,
            "stage_c_5m": None,
            "final_checkpoint": None,
        },
        "final_holdout": {
            "status": "locked",
            "group_seeds": list(config.holdout_group_seeds),
            "concrete_episode_seeds": list(config.holdout_concrete_seeds),
            "episodes_per_group": config.holdout_episodes_per_group,
            "seed_scope": (
                "independent holdout seeds under the same Contract v2 semantics; "
                "does not modify the Contract v2 canonical selection seed list"
            ),
            "results": None,
            "episodes": None,
            "summary": None,
            "opened_after_final_freeze": False,
        },
        "canonical_final_model": None,
        "runs": entries,
        "artifacts": {
            "evidence_root": relative_path(evidence_base, start=root),
            "report_json": "assets/day21/final-training-report.json",
            "report_markdown": "reports/day21-final-long-training.md",
        },
        "command": list(command) if command is not None else None,
    }


def read_day21_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != DAY21_SCHEMA_VERSION:
        raise ValueError("unsupported Day 21 manifest schema_version")
    if payload.get("experiment_id") != DAY21_EXPERIMENT_ID:
        raise ValueError("unexpected Day 21 experiment_id")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Day 21 manifest runs must be a non-empty array")
    seen: set[int] = set()
    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            raise ValueError(f"runs[{index}] must be an object")
        seed = _positive_int(entry.get("training_seed"), name=f"runs[{index}].training_seed")
        if seed in seen:
            raise ValueError(f"duplicate Day 21 training seed: {seed}")
        seen.add(seed)
        stages = entry.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != set(DAY21_STAGE_TARGETS):
            raise ValueError(f"runs[{index}].stages do not match Day 21 milestones")
        for stage, target in DAY21_STAGE_TARGETS.items():
            record = stages[stage]
            if not isinstance(record, Mapping) or int(record.get("target_transitions", -1)) != target:
                raise ValueError(f"runs[{index}].{stage} target is invalid")
    return payload


def validate_day21_manifest(
    path: str | Path,
    *,
    config: Day21FinalTrainingConfig,
) -> dict[str, Any]:
    payload = read_day21_manifest(path)
    if payload.get("protocol") != config.protocol():
        raise ValueError("existing Day 21 manifest protocol does not match config")
    sources = payload.get("source_of_truth")
    if not isinstance(sources, Mapping):
        raise ValueError("existing Day 21 manifest is missing source_of_truth")
    expected = {
        "config": config.source_path,
        "day20_config": config.day20_config_path,
        "day20_selection": config.day20_selection_path,
        "backend_manifest": config.backend_manifest_path,
        "contract": config.contract_path,
        "evaluation_config": config.evaluation_config_path,
    }
    for name, source in expected.items():
        reference = sources.get(name)
        if not isinstance(reference, Mapping) or reference.get("sha256") != sha256_file(source):
            raise ValueError(f"existing Day 21 manifest {name} hash does not match config")
    return payload


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _record_summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = record.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("summary"), Mapping):
        return evaluation["summary"]
    summary = record.get("evaluation_summary")
    return summary if isinstance(summary, Mapping) else {}


def _record_is_healthy(record: Mapping[str, Any]) -> bool:
    health = record.get("health")
    return isinstance(health, Mapping) and health.get("healthy") is True


def _record_quality(record: Mapping[str, Any]) -> tuple[float, float, float, int]:
    summary = _record_summary(record)
    mean = _number(summary.get("mean_return"))
    median = _number(summary.get("median_return"))
    spread = _number(summary.get("std_return"))
    count = summary.get("count")
    if mean is None or median is None or spread is None:
        raise ValueError("selection record must contain finite mean, median, and std return")
    if not isinstance(count, int) or count < 1:
        raise ValueError("selection record must contain a positive evaluation count")
    return mean, median, spread, count


def select_extension_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    limit: int = DAY21_MAX_STAGE_B_CANDIDATES,
) -> list[dict[str, Any]]:
    """Select candidates by aggregate fixed-evaluation quality only.

    The per-episode maximum is intentionally not read.  Ties use the
    aggregate median, episode spread, then training seed for determinism.
    """

    parsed_limit = _positive_int(limit, name="limit")
    eligible: list[dict[str, Any]] = []
    for record in records:
        if not _record_is_healthy(record):
            continue
        _record_quality(record)
        eligible.append(dict(record))
    eligible.sort(
        key=lambda item: (
            -_record_quality(item)[0],
            -_record_quality(item)[1],
            _record_quality(item)[2],
            int(item.get("training_seed", 0)),
        )
    )
    return eligible[:parsed_limit]


def select_final_checkpoint(
    candidates: Iterable[Mapping[str, Any]],
    *,
    near_equal_absolute_gap: float = 1.0,
) -> dict[str, Any]:
    """Choose one healthy checkpoint using the frozen final-selection rule."""

    tolerance = _finite_float(
        near_equal_absolute_gap,
        name="near_equal_absolute_gap",
    )
    if tolerance < 0.0:
        raise ValueError("near_equal_absolute_gap must not be negative")
    eligible = [
        dict(candidate)
        for candidate in candidates
        if _record_is_healthy(candidate)
    ]
    if not eligible:
        raise ValueError("no healthy final-training checkpoint candidates are available")
    for candidate in eligible:
        _record_quality(candidate)
    best_mean = max(_record_quality(item)[0] for item in eligible)
    near_equal = [
        item
        for item in eligible
        if best_mean - _record_quality(item)[0] <= tolerance
    ]
    stage_index = {stage: index for index, stage in enumerate(DAY21_STAGE_ORDER)}
    selected = min(
        near_equal,
        key=lambda item: (
            _record_quality(item)[2],
            stage_index.get(str(item.get("stage")), len(stage_index)),
            int(item.get("target_transitions", 0)),
            int(item.get("training_seed", 0)),
        ),
    )
    return dict(selected)


def _health_rule_value(
    health_rule: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = health_rule.get(name, default)
    return default if value is None else value


def assess_training_health(
    summary: Mapping[str, Any],
    metrics_path: str | Path,
    *,
    expected_transitions: int,
    contract_id: str,
    health_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the hard correctness and numerical health gates for a run."""

    failures: list[str] = []
    runtime = summary.get("runtime")
    if not isinstance(runtime, Mapping):
        runtime = {}
        failures.append("runtime metadata is missing")
    if summary.get("status") != "completed":
        failures.append(f"summary status is {summary.get('status')!r}, expected 'completed'")
    try:
        observed_transitions = int(summary.get("total_transitions", -1))
    except (TypeError, ValueError):
        observed_transitions = -1
    if observed_transitions != int(expected_transitions):
        failures.append(
            f"transition count is {observed_transitions}, expected {int(expected_transitions)}"
        )
    for field in ("total_steps", "training_steps", "physical_environment_steps"):
        try:
            observed_value = int(summary.get(field, -1))
        except (TypeError, ValueError):
            observed_value = -1
        if observed_value != int(expected_transitions):
            failures.append(
                f"{field} is {observed_value}, expected {int(expected_transitions)}"
            )
    required_runtime = health_rule.get("required_runtime")
    if not isinstance(required_runtime, Mapping):
        required_runtime = {}
    required_requested_device = str(
        required_runtime.get("requested_device", "cuda")
    )
    resolved_prefix = str(
        required_runtime.get("resolved_device_prefix", "cuda:")
    )
    required_cuda_available = required_runtime.get("cuda_available", True)
    if runtime.get("requested_device") != required_requested_device:
        failures.append("requested device is not cuda")
    if not str(runtime.get("resolved_device", "")).startswith(resolved_prefix):
        failures.append("resolved device is not cuda:<index>")
    if runtime.get("cuda_available") is not required_cuda_available:
        failures.append("CUDA runtime is unavailable")
    required_contract_id = str(
        required_runtime.get("contract_id", contract_id)
    )
    if (
        summary.get("contract_id") != required_contract_id
        and runtime.get("contract_id") != required_contract_id
    ):
        failures.append("Contract v2 id is missing or mismatched")

    path = Path(metrics_path)
    metric_fields = tuple(
        str(field)
        for field in _health_rule_value(
            health_rule,
            "required_metric_fields",
            DAY21_REQUIRED_TRAINING_METRICS,
        )
    )
    metric_rows = 0
    non_finite_values = 0
    invalid_values = 0
    maximum_abs_value = _finite_float(
        _health_rule_value(health_rule, "max_abs_metric_value", 1000.0),
        name="health_rule.max_abs_metric_value",
    )
    observed_max_abs = 0.0
    missing_fields: list[str] = []
    if not path.is_file():
        failures.append(f"metrics file is missing: {path}")
    else:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing_fields = [field for field in metric_fields if field not in fields]
            if missing_fields:
                failures.append(
                    "metrics file is missing required fields: " + ", ".join(missing_fields)
                )
            for row in reader:
                metric_rows += 1
                for field in metric_fields:
                    raw_value = row.get(field)
                    if raw_value in (None, ""):
                        continue
                    try:
                        parsed = float(raw_value)
                    except (TypeError, ValueError):
                        invalid_values += 1
                        continue
                    if not math.isfinite(parsed):
                        non_finite_values += 1
                        continue
                    observed_max_abs = max(observed_max_abs, abs(parsed))
                    if abs(parsed) > maximum_abs_value:
                        failures.append(
                            f"{field} exceeds health bound {maximum_abs_value}: {parsed}"
                        )
        if metric_rows == 0:
            failures.append("metrics file contains no rows")
    if non_finite_values:
        failures.append(f"metrics contain {non_finite_values} non-finite values")
    if invalid_values:
        failures.append(f"metrics contain {invalid_values} non-numeric values")

    distribution = summary.get("action_distribution")
    if not isinstance(distribution, Mapping):
        distribution = {}
        failures.append("executed action distribution is missing")
    counts: list[float] = []
    for value in distribution.values():
        parsed = _number(value)
        if parsed is not None and parsed >= 0.0:
            counts.append(parsed)
    total_actions = sum(counts)
    distinct_actions = sum(count > 0.0 for count in counts)
    min_distinct = _positive_int(
        _health_rule_value(health_rule, "min_distinct_executed_actions", 2),
        name="health_rule.min_distinct_executed_actions",
    )
    if distinct_actions < min_distinct:
        failures.append(
            f"executed action coverage is {distinct_actions}, expected at least {min_distinct}"
        )
    dominant_ratio = max(counts) / total_actions if counts and total_actions else 1.0
    max_dominant_ratio = _finite_float(
        _health_rule_value(health_rule, "max_dominant_action_ratio", 0.995),
        name="health_rule.max_dominant_action_ratio",
    )
    if dominant_ratio > max_dominant_ratio:
        failures.append(
            f"dominant executed action ratio is {dominant_ratio:.6f}, "
            f"above {max_dominant_ratio:.6f}"
        )

    unique_failures = list(dict.fromkeys(failures))
    return {
        "healthy": not unique_failures,
        "failures": unique_failures,
        "checks": {
            "expected_transitions": int(expected_transitions),
            "observed_transitions": observed_transitions,
            "requested_device": runtime.get("requested_device"),
            "resolved_device": runtime.get("resolved_device"),
            "cuda_available": runtime.get("cuda_available"),
            "contract_id": summary.get("contract_id", runtime.get("contract_id")),
            "metric_rows": metric_rows,
            "required_metric_fields": list(metric_fields),
            "missing_metric_fields": missing_fields,
            "non_finite_metric_values": non_finite_values,
            "invalid_metric_values": invalid_values,
            "observed_max_abs_metric_value": observed_max_abs,
            "distinct_executed_actions": distinct_actions,
            "dominant_executed_action_ratio": dominant_ratio,
        },
        "rule": dict(health_rule),
    }


def assess_evaluation_contract_health(
    result: Mapping[str, Any],
    *,
    contract_id: str,
    expected_seeds: Sequence[int],
    episodes_per_seed: int,
    expected_concrete_seeds: Sequence[int],
    requested_device: str = "cuda",
) -> dict[str, Any]:
    """Validate evaluation rows against Contract v2 serve/TimeLimit semantics."""

    failures: list[str] = []
    parsed_seeds = tuple(int(seed) for seed in expected_seeds)
    expected_rows = {
        (seed, episode_index)
        for seed in parsed_seeds
        for episode_index in range(1, int(episodes_per_seed) + 1)
    }
    expected_episode_seeds = {int(seed) for seed in expected_concrete_seeds}
    rows = result.get("per_episode")
    if not isinstance(rows, list) or not rows:
        rows = []
        failures.append("evaluation contains no per-episode rows")
    identities: set[tuple[int, int]] = set()
    observed_episode_seeds: set[int] = set()
    invalid_time_limit_rows = 0
    invalid_action_rows = 0
    invalid_fire_reasons = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"per_episode[{index}] is not an object")
            continue
        try:
            seed = int(row["evaluation_seed"])
            episode_index = int(row["episode_index"])
            episode_seed = int(row["episode_seed"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"per_episode[{index}] has invalid seed identity")
            continue
        identity = (seed, episode_index)
        if identity in identities:
            failures.append(f"duplicate evaluation episode identity {identity}")
        identities.add(identity)
        observed_episode_seeds.add(episode_seed)
        if identity not in expected_rows:
            failures.append(f"unexpected evaluation episode identity {identity}")
        if episode_seed != seed + episode_index - 1:
            failures.append(f"episode seed expansion is invalid for {identity}")
        terminated = row.get("terminated")
        truncated = row.get("truncated")
        time_limit = row.get("time_limit")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            failures.append(f"termination flags are invalid for {identity}")
        elif terminated and truncated:
            failures.append(f"terminated and truncated are both true for {identity}")
        if not isinstance(time_limit, bool):
            failures.append(f"time_limit flag is invalid for {identity}")
        elif time_limit != bool(truncated):
            invalid_time_limit_rows += 1
        if row.get("action_distribution_semantics") != ACTION_DISTRIBUTION_SEMANTICS:
            invalid_action_rows += 1
        if not isinstance(row.get("requested_action_distribution"), Mapping):
            invalid_action_rows += 1
        if not isinstance(row.get("executed_action_distribution"), Mapping):
            invalid_action_rows += 1
        reason_counts = row.get("auto_fire_reason_counts")
        if isinstance(reason_counts, Mapping):
            unexpected_reasons = set(reason_counts).difference(
                {"initial_serve", "after_life_loss"}
            )
            invalid_fire_reasons += len(unexpected_reasons)

    if identities != expected_rows:
        failures.append("evaluation episode identities do not match the predeclared seed groups")
    if observed_episode_seeds != expected_episode_seeds:
        failures.append("evaluation concrete episode seeds do not match the predeclared groups")
    if invalid_time_limit_rows:
        failures.append(
            f"{invalid_time_limit_rows} rows violate the Contract v2 TimeLimit/truncated relationship"
        )
    if invalid_action_rows:
        failures.append(
            f"{invalid_action_rows} rows violate executed/requested action provenance"
        )
    if invalid_fire_reasons:
        failures.append("evaluation contains an unknown environment-owned FIRE reason")

    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
        failures.append("evaluation summary is missing")
    expected_count = len(expected_rows)
    try:
        summary_count = int(summary.get("count", -1))
        complete_count = int(summary.get("complete_episodes", -1))
        finished_count = int(summary.get("finished_episode_count", -1))
        terminated_count = int(summary.get("terminated_count", -1))
        truncated_count = int(summary.get("truncated_count", -1))
        time_limit_count = int(summary.get("time_limit_truncated_count", -1))
    except (TypeError, ValueError):
        summary_count = complete_count = finished_count = -1
        terminated_count = truncated_count = time_limit_count = -1
    if summary_count != expected_count:
        failures.append(f"evaluation summary count is {summary_count}, expected {expected_count}")
    if complete_count != expected_count or finished_count != expected_count:
        failures.append("not every evaluation episode finished under Contract v2")
    if terminated_count + truncated_count != expected_count:
        failures.append("evaluation termination totals do not cover every episode")
    if time_limit_count != truncated_count:
        failures.append("truncated episodes are not all identified as Contract v2 TimeLimit episodes")
    truncation_rate = _number(summary.get("truncation_rate"))
    if truncation_rate is None or not math.isclose(
        truncation_rate,
        truncated_count / expected_count if expected_count else 0.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        failures.append("evaluation truncation_rate is inconsistent with episode rows")

    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        runtime = {}
        failures.append("evaluation runtime metadata is missing")
    if runtime.get("requested_device") != requested_device:
        failures.append("evaluation requested device is not the formal CUDA request")
    if not str(runtime.get("resolved_device", "")).startswith("cuda:"):
        failures.append("evaluation resolved device is not cuda:<index>")
    if result.get("evaluation_epsilon") != 0.0:
        failures.append("evaluation epsilon is not zero")
    for field_name in ("training", "checkpoint"):
        provenance = result.get(field_name)
        if isinstance(provenance, Mapping):
            observed_contract = provenance.get("contract_id")
            if observed_contract is not None and observed_contract != contract_id:
                failures.append(f"{field_name} Contract v2 id does not match")

    return {
        "healthy": not list(dict.fromkeys(failures)),
        "failures": list(dict.fromkeys(failures)),
        "checks": {
            "expected_episode_count": expected_count,
            "observed_episode_count": len(rows),
            "expected_concrete_episode_seeds": sorted(expected_episode_seeds),
            "observed_concrete_episode_seeds": sorted(observed_episode_seeds),
            "action_distribution_semantics": result.get("action_distribution_semantics"),
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "time_limit_truncated_count": time_limit_count,
            "requested_device": runtime.get("requested_device"),
            "resolved_device": runtime.get("resolved_device"),
        },
        "contract_id": contract_id,
    }


def compact_metrics(
    source: str | Path,
    destination: str | Path,
    *,
    sampling_interval: int = 5_000,
    max_global_step: int | None = None,
) -> dict[str, Any]:
    """Keep immutable episode/diagnostic evidence for one transition horizon."""

    source_path = Path(source)
    destination_path = Path(destination)
    interval = _positive_int(sampling_interval, name="sampling_interval")
    limit = (
        None
        if max_global_step is None
        else _positive_int(max_global_step, name="max_global_step")
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with source_path.open("r", newline="", encoding="utf-8") as input_stream:
        reader = csv.DictReader(input_stream)
        fieldnames = list(reader.fieldnames or ())
        with destination_path.open("w", newline="", encoding="utf-8") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                try:
                    step = int(float(row.get("global_step", "-1")))
                except (TypeError, ValueError):
                    step = -1
                if limit is not None and (step < 0 or step > limit):
                    continue
                keep = step >= 0 and step % interval == 0
                keep = keep or any(
                    row.get(field) not in (None, "")
                    for field in ("raw_episode_return", "loss", "q_mean")
                )
                if keep:
                    writer.writerow(row)
                    kept += 1
    return {
        "path": destination_path,
        "source": source_path,
        "sampling_interval": interval,
        "max_global_step": limit,
        "source_rows": total,
        "kept_rows": kept,
        "sha256": sha256_file(destination_path),
    }


def _compact_training_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "status",
        "trainer",
        "run_id",
        "seed",
        "algorithm",
        "architecture",
        "num_envs",
        "model_config",
        "contract_id",
        "contract_path",
        "total_steps",
        "total_transitions",
        "training_steps",
        "physical_environment_steps",
        "vector_iterations",
        "episodes",
        "optimizer_updates",
        "target_sync_count",
        "last_target_sync_step",
        "replay_backend",
        "replay_transfer",
        "replay_size",
        "replay_occupancy",
        "steps_per_second",
        "environment_transitions_per_second",
        "stage_start_step",
        "stage_counters",
        "stage_rates",
        "stage_training_steps",
        "stage_physical_environment_steps",
        "last_loss",
        "last_q_mean",
        "last_q_max",
        "last_target_mean",
        "last_td_error_mean_abs",
        "last_gradient_norm",
        "action_distribution",
        "random_decision_ratio",
        "last_checkpoint",
        "resume_provenance",
        "runtime",
        "metadata",
        "environment_contract",
    }
    return {key: summary[key] for key in keep if key in summary}


def build_day21_report(
    manifest_path: str | Path,
    *,
    config: Day21FinalTrainingConfig | None = None,
) -> dict[str, Any]:
    """Build a source-backed report from the current runtime manifest."""

    source = Path(manifest_path).resolve()
    manifest = read_day21_manifest(source)
    if config is None:
        config = load_day21_config()
    stage_rows: list[dict[str, Any]] = []
    for entry in manifest["runs"]:
        if not isinstance(entry, Mapping):
            continue
        for stage in DAY21_STAGE_ORDER:
            record = entry["stages"][stage]
            if not isinstance(record, Mapping):
                continue
            training = record.get("training", {})
            evaluation = record.get("evaluation", {})
            stage_rows.append(
                {
                    "run_id": entry.get("run_id"),
                    "training_seed": entry.get("training_seed"),
                    "stage": stage,
                    "target_transitions": record.get("target_transitions"),
                    "status": record.get("status"),
                    "eligible": record.get("eligible"),
                    "health": record.get("health"),
                    "checkpoint": record.get("checkpoint"),
                    "gameplay": record.get("gameplay"),
                    "training": {
                        "summary": (
                            _compact_training_summary(training["summary"])
                            if isinstance(training, Mapping)
                            and isinstance(training.get("summary"), Mapping)
                            else None
                        ),
                        "runtime": training.get("runtime")
                        if isinstance(training, Mapping)
                        else None,
                        "metrics_path": training.get("compact_metrics_path")
                        if isinstance(training, Mapping)
                        else None,
                    },
                    "evaluation": evaluation,
                }
            )
    selection_decisions = manifest.get("selection_decisions")
    stage_c_decision = (
        selection_decisions.get("stage_c_5m")
        if isinstance(selection_decisions, Mapping)
        else None
    )
    stage_c_provenance = (
        {
            key: stage_c_decision[key]
            for key in (
                "primary_trigger",
                "trigger_evidence",
                "user_requested_5m",
                "request_is_supplemental_provenance",
            )
            if key in stage_c_decision
        }
        if isinstance(stage_c_decision, Mapping)
        else None
    )
    return {
        "schema_version": DAY21_SCHEMA_VERSION,
        "generated_at_utc": utc_timestamp(),
        "experiment_id": manifest["experiment_id"],
        "status": manifest.get("status"),
        "source_manifest": relative_path(source, start=config.repository_root),
        "source_manifest_sha256": sha256_file(source),
        "source_of_truth": manifest.get("source_of_truth"),
        "protocol": manifest.get("protocol"),
        "winner": manifest.get("winner"),
        "day20_selection": config.day20_selection.get("selection"),
        "selection_rule": manifest.get("selection_rule"),
        "health_rule": manifest.get("health_rule"),
        "selection_decisions": selection_decisions,
        "stage_c_provenance": stage_c_provenance,
        "training": stage_rows,
        "canonical_final_model": manifest.get("canonical_final_model"),
        "final_holdout": manifest.get("final_holdout"),
        "artifacts": manifest.get("artifacts"),
        "limitations": [
            "Training quality is interpreted only under Contract v2 and its raw-reward evaluation semantics.",
            "A continuous in-process run preserves Replay state; a crash resume is explicitly non-exact when Replay was not serialized.",
            "Stage C was justified by substantial 2.5M selection improvement; user_requested_5m is supplemental run-horizon provenance, while final checkpoint selection remains gate-driven.",
        ],
    }


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_day21_markdown(report: Mapping[str, Any]) -> str:
    """Render the engineering report without inventing unavailable values."""

    protocol = report.get("protocol", {})
    final_model = report.get("canonical_final_model")
    holdout = report.get("final_holdout", {})
    decisions = report.get("selection_decisions", {})
    stage_c_provenance = report.get("stage_c_provenance")
    final_checkpoint = (
        decisions.get("final_checkpoint", {})
        if isinstance(decisions, Mapping)
        else {}
    )
    selected_final = (
        final_checkpoint.get("selected", {})
        if isinstance(final_checkpoint, Mapping)
        else {}
    )
    lines = [
        "# Day 21 Final Long Training",
        "",
        "這份 report 記錄 Day 20 winner 在新的 training seeds 上如何經過 1M、2.5M 與 5M actual environment transitions 的 gate，並在最後 freeze 一個 canonical model。",
        "",
        f"- manifest status: `{report.get('status')}`",
        f"- winner: `{protocol.get('winner_family_id')}` ({protocol.get('algorithm')} / {protocol.get('architecture')})",
        f"- training seeds: `{protocol.get('training_seeds')}`",
        f"- stage targets: `{protocol.get('stage_targets')}` actual environment transitions",
        f"- selection evaluation: `{protocol.get('selection_concrete_seeds')}` concrete seeds, `{protocol.get('selection_episodes_per_seed')}` episodes per group",
        f"- final holdout: `{protocol.get('holdout_concrete_seeds')}` concrete seeds, opened after freeze: `{holdout.get('opened_after_final_freeze') if isinstance(holdout, Mapping) else None}`",
        f"- evaluation order: `{holdout.get('evaluation_order') if isinstance(holdout, Mapping) else None}`",
        "",
        "## Stage C trigger",
        "",
    ]
    if isinstance(stage_c_provenance, Mapping):
        lines.extend(
            [
                f"- primary trigger: `{stage_c_provenance.get('primary_trigger')}`",
                f"- evidence: `{stage_c_provenance.get('trigger_evidence')}`",
                f"- user_requested_5m (supplemental): `{stage_c_provenance.get('user_requested_5m')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Stage evidence",
            "",
            "| seed | stage | run id | target transitions | status | eligible | mean selection return | std return |",
            "| ---: | --- | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    training = report.get("training", [])
    if isinstance(training, list):
        for row in training:
            if not isinstance(row, Mapping):
                continue
            evaluation = row.get("evaluation")
            summary = evaluation.get("summary", {}) if isinstance(evaluation, Mapping) else {}
            lines.append(
                "| {seed} | {stage} | {run_id} | {target} | {status} | {eligible} | {mean} | {std} |".format(
                    seed=row.get("training_seed"),
                    stage=row.get("stage"),
                    run_id=row.get("run_id"),
                    target=row.get("target_transitions"),
                    status=row.get("status"),
                    eligible=row.get("eligible"),
                    mean=_display(summary.get("mean_return") if isinstance(summary, Mapping) else None),
                    std=_display(summary.get("std_return") if isinstance(summary, Mapping) else None),
                )
            )
    lines.extend(
        [
            "",
            "## Runtime and provenance",
            "",
            "| run id | seed | stage | resolved device | GPU | transitions | Contract v2 |",
            "| --- | ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    if isinstance(training, list):
        for row in training:
            if not isinstance(row, Mapping):
                continue
            training_meta = row.get("training")
            runtime = training_meta.get("runtime", {}) if isinstance(training_meta, Mapping) else {}
            summary = training_meta.get("summary", {}) if isinstance(training_meta, Mapping) else {}
            if not isinstance(runtime, Mapping) or not isinstance(summary, Mapping):
                continue
            lines.append(
                "| {run_id} | {seed} | {stage} | {device} | {gpu} | {transitions} | {contract} |".format(
                    run_id=row.get("run_id"),
                    seed=row.get("training_seed"),
                    stage=row.get("stage"),
                    device=runtime.get("resolved_device"),
                    gpu=runtime.get("gpu_model") or runtime.get("cuda_device_name"),
                    transitions=summary.get("total_transitions"),
                    contract=runtime.get("contract_id"),
                )
            )
    contract_source = report.get("source_of_truth", {}).get("contract", {})
    if isinstance(contract_source, Mapping):
        lines.append(
            f"- Contract v2 artifact: `{contract_source.get('path')}`, SHA256 `{contract_source.get('sha256')}`."
        )
    lines.extend(
        [
            "",
            "## Selection decisions",
            "",
            "```json",
            json.dumps(report.get("selection_decisions"), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Canonical final model",
            "",
        ]
    )
    if isinstance(final_model, Mapping):
        lines.extend(
            [
                f"- model: `{final_model.get('model_path')}`",
                f"- model SHA256: `{final_model.get('model_sha256')}`",
                f"- source checkpoint: `{final_model.get('source_checkpoint', {}).get('path') if isinstance(final_model.get('source_checkpoint'), Mapping) else None}`",
                f"- selected stage/seed: `{selected_final.get('stage')}` / `{selected_final.get('training_seed')}`",
                f"- holdout status: `{holdout.get('status') if isinstance(holdout, Mapping) else None}`",
            ]
        )
    else:
        lines.append("Final model has not been frozen.")
    holdout_summary = holdout.get("summary") if isinstance(holdout, Mapping) else None
    if isinstance(holdout_summary, Mapping):
        lines.extend(
            [
                "",
                "## Final holdout summary",
                "",
                f"- episodes: `{holdout_summary.get('count')}` complete; mean raw return `{_display(holdout_summary.get('mean_return'))}`; std `{_display(holdout_summary.get('std_return'))}`",
                f"- terminated: `{holdout_summary.get('terminated_count')}`; truncated: `{holdout_summary.get('truncated_count')}`; time-limit truncated: `{holdout_summary.get('time_limit_truncated_count')}`",
                f"- Contract v2 health gate: `{holdout.get('contract_health', {}).get('healthy') if isinstance(holdout.get('contract_health'), Mapping) else 'unavailable'}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = report.get("limitations", [])
    if isinstance(limitations, list):
        lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DAY21_ALGORITHM",
    "DAY21_ARCHITECTURE",
    "DAY21_DEFAULT_CUDA_HEADROOM_BYTES",
    "DAY21_EPISODES_PER_GROUP",
    "DAY21_EXPERIMENT_ID",
    "DAY21_HOLDOUT_GROUP_SEEDS",
    "DAY21_MAX_STAGE_B_CANDIDATES",
    "DAY21_MAX_STAGE_C_CANDIDATES",
    "DAY21_REQUIRED_TRAINING_METRICS",
    "DAY21_SCHEMA_VERSION",
    "DAY21_SELECTION_SEEDS",
    "DAY21_STAGE_ORDER",
    "DAY21_STAGE_TARGETS",
    "DAY21_TRAINING_SEEDS",
    "DAY21_WINNER_FAMILY_ID",
    "Day21FinalTrainingConfig",
    "assess_training_health",
    "assess_evaluation_contract_health",
    "build_day21_manifest",
    "build_day21_report",
    "compact_metrics",
    "load_day21_config",
    "manifest_reference",
    "read_day21_manifest",
    "relative_path",
    "render_day21_markdown",
    "select_extension_candidates",
    "select_final_checkpoint",
    "sha256_file",
    "utc_timestamp",
    "validate_day21_manifest",
    "write_json",
]
