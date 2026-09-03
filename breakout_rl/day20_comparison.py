"""Validation, reuse, aggregation, and selection helpers for Day 20.

Day 20 compares three model families under one Contract v2 training protocol.
The module keeps the comparison manifest machine-readable so existing Day 18
evidence can be reused only after its non-family conditions are verified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.day18_comparison import (
    load_evaluation_entries as load_day18_evaluation_entries,
    load_q_probe_entries as load_day18_q_probe_entries,
    load_training_entries as load_day18_training_entries,
)
from breakout_rl.evaluation_contract import (
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summary_from_episode_rows,
    validate_embedded_summary,
    validate_episode_rows,
)
from breakout_rl.models.factory import build_q_network
from breakout_rl.training.backend_manifest import load_day16_backend_manifest
from breakout_rl.training.config import DQNConfig


DAY20_SCHEMA_VERSION = 1
DAY20_TRAINING_SEEDS: tuple[int, int, int] = (11, 22, 33)
DAY20_MILESTONES: dict[str, int] = {
    "screening": 100_000,
    "pilot": 250_000,
    "main": 500_000,
}
DAY20_FORMAL_STAGE = "main"
DAY20_FAMILY_IDS: tuple[str, str, str] = (
    "dqn",
    "double_dqn",
    "dueling_double_dqn",
)
DAY20_STAGE_ORDER: tuple[str, ...] = tuple(DAY20_MILESTONES)
DAY20_EXTENSION_STAGE = "extension_1m"
DAY20_EXTENSION_TARGET = 1_000_000
DAY20_SOURCE_FAMILY_IDS = frozenset({"dqn", "double_dqn"})

# These are the fields that must be identical when an old run is reused. The
# family-specific algorithm/architecture and stage-specific budget are checked
# separately so a changed model family cannot hide a changed backend setting.
DAY20_COMMON_CONFIG_FIELDS: tuple[str, ...] = (
    "gamma",
    "learning_rate",
    "batch_size",
    "replay_capacity",
    "learning_starts",
    "train_frequency",
    "target_update_interval",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "gradient_clip_norm",
    "reward_clip",
    "device",
    "precision",
    "cpu_threads",
    "replay_transfer",
    "replay_backend",
    "profile_stages",
    "num_envs",
    "strict_action_selection_parity",
    "diagnostics_interval",
    "metrics_flush_interval",
)
DAY20_REQUIRED_TRAINING_METRICS: tuple[str, ...] = (
    "loss",
    "q_mean",
    "q_max",
    "target_mean",
    "td_error_mean_abs",
    "gradient_norm",
    "sps",
)


@dataclass(frozen=True)
class DQNFamily:
    """One intentionally named algorithm/architecture combination."""

    family_id: str
    label: str
    algorithm: str
    architecture: str


DAY20_FAMILIES: tuple[DQNFamily, ...] = (
    DQNFamily("dqn", "DQN", "dqn", "standard"),
    DQNFamily("double_dqn", "Double DQN", "double_dqn", "standard"),
    DQNFamily(
        "dueling_double_dqn",
        "Dueling Double DQN",
        "double_dqn",
        "dueling",
    ),
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(
                dict(payload),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination


def relative_path(path: str | Path, *, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve()).replace(
        os.sep,
        "/",
    )


def resolve_manifest_reference(manifest_path: str | Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(manifest_path).resolve().parent / candidate).resolve()


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


def _resolve_reference(value: Any, *, source: Path, repository_root: Path) -> Path:
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


def _artifact_reference(path: Path, *, start: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative_path(path, start=start), "sha256": sha256_file(path)}


def _family_from_payload(value: Any, *, index: int) -> DQNFamily:
    if not isinstance(value, Mapping):
        raise ValueError(f"families[{index}] must be an object")
    fields = ("family_id", "label", "algorithm", "architecture")
    if any(not isinstance(value.get(field), str) or not value[field].strip() for field in fields):
        raise ValueError(f"families[{index}] must contain non-empty family fields")
    return DQNFamily(
        str(value["family_id"]).strip(),
        str(value["label"]).strip(),
        str(value["algorithm"]).strip().lower(),
        str(value["architecture"]).strip().lower(),
    )


@dataclass(frozen=True)
class Day20ExperimentConfig:
    """Validated Day 20 protocol and its source-of-truth dependencies."""

    source_path: Path
    repository_root: Path
    experiment_id: str
    backend_manifest_path: Path
    contract_path: Path
    evaluation_config_path: Path
    probe_states_path: Path
    families: tuple[DQNFamily, ...]
    training_seeds: tuple[int, ...]
    milestones: Mapping[str, int]
    formal_quality_horizon: str
    sequential: bool
    require_cuda: bool
    requested_device: str
    precision: str
    cuda_headroom_bytes: int
    backend_manifest: Mapping[str, Any]
    contract: Any
    evaluation_config: Mapping[str, Any]
    backend_config: DQNConfig
    reuse_enabled: bool
    reuse_manifest_path: Path | None
    extension: Mapping[str, Any]
    selection: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def formal_steps(self) -> int:
        return int(self.milestones[self.formal_quality_horizon])

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(family.family_id for family in self.families)

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
    def evaluation_seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.evaluation_config["seeds"])

    @property
    def episodes_per_seed(self) -> int:
        return int(self.evaluation_config["episodes_per_seed"])

    @property
    def backend_id(self) -> str | None:
        value = self.backend_manifest.get("backend_id")
        return None if value is None else str(value)

    def family(self, family_id: str) -> DQNFamily:
        for family in self.families:
            if family.family_id == family_id:
                return family
        raise ValueError(f"unknown Day 20 family: {family_id}")

    def stage_steps(self, stage: str) -> int:
        if stage not in self.milestones:
            raise ValueError(f"unknown Day 20 stage: {stage}")
        return int(self.milestones[stage])

    def training_config(
        self,
        *,
        family_id: str,
        seed: int,
        stage: str,
    ) -> DQNConfig:
        family = self.family(family_id)
        return self.backend_config.with_overrides(
            algorithm=family.algorithm,
            architecture=family.architecture,
            seed=int(seed),
            total_steps=self.stage_steps(stage),
            checkpoint_interval=self.stage_steps(stage),
            contract_id=self.contract.contract_id,
            contract_path=relative_path(self.contract_path, start=self.repository_root),
        )

    def protocol(self) -> dict[str, Any]:
        return {
            "families": [
                {
                    "family_id": family.family_id,
                    "label": family.label,
                    "algorithm": family.algorithm,
                    "architecture": family.architecture,
                }
                for family in self.families
            ],
            "training_seeds": list(self.training_seeds),
            "milestones": dict(self.milestones),
            "formal_quality_horizon": self.formal_quality_horizon,
            "formal_quality_transitions": self.formal_steps,
            "screening_is_not_final_selection": True,
            "evaluation_seeds": list(self.evaluation_seeds),
            "episodes_per_evaluation_seed": self.episodes_per_seed,
            "evaluation_epsilon": float(self.evaluation_config["epsilon"]),
            "raw_reward": True,
            "requires_cuda": self.require_cuda,
            "requested_device": self.requested_device,
            "precision": self.precision,
            "sequential": self.sequential,
            "cuda_headroom_bytes": self.cuda_headroom_bytes,
            "actual_transition_definition": (
                "global_step is accepted environment transitions, not vector "
                "iterations, optimizer updates, or raw Atari frames"
            ),
        }

    def source_hashes(self) -> dict[str, str | None]:
        paths = {
            "comparison_manifest": self.source_path,
            "backend_manifest": self.backend_manifest_path,
            "contract": self.contract_path,
            "evaluation_config": self.evaluation_config_path,
            "probe_states": self.probe_states_path,
        }
        return {
            name: sha256_file(path) if path.is_file() else None
            for name, path in paths.items()
        }


def load_day20_config(
    path: str | Path = "configs/comparisons/dqn-family/manifest.json",
    *,
    repository_root: str | Path | None = None,
    require_probe_states: bool = False,
) -> Day20ExperimentConfig:
    source = Path(path).resolve()
    payload = _read_json(source)
    root = Path(repository_root).resolve() if repository_root else _repository_root(source)

    if payload.get("schema_version") != DAY20_SCHEMA_VERSION:
        raise ValueError("Day 20 config has an unsupported schema_version")
    if payload.get("day") != 20:
        raise ValueError("Day 20 config must have day=20")
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Day 20 experiment_id must be a non-empty string")

    raw_families = payload.get("families")
    if isinstance(raw_families, (str, bytes)) or not isinstance(raw_families, Sequence):
        raise ValueError("Day 20 families must be a sequence")
    families = tuple(
        _family_from_payload(value, index=index)
        for index, value in enumerate(raw_families)
    )
    if families != DAY20_FAMILIES:
        raise ValueError(
            "Day 20 families must be DQN, Double DQN, and Dueling Double DQN "
            "in that order"
        )
    if len({family.family_id for family in families}) != len(families):
        raise ValueError("Day 20 family_id values must be unique")

    training_seeds = _unique_ints(payload.get("training_seeds"), name="training_seeds")
    if training_seeds != DAY20_TRAINING_SEEDS:
        raise ValueError("Day 20 training_seeds must be [11, 22, 33]")
    raw_milestones = payload.get("milestones")
    if not isinstance(raw_milestones, Mapping):
        raise ValueError("Day 20 milestones must be an object")
    milestones = {
        stage: _positive_int(raw_milestones.get(stage), name=f"milestones.{stage}")
        for stage in DAY20_MILESTONES
    }
    if milestones != DAY20_MILESTONES:
        raise ValueError(
            "Day 20 milestones must be screening=100000, pilot=250000, main=500000"
        )
    formal_horizon = payload.get("formal_quality_horizon")
    if formal_horizon != DAY20_FORMAL_STAGE:
        raise ValueError("formal_quality_horizon must be 'main'")

    backend_path = _resolve_reference(
        payload.get("backend_manifest"),
        source=source,
        repository_root=root,
    )
    contract_path = _resolve_reference(
        payload.get("contract"),
        source=source,
        repository_root=root,
    )
    evaluation_path = _resolve_reference(
        payload.get("evaluation_config"),
        source=source,
        repository_root=root,
    )
    probe_path = _resolve_reference(
        payload.get("probe_states"),
        source=source,
        repository_root=root,
    )
    if require_probe_states and not probe_path.is_file():
        raise FileNotFoundError(probe_path)

    backend_manifest = load_day16_backend_manifest(
        backend_path,
        repository_root=root,
        verify_evidence_files=False,
    )
    trainer = backend_manifest.get("trainer")
    if not isinstance(trainer, Mapping) or not isinstance(trainer.get("config"), Mapping):
        raise ValueError("Day 16 backend manifest is missing trainer.config")
    backend_config = DQNConfig.from_dict(trainer["config"])
    if (
        backend_config.num_envs != 2
        or backend_config.replay_backend != "gpu"
        or backend_config.device != "cuda"
        or backend_config.precision != "float32"
        or not backend_config.strict_action_selection_parity
    ):
        raise ValueError(
            "Day 20 must reuse the Day 16 canonical N=2 GPU Replay CUDA float32 backend"
        )

    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    evaluation_config = _read_json(evaluation_path)
    eval_seeds = _unique_ints(evaluation_config.get("seeds"), name="evaluation.seeds")
    episodes_per_seed = _positive_int(
        evaluation_config.get("episodes_per_seed"),
        name="evaluation.episodes_per_seed",
    )
    epsilon = _finite_float(evaluation_config.get("epsilon"), name="evaluation.epsilon")
    if contract.environment_id != evaluation_config.get("environment_id"):
        raise ValueError("Day 20 evaluation config and Contract v2 disagree on environment")
    if contract.concrete_episode_seeds != expand_concrete_episode_seeds(
        eval_seeds,
        episodes_per_seed=episodes_per_seed,
    ):
        raise ValueError("Day 20 evaluation seeds do not match Contract v2")
    if epsilon != 0.0 or epsilon != contract.evaluation_epsilon:
        raise ValueError("Day 20 formal evaluation requires Contract v2 epsilon=0")

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Day 20 execution must be an object")
    if execution.get("sequential") is not True:
        raise ValueError("Day 20 formal runs must be sequential")
    if execution.get("require_cuda") is not True:
        raise ValueError("Day 20 formal comparison must require CUDA")
    requested_device = str(execution.get("requested_device", "cuda")).strip().lower()
    if requested_device != "cuda":
        raise ValueError("Day 20 formal comparison must request exactly cuda")
    precision = str(execution.get("precision", "float32")).strip().lower()
    if precision not in {"float32", "fp32"}:
        raise ValueError("Day 20 formal comparison must use float32")
    headroom = _positive_int(
        execution.get("cuda_headroom_bytes"),
        name="execution.cuda_headroom_bytes",
    )

    reuse = payload.get("evidence_reuse", {})
    if not isinstance(reuse, Mapping):
        raise ValueError("evidence_reuse must be an object")
    reuse_enabled = bool(reuse.get("enabled", True))
    raw_reuse_path = reuse.get("source_manifest")
    reuse_path = (
        _resolve_reference(raw_reuse_path, source=source, repository_root=root)
        if raw_reuse_path is not None
        else None
    )

    extension = payload.get("optional_1m_extension", {})
    if not isinstance(extension, Mapping):
        raise ValueError("optional_1m_extension must be an object")
    if extension.get("enabled", True) is not True:
        raise ValueError("optional_1m_extension.enabled must be true")
    if _positive_int(extension.get("target_transitions"), name="optional_1m_extension.target_transitions") != 1_000_000:
        raise ValueError("Day 20 optional extension target must be 1000000 transitions")
    if _positive_int(extension.get("max_families"), name="optional_1m_extension.max_families") != 2:
        raise ValueError("Day 20 optional extension may contain at most two families")

    selection = payload.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be an object")

    return Day20ExperimentConfig(
        source_path=source,
        repository_root=root,
        experiment_id=experiment_id.strip(),
        backend_manifest_path=backend_path,
        contract_path=contract_path,
        evaluation_config_path=evaluation_path,
        probe_states_path=probe_path,
        families=families,
        training_seeds=training_seeds,
        milestones=milestones,
        formal_quality_horizon=formal_horizon,
        sequential=True,
        require_cuda=True,
        requested_device=requested_device,
        precision="float32",
        cuda_headroom_bytes=headroom,
        backend_manifest=backend_manifest,
        contract=contract,
        evaluation_config={
            **evaluation_config,
            "seeds": list(eval_seeds),
            "episodes_per_seed": episodes_per_seed,
            "epsilon": epsilon,
        },
        backend_config=backend_config,
        reuse_enabled=reuse_enabled,
        reuse_manifest_path=reuse_path,
        extension=dict(extension),
        selection=dict(selection),
        raw=payload,
    )


def config_diff(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    fields: Iterable[str] = DAY20_COMMON_CONFIG_FIELDS,
) -> dict[str, dict[str, Any]]:
    """Compare only the non-family fields that define fair training."""

    return {
        field: {"first": first.get(field), "second": second.get(field)}
        for field in fields
        if first.get(field) != second.get(field)
    }


def build_day20_manifest(
    config: Day20ExperimentConfig,
    *,
    manifest_path: str | Path,
    runs_root: str | Path = "runs",
    evaluations_root: str | Path = "evaluations",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a 27-entry staged runtime manifest without starting a run."""

    destination = Path(manifest_path).resolve()
    parent = destination.parent
    runs_base = Path(runs_root).resolve() / config.experiment_id
    evaluations_base = Path(evaluations_root).resolve() / config.experiment_id
    entries: list[dict[str, Any]] = []
    for stage, target in config.milestones.items():
        for seed in config.training_seeds:
            for family in config.families:
                run_dir = runs_base / family.family_id / f"seed{seed}" / f"stage-{target // 1000}k"
                evaluation_dir = evaluations_base / family.family_id / f"seed{seed}" / f"step-{target:08d}"
                stage_config = config.training_config(
                    family_id=family.family_id,
                    seed=seed,
                    stage=stage,
                )
                entries.append(
                    {
                        "run_id": f"{config.experiment_id}-{family.family_id}-seed{seed}-{stage}",
                        "pair_id": f"training-seed-{seed}",
                        "family_id": family.family_id,
                        "family_label": family.label,
                        "algorithm": family.algorithm,
                        "architecture": family.architecture,
                        "training_seed": seed,
                        "stage": stage,
                        "target_transitions": target,
                        "status": "pending",
                        "eligible": False,
                        "run_dir": relative_path(run_dir, start=parent),
                        "resume_from": None,
                        "checkpoint": None,
                        "training": {
                            "metrics_path": relative_path(run_dir / "metrics.csv", start=parent),
                            "summary": None,
                            "runtime": None,
                        },
                        "evaluation": {
                            "directory": relative_path(evaluation_dir, start=parent),
                            "results": None,
                            "episodes": None,
                            "summary": None,
                            "status": "pending",
                        },
                        "q_probe": None,
                        "training_config": stage_config.to_dict(),
                        "summary": None,
                        "runtime": None,
                        "source": None,
                        "error": None,
                    }
                )

    random_path = config.repository_root / "evaluations/day16-contract-v2-random/results.json"
    return {
        "schema_version": DAY20_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "status": "planned",
        "sequential": True,
        "source_of_truth": {
            "comparison_manifest": _artifact_reference(
                config.source_path,
                start=config.repository_root,
            ),
            "backend_manifest": {
                **_artifact_reference(config.backend_manifest_path, start=config.repository_root),
                "backend_id": config.backend_id,
            },
            "contract": {
                **_artifact_reference(config.contract_path, start=config.repository_root),
                **config.contract_provenance,
            },
            "evaluation_config": _artifact_reference(
                config.evaluation_config_path,
                start=config.repository_root,
            ),
            "probe_states": {
                "path": relative_path(config.probe_states_path, start=config.repository_root),
                "sha256": sha256_file(config.probe_states_path)
                if config.probe_states_path.is_file()
                else None,
            },
        },
        "protocol": config.protocol(),
        "base_backend": {
            "backend_id": config.backend_id,
            "trainer": config.backend_manifest.get("trainer"),
            "control_values": config.backend_config.to_dict(),
        },
        "evidence_reuse": {
            "enabled": config.reuse_enabled,
            "source_manifest": (
                relative_path(config.reuse_manifest_path, start=parent)
                if config.reuse_manifest_path is not None
                else None
            ),
            "decision": "pending_audit",
            "audit": None,
        },
        "optional_1m_extension": dict(config.extension),
        "selection_rule": dict(config.selection),
        "provenance": {
            "source_hashes": config.source_hashes(),
            "historical_day18_reuse": {
                "training_repeated": False,
                "source_manifest": (
                    relative_path(config.reuse_manifest_path, start=parent)
                    if config.reuse_manifest_path is not None
                    else None
                ),
            },
        },
        "runs": entries,
        "random_baseline_results": (
            relative_path(random_path, start=parent) if random_path.is_file() else None
        ),
        "command": list(command) if command is not None else None,
    }


def read_day20_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != DAY20_SCHEMA_VERSION:
        raise ValueError("unsupported Day 20 manifest schema_version")
    if payload.get("sequential") is not True:
        raise ValueError("Day 20 manifest must record sequential execution")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Day 20 manifest runs must be a non-empty array")
    seen: set[tuple[str, int, str]] = set()
    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Day 20 runs[{index}] must be an object")
        required = ("family_id", "algorithm", "architecture", "training_seed", "stage", "target_transitions")
        if any(field not in entry for field in required):
            raise ValueError(f"Day 20 runs[{index}] is missing a required field")
        key = (str(entry["family_id"]), int(entry["training_seed"]), str(entry["stage"]))
        if key in seen:
            raise ValueError(f"duplicate Day 20 run entry: {key}")
        seen.add(key)
        if key[0] not in DAY20_FAMILY_IDS:
            raise ValueError(f"unsupported Day 20 family: {key[0]}")
        family = next(item for item in DAY20_FAMILIES if item.family_id == key[0])
        if (
            str(entry["algorithm"]) != family.algorithm
            or str(entry["architecture"]) != family.architecture
        ):
            raise ValueError(f"run {index} family metadata does not match {key[0]}")
        if key[2] not in DAY20_MILESTONES and key[2] != DAY20_EXTENSION_STAGE:
            raise ValueError(f"unsupported Day 20 stage: {key[2]}")
        expected_target = (
            DAY20_EXTENSION_TARGET
            if key[2] == DAY20_EXTENSION_STAGE
            else DAY20_MILESTONES[key[2]]
        )
        if int(entry["target_transitions"]) != expected_target:
            raise ValueError(f"run {index} target_transitions does not match its stage")
    return payload


def validate_day20_manifest(
    path: str | Path,
    *,
    config: Day20ExperimentConfig,
) -> dict[str, Any]:
    """Validate a resumable manifest against the current Day 20 protocol."""

    payload = read_day20_manifest(path)
    if payload.get("experiment_id") != config.experiment_id:
        raise ValueError("existing Day 20 manifest belongs to a different experiment")
    protocol = payload.get("protocol")
    if protocol != config.protocol():
        raise ValueError("existing Day 20 manifest protocol does not match the current config")
    source_of_truth = payload.get("source_of_truth")
    if not isinstance(source_of_truth, Mapping):
        raise ValueError("existing Day 20 manifest is missing source_of_truth hashes")
    expected_sources = {
        "comparison_manifest": config.source_path,
        "backend_manifest": config.backend_manifest_path,
        "contract": config.contract_path,
        "evaluation_config": config.evaluation_config_path,
        "probe_states": config.probe_states_path,
    }
    for name, expected_path in expected_sources.items():
        reference = source_of_truth.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"existing Day 20 manifest is missing {name} provenance")
        expected_hash = sha256_file(expected_path) if expected_path.is_file() else None
        if expected_hash is None or reference.get("sha256") != expected_hash:
            raise ValueError(f"existing Day 20 manifest {name} hash does not match current config")
    backend = payload.get("base_backend")
    control_values = backend.get("control_values") if isinstance(backend, Mapping) else None
    if not isinstance(control_values, Mapping):
        raise ValueError("existing Day 20 manifest is missing backend control values")
    backend_diffs = config_diff(control_values, config.backend_config.to_dict())
    if backend_diffs:
        raise ValueError(f"existing Day 20 backend controls differ: {backend_diffs}")
    if payload.get("optional_1m_extension") != dict(config.extension):
        raise ValueError("existing Day 20 extension policy does not match the current config")
    if payload.get("selection_rule") != dict(config.selection):
        raise ValueError("existing Day 20 selection rule does not match the current config")
    return payload


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(entry["family_id"]),
        int(entry["training_seed"]),
        str(entry["stage"]),
    )


def _source_path(source_manifest: Path, value: Any) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve_manifest_reference(source_manifest, value)


def _source_entry_map(
    source_manifest: Path,
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[tuple[str, int, str], dict[str, Any]],
    dict[tuple[str, int, str], dict[str, Any]],
]:
    training = load_day18_training_entries(
        source_manifest,
        include_metrics=True,
        require_checkpoint=False,
    )
    evaluations = load_day18_evaluation_entries(source_manifest, training)
    probes = load_day18_q_probe_entries(source_manifest, training)
    training_map = {
        (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"])): dict(entry)
        for entry in training
    }
    evaluation_map = {
        (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"])): dict(entry)
        for entry in evaluations
    }
    probe_map = {
        (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"])): dict(entry)
        for entry in probes
    }
    return training_map, evaluation_map, probe_map


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    expected: Any = None,
    observed: Any = None,
    detail: str | None = None,
) -> None:
    check = {"name": name, "passed": bool(passed)}
    if expected is not None:
        check["expected"] = expected
    if observed is not None:
        check["observed"] = observed
    if detail:
        check["detail"] = detail
    checks.append(check)


def _metrics_have_required_fields(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(
        isinstance(row, Mapping)
        and all(field in row for field in DAY20_REQUIRED_TRAINING_METRICS)
        for row in value
    )


def _summary_proves_budget(summary: Mapping[str, Any], *, target: int) -> bool:
    return all(
        summary.get(field) == target
        for field in (
            "total_transitions",
            "training_steps",
            "physical_environment_steps",
        )
    )


def _runtime_proves_formal_cuda(
    runtime: Mapping[str, Any],
    *,
    target: int,
) -> bool:
    finite_positive = (
        "steps_per_second",
        "wall_clock_seconds",
        "cuda_peak_allocated_bytes",
        "gpu_memory_total_bytes",
    )
    return bool(
        runtime.get("requested_device") == "cuda"
        and str(runtime.get("resolved_device", "")).startswith("cuda:")
        and runtime.get("precision") == "float32"
        and runtime.get("cuda_available") is True
        and isinstance(runtime.get("cuda_device_index"), int)
        and isinstance(runtime.get("cuda_device_name"), str)
        and bool(runtime.get("cuda_device_name"))
        and isinstance(runtime.get("pytorch_version"), str)
        and bool(runtime.get("pytorch_version"))
        and isinstance(runtime.get("torch_cuda_version"), str)
        and bool(runtime.get("torch_cuda_version"))
        and all(
            _finite_metric(runtime.get(field)) is not None
            and float(runtime[field]) > 0.0
            for field in finite_positive
        )
        and all(
            runtime.get(field) == target
            for field in (
                "training_steps",
                "physical_environment_steps",
                "action_inference_transitions",
                "replay_insertion_transitions",
            )
        )
    )


def audit_day18_evidence_reuse(
    config: Day20ExperimentConfig,
    *,
    source_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Verify whether Day 18 DQN/Double evidence can enter Day 20.

    The audit treats implementation hashes from the historical run as
    provenance, not as a reason to silently reject compatible artifacts after
    a later architecture-aware refactor. Fairness is decided by the concrete
    Contract/backend/evaluation hashes and every run's recorded config/runtime.
    """

    source = Path(source_manifest).resolve() if source_manifest else config.reuse_manifest_path
    checks: list[dict[str, Any]] = []
    incompatibilities: list[str] = []
    reusable_entries: list[dict[str, Any]] = []
    runtime_identities: dict[str, set[str]] = {}
    if source is None:
        return {
            "schema_version": DAY20_SCHEMA_VERSION,
            "status": "incompatible",
            "reuse_allowed": False,
            "source_manifest": None,
            "checks": [],
            "incompatibilities": ["no Day 18 evidence manifest was configured"],
            "reusable_entries": [],
        }
    source = source.resolve()
    try:
        source_payload = _read_json(source)
        training_map, evaluation_map, probe_map = _source_entry_map(source)
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        return {
            "schema_version": DAY20_SCHEMA_VERSION,
            "status": "incompatible",
            "reuse_allowed": False,
            "source_manifest": source.as_posix(),
            "checks": [],
            "incompatibilities": [f"unable to read Day 18 evidence: {error}"],
            "reusable_entries": [],
        }

    protocol = source_payload.get("protocol", {})
    if not isinstance(protocol, Mapping):
        protocol = {}
    _check(
        checks,
        "source status completed",
        source_payload.get("status") == "completed",
        expected="completed",
        observed=source_payload.get("status"),
    )
    expected_protocol = {
        "training_seeds": list(config.training_seeds),
        "milestones": dict(config.milestones),
        "formal_quality_horizon": config.formal_quality_horizon,
        "formal_quality_transitions": config.formal_steps,
        "evaluation_seeds": list(config.evaluation_seeds),
        "episodes_per_evaluation_seed": config.episodes_per_seed,
        "evaluation_epsilon": 0.0,
        "raw_reward": True,
        "requires_cuda": True,
        "sequential": True,
    }
    for field, expected in expected_protocol.items():
        observed = protocol.get(field)
        passed = observed == expected
        _check(checks, f"protocol.{field}", passed, expected=expected, observed=observed)
        if not passed:
            incompatibilities.append(f"protocol.{field}: expected {expected!r}, got {observed!r}")

    source_truth = source_payload.get("source_of_truth", {})
    if not isinstance(source_truth, Mapping):
        source_truth = {}
    for name, path, source_key in (
        ("backend manifest", config.backend_manifest_path, "backend_manifest"),
        ("Contract v2", config.contract_path, "contract"),
        ("evaluation config", config.evaluation_config_path, "evaluation_config"),
        ("probe states", config.probe_states_path, "probe_states"),
    ):
        reference = source_truth.get(source_key)
        observed_hash = reference.get("sha256") if isinstance(reference, Mapping) else None
        expected_hash = sha256_file(path) if path.is_file() else None
        passed = expected_hash is not None and observed_hash == expected_hash
        _check(checks, f"{name} hash", passed, expected=expected_hash, observed=observed_hash)
        if not passed:
            incompatibilities.append(
                f"{name} hash mismatch: expected {expected_hash!r}, got {observed_hash!r}"
            )

    source_backend = source_payload.get("base_backend", {})
    source_values = source_backend.get("control_values", {}) if isinstance(source_backend, Mapping) else {}
    if not isinstance(source_values, Mapping):
        source_values = {}
    expected_values = config.backend_config.to_dict()
    backend_diffs = config_diff(source_values, expected_values)
    _check(
        checks,
        "Day 16 backend control values",
        not backend_diffs,
        expected={field: expected_values.get(field) for field in DAY20_COMMON_CONFIG_FIELDS},
        observed={field: source_values.get(field) for field in DAY20_COMMON_CONFIG_FIELDS},
    )
    if backend_diffs:
        incompatibilities.append(f"backend control values differ: {backend_diffs}")

    expected_keys = {
        (algorithm, seed, stage)
        for algorithm in ("dqn", "double_dqn")
        for seed in config.training_seeds
        for stage in DAY20_MILESTONES
    }
    observed_keys = set(training_map)
    keys_match = expected_keys.issubset(observed_keys)
    _check(
        checks,
        "all Day 18 family/seed/stage entries exist",
        keys_match,
        expected=len(expected_keys),
        observed=len(expected_keys & observed_keys),
    )
    if not keys_match:
        incompatibilities.append("Day 18 is missing one or more DQN/Double seed-stage entries")
    raw_source_entries = {
        (
            str(item.get("algorithm")),
            int(item.get("training_seed", -1)),
            str(item.get("stage")),
        ): item
        for item in source_payload.get("runs", [])
        if isinstance(item, Mapping)
    }

    for algorithm, seed, stage in sorted(expected_keys, key=lambda value: (value[2], value[1], value[0])):
        key = (algorithm, seed, stage)
        training = training_map.get(key)
        evaluation = evaluation_map.get(key)
        probe = probe_map.get(key)
        expected_config = config.training_config(
            family_id="dqn" if algorithm == "dqn" else "double_dqn",
            seed=seed,
            stage=stage,
        ).to_dict()
        if training is None:
            continue
        training_config = training.get("config", {})
        if not isinstance(training_config, Mapping):
            training_config = {}
        variable_checks = {
            "algorithm": training.get("algorithm", training_config.get("algorithm")) == algorithm,
            "architecture": training.get("architecture", training_config.get("architecture", "standard"))
            == "standard",
            "seed": training.get("training_seed", training_config.get("seed")) == seed,
            "total_steps": training_config.get("total_steps") == DAY20_MILESTONES[stage],
            "checkpoint_interval": training_config.get("checkpoint_interval")
            == DAY20_MILESTONES[stage],
        }
        fixed_fields = tuple(
            field
            for field in expected_config
            if field
            not in {"algorithm", "architecture", "seed", "total_steps", "contract_id", "contract_path"}
        )
        diffs = config_diff(training_config, expected_config, fields=fixed_fields)
        passed_config = not diffs and all(variable_checks.values())
        _check(
            checks,
            f"run {algorithm}/seed{seed}/{stage} common config",
            passed_config,
            observed={
                "diffs": diffs,
                "variable_checks": variable_checks,
            }
            if not passed_config
            else "match",
        )
        if not passed_config:
            incompatibilities.append(
                f"{algorithm}/seed{seed}/{stage} config differs: "
                f"diffs={diffs}, variable_checks={variable_checks}"
            )

        summary = training.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        runtime = training.get("runtime", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        metrics_ok = _metrics_have_required_fields(training.get("metrics"))
        _check(
            checks,
            f"run {algorithm}/seed{seed}/{stage} required training metrics",
            metrics_ok,
            expected=list(DAY20_REQUIRED_TRAINING_METRICS),
        )
        if not metrics_ok:
            incompatibilities.append(
                f"{algorithm}/seed{seed}/{stage} is missing required training metrics"
            )
        exact_steps = _summary_proves_budget(summary, target=DAY20_MILESTONES[stage])
        completed = (
            training.get("eligible") is True
            and summary.get("status") == "completed"
            and exact_steps
        )
        _check(
            checks,
            f"run {algorithm}/seed{seed}/{stage} completed exact budget",
            completed,
            expected=DAY20_MILESTONES[stage],
            observed={"status": summary.get("status"), "total_transitions": summary.get("total_transitions")},
        )
        if not completed:
            incompatibilities.append(f"{algorithm}/seed{seed}/{stage} is incomplete or ineligible")

        formal_runtime = _runtime_proves_formal_cuda(
            runtime,
            target=DAY20_MILESTONES[stage],
        )
        for field in (
            "cuda_device_name",
            "cuda_device_index",
            "pytorch_version",
            "torch_cuda_version",
        ):
            runtime_identities.setdefault(field, set()).add(str(runtime.get(field)))
        _check(
            checks,
            f"run {algorithm}/seed{seed}/{stage} formal CUDA runtime",
            formal_runtime,
            expected={"requested_device": "cuda", "precision": "float32", "cuda_available": True},
            observed={
                "requested_device": runtime.get("requested_device"),
                "resolved_device": runtime.get("resolved_device"),
                "precision": runtime.get("precision"),
                "cuda_available": runtime.get("cuda_available"),
            },
        )
        if not formal_runtime:
            incompatibilities.append(f"{algorithm}/seed{seed}/{stage} did not run under formal CUDA")

        contract = training.get("environment_contract")
        contract_match = isinstance(contract, Mapping) and (
            contract.get("contract_id") == config.contract.contract_id
            and contract.get("contract_sha256") == sha256_file(config.contract_path)
        )
        _check(checks, f"run {algorithm}/seed{seed}/{stage} Contract v2", contract_match)
        if not contract_match:
            incompatibilities.append(f"{algorithm}/seed{seed}/{stage} Contract v2 provenance is missing or mismatched")

        eval_ok = isinstance(evaluation, Mapping) and evaluation.get("eligible") is True
        probe_ok = isinstance(probe, Mapping) and probe.get("eligible") is True
        _check(checks, f"run {algorithm}/seed{seed}/{stage} evaluation", eval_ok)
        _check(checks, f"run {algorithm}/seed{seed}/{stage} Q probe", probe_ok)
        if not eval_ok:
            incompatibilities.append(f"{algorithm}/seed{seed}/{stage} evaluation is incomplete or ineligible")
        if not probe_ok:
            incompatibilities.append(f"{algorithm}/seed{seed}/{stage} Q probe is incomplete or ineligible")

        artifact_entry = dict(training)
        artifact_entry["algorithm"] = algorithm
        artifact_entry["architecture"] = "standard"
        artifact_entry["training_seed"] = seed
        raw_source_entry = raw_source_entries.get(key)
        if isinstance(raw_source_entry, Mapping):
            artifact_entry["checkpoint"] = raw_source_entry.get("checkpoint")
            raw_evaluation = raw_source_entry.get("evaluation")
            if isinstance(raw_evaluation, Mapping):
                artifact_entry["evaluation"] = dict(raw_evaluation)
            raw_probe = raw_source_entry.get("q_probe")
            if raw_probe is not None:
                artifact_entry["q_probe"] = raw_probe
        artifact_entry["summary"] = summary
        artifact_entry["runtime"] = runtime
        artifact_entry["status"] = "completed"
        artifact_entry["eligible"] = completed and formal_runtime
        artifact_evaluation = artifact_entry.get("evaluation")
        if isinstance(artifact_evaluation, Mapping):
            artifact_evaluation = dict(artifact_evaluation)
        else:
            artifact_evaluation = dict(evaluation) if isinstance(evaluation, Mapping) else {}
        artifact_evaluation["status"] = (
            evaluation.get("status") if isinstance(evaluation, Mapping) else None
        )
        artifact_evaluation["eligible"] = (
            evaluation.get("eligible") if isinstance(evaluation, Mapping) else False
        )
        artifact_entry["evaluation"] = artifact_evaluation
        artifact_entry["q_probe"] = probe
        artifact_valid, _computed_summary, artifact_error = _validated_entry_artifacts(
            source,
            artifact_entry,
            config=config,
        )
        _check(
            checks,
            f"run {algorithm}/seed{seed}/{stage} raw artifacts",
            artifact_valid,
            observed=artifact_error if not artifact_valid else "match",
        )
        if not artifact_valid:
            incompatibilities.append(
                f"{algorithm}/seed{seed}/{stage} raw evidence is invalid: {artifact_error}"
            )

        if (
            completed
            and formal_runtime
            and contract_match
            and eval_ok
            and probe_ok
            and artifact_valid
            and not diffs
            and all(variable_checks.values())
        ):
            reusable_entries.append(
                {
                    "algorithm": algorithm,
                    "architecture": "standard",
                    "training_seed": seed,
                    "stage": stage,
                    "source_run_id": training.get("run_id"),
                    "source_run_dir": training.get("run_dir"),
                }
            )

    runtime_identity_match = all(len(values) <= 1 for values in runtime_identities.values())
    _check(
        checks,
        "Day 18 formal runtime identity is consistent",
        runtime_identity_match,
        observed={field: sorted(values) for field, values in runtime_identities.items()},
    )
    if not runtime_identity_match:
        incompatibilities.append("Day 18 formal runs do not share one CUDA runtime identity")

    reuse_allowed = not incompatibilities and len(reusable_entries) == len(expected_keys)
    return {
        "schema_version": DAY20_SCHEMA_VERSION,
        "status": "compatible" if reuse_allowed else "incompatible",
        "reuse_allowed": reuse_allowed,
        "source_manifest": source.as_posix(),
        "source_manifest_sha256": sha256_file(source),
        "historical_source_note": (
            "Day 18 implementation hashes are retained as provenance; fairness is "
            "decided by the shared Contract/backend/config/runtime checks above."
        ),
        "checks": checks,
        "incompatibilities": incompatibilities,
        "reusable_entries": reusable_entries,
        "reusable_entry_count": len(reusable_entries),
        "expected_reusable_entry_count": len(expected_keys),
    }


def _metrics_path_for_entry(manifest_path: Path, entry: Mapping[str, Any]) -> Path | None:
    training = entry.get("training")
    if isinstance(training, Mapping):
        value = training.get("metrics_path")
        if isinstance(value, str):
            candidate = resolve_manifest_reference(manifest_path, value)
            if candidate.is_file():
                return candidate
    value = entry.get("metrics_path")
    if isinstance(value, str):
        candidate = resolve_manifest_reference(manifest_path, value)
        if candidate.is_file():
            return candidate
    run_dir = entry.get("run_dir")
    if isinstance(run_dir, str):
        candidate = resolve_manifest_reference(manifest_path, run_dir) / "metrics.csv"
        if candidate.is_file():
            return candidate
    return None


def read_metrics(manifest_path: str | Path, entry: Mapping[str, Any]) -> list[dict[str, str]]:
    path = _metrics_path_for_entry(Path(manifest_path).resolve(), entry)
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _validated_evaluation_summary(
    path: str | Path,
    *,
    config: Day20ExperimentConfig,
    require_cuda: bool,
    expected_checkpoint_sha: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_algorithm: str | None = None,
    expected_architecture: str | None = None,
    expected_training_seed: int | None = None,
    expected_training_budget: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    payload = read_evaluation_results(source)
    if payload.get("environment_id") != config.contract.environment_id:
        raise ValueError("evaluation environment does not match Contract v2")
    if payload.get("evaluation_seeds") != list(config.evaluation_seeds):
        raise ValueError("evaluation seeds do not match the Day 20 evaluation config")
    if payload.get("episodes_per_seed") != config.episodes_per_seed:
        raise ValueError("evaluation episode count does not match the Day 20 evaluation config")
    if float(payload.get("evaluation_epsilon", float("nan"))) != 0.0:
        raise ValueError("evaluation epsilon is not zero")
    metadata = payload.get("metadata", {})
    contract_provenance = (
        metadata.get("evaluation_contract_provenance")
        if isinstance(metadata, Mapping)
        else None
    )
    embedded_contract = metadata.get("evaluation_contract") if isinstance(metadata, Mapping) else None
    contract_valid = (
        isinstance(contract_provenance, Mapping)
        and contract_provenance.get("contract_id") == config.contract.contract_id
        and contract_provenance.get("contract_sha256") == sha256_file(config.contract_path)
    ) or embedded_contract == config.contract.to_dict()
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("raw_reward") is not True
        or not contract_valid
    ):
        raise ValueError("evaluation Contract v2/raw-reward provenance is invalid")
    checkpoint = payload.get("checkpoint")
    if expected_checkpoint_sha is not None:
        if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != expected_checkpoint_sha:
            raise ValueError("evaluation is not bound to the expected checkpoint hash")
    if expected_checkpoint_step is not None:
        if not isinstance(checkpoint, Mapping) or checkpoint.get("step") != expected_checkpoint_step:
            raise ValueError("evaluation is not bound to the expected checkpoint step")
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    for field, expected in (
        ("algorithm", expected_algorithm),
        ("architecture", expected_architecture),
        ("training_seed", expected_training_seed),
        ("training_budget", expected_training_budget),
    ):
        if expected is not None and training.get(field) != expected:
            raise ValueError(f"evaluation training.{field} does not match the run entry")
    if require_cuda:
        runtime = payload.get("runtime", {})
        if (
            payload.get("requested_device") != "cuda"
            or not str(payload.get("resolved_device", "")).startswith("cuda:")
            or not isinstance(runtime, Mapping)
            or runtime.get("cuda_available") is not True
        ):
            raise ValueError("formal evaluation did not run on CUDA")
    rows = validate_episode_rows(
        payload,
        source=source,
        expected_seeds=config.evaluation_seeds,
        expected_episodes_per_seed=config.episodes_per_seed,
        require_complete=True,
    )
    computed = summary_from_episode_rows(rows)
    validate_embedded_summary(
        payload,
        computed,
        source=source,
        require_time_limit_fields=True,
    )
    return computed


def _evaluation_summary_matches(
    cached: Any,
    computed: Mapping[str, Any],
) -> bool:
    if not isinstance(cached, Mapping):
        return False
    for field in (
        "count",
        "mean_return",
        "median_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_episode_length",
        "complete_episodes",
    ):
        try:
            if not math.isclose(float(cached[field]), float(computed[field]), rel_tol=1e-9, abs_tol=1e-9):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _validated_entry_artifacts(
    manifest_path: Path,
    entry: Mapping[str, Any],
    *,
    config: Day20ExperimentConfig,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Validate raw metrics/evaluation/Q-probe artifacts before aggregation."""

    try:
        metrics_path = _metrics_path_for_entry(manifest_path, entry)
        if metrics_path is None:
            raise FileNotFoundError("training metrics.csv is missing")
        training_record = entry.get("training")
        expected_metrics_hash = (
            training_record.get("metrics_sha256")
            if isinstance(training_record, Mapping)
            else entry.get("metrics_sha256")
        )
        if (
            isinstance(expected_metrics_hash, str)
            and sha256_file(metrics_path) != expected_metrics_hash
        ):
            raise ValueError("training metrics hash does not match its recorded provenance")
        metrics = read_metrics(manifest_path, entry)
        if not _metrics_have_required_fields(metrics):
            raise ValueError("training metrics are missing required fields")
        target = int(entry.get("target_transitions", -1))
        observed_steps = max(
            (int(float(row["global_step"])) for row in metrics if row.get("global_step")),
            default=-1,
        )
        if observed_steps != target:
            raise ValueError(
                f"training metrics end at {observed_steps}, expected {target} transitions"
            )

        checkpoint = entry.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint provenance is missing")
        if checkpoint.get("step") != target or not isinstance(checkpoint.get("sha256"), str):
            raise ValueError("checkpoint provenance does not match the target transition budget")
        for field, expected in (
            ("algorithm", entry.get("algorithm")),
            ("architecture", entry.get("architecture")),
            ("training_steps", target),
        ):
            if field in checkpoint and checkpoint.get(field) != expected:
                raise ValueError(f"checkpoint {field} does not match the run entry")
        checkpoint_path = _source_path(manifest_path, checkpoint.get("path"))
        if checkpoint_path is not None and checkpoint_path.is_file():
            if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
                raise ValueError("checkpoint hash does not match its recorded provenance")

        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("evaluation record is missing")
        results_path = _source_path(manifest_path, evaluation.get("results"))
        if results_path is None or not results_path.is_file():
            raise FileNotFoundError("evaluation results.json is missing")
        episodes_path = _source_path(manifest_path, evaluation.get("episodes"))
        if episodes_path is None or not episodes_path.is_file():
            raise FileNotFoundError("evaluation episodes.csv is missing")
        for field, artifact_path in (
            ("results_sha256", results_path),
            ("episodes_sha256", episodes_path),
        ):
            expected_hash = evaluation.get(field)
            if isinstance(expected_hash, str) and sha256_file(artifact_path) != expected_hash:
                raise ValueError(f"evaluation {field} does not match its artifact")
        computed_summary = _validated_evaluation_summary(
            results_path,
            config=config,
            require_cuda=True,
            expected_checkpoint_sha=(
                checkpoint.get("sha256")
            ),
            expected_checkpoint_step=target,
            expected_algorithm=str(entry.get("algorithm")),
            expected_architecture=str(entry.get("architecture")),
            expected_training_seed=int(entry.get("training_seed", -1)),
            expected_training_budget=int(entry.get("target_transitions", -1)),
        )
        if not _evaluation_summary_matches(evaluation.get("summary"), computed_summary):
            raise ValueError("cached evaluation summary does not match raw episodes")

        probe = entry.get("q_probe")
        if not isinstance(probe, Mapping):
            raise ValueError("Q-probe record is missing")
        probe_path = _source_path(manifest_path, probe.get("path"))
        if probe_path is None or not probe_path.is_file():
            raise FileNotFoundError("Q-probe artifact is missing")
        expected_probe_hash = probe.get("sha256")
        if isinstance(expected_probe_hash, str) and sha256_file(probe_path) != expected_probe_hash:
            raise ValueError("Q-probe hash does not match its recorded provenance")
        probe_payload = _read_json(probe_path)
        probe_states = probe_payload.get("probe_states")
        if (
            not isinstance(probe_states, Mapping)
            or probe_states.get("sha256") != sha256_file(config.probe_states_path)
        ):
            raise ValueError("Q-probe is not bound to the configured Day 17 probe states")
        analysis = probe_payload.get("analysis")
        if not isinstance(analysis, Mapping):
            raise ValueError("Q-probe analysis is missing")
        probe_count = int(analysis.get("probe_count", 0))
        action_count = int(analysis.get("action_count", 0))
        q_values = analysis.get("q_values")
        if probe_count < 1 or action_count != 4 or not isinstance(q_values, list):
            raise ValueError("Q-probe analysis has invalid dimensions")
        if len(q_values) != probe_count or any(
            not isinstance(row, list) or len(row) != action_count for row in q_values
        ):
            raise ValueError("Q-probe values do not match their recorded dimensions")
        return True, computed_summary, None
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        return False, None, str(error)


def _copy_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def apply_day18_reuse(
    manifest_path: str | Path,
    manifest: dict[str, Any],
    *,
    config: Day20ExperimentConfig,
    audit: Mapping[str, Any],
    source_manifest: str | Path,
) -> int:
    """Populate DQN/Double entries with compact references from Day 18."""

    if audit.get("reuse_allowed") is not True:
        return 0
    reuse_state = manifest.get("evidence_reuse")
    if isinstance(reuse_state, Mapping) and reuse_state.get("decision") in {
        "disabled",
        "incompatible_fresh_required",
    }:
        return 0
    source = Path(source_manifest).resolve()
    training_map, evaluation_map, probe_map = _source_entry_map(source)
    raw_source_entries = {}
    try:
        raw_source = _read_json(source)
        raw_source_entries = {
            (
                str(item.get("algorithm")),
                int(item.get("training_seed", -1)),
                str(item.get("stage")),
            ): item
            for item in raw_source.get("runs", [])
            if isinstance(item, Mapping)
        }
    except (FileNotFoundError, TypeError, ValueError):
        raw_source_entries = {}
    target_path = Path(manifest_path).resolve()
    count = 0
    for entry in manifest.get("runs", []):
        if (
            not isinstance(entry, dict)
            or entry.get("family_id") not in DAY20_SOURCE_FAMILY_IDS
            or entry.get("stage") not in DAY20_MILESTONES
        ):
            continue
        entry_source = entry.get("source")
        is_historical = entry.get("status") == "reused" or (
            isinstance(entry_source, Mapping)
            and entry_source.get("kind") == "day18_evidence_reuse"
        )
        if entry.get("status") not in {"pending", "reused"} and not is_historical:
            raise ValueError(
                "cannot apply Day 18 reuse over an existing fresh Day 20 run; "
                "create a new manifest to change reuse mode"
            )
        key = (str(entry["algorithm"]), int(entry["training_seed"]), str(entry["stage"]))
        training = training_map.get(key)
        evaluation = evaluation_map.get(key)
        probe = probe_map.get(key)
        if training is None or evaluation is None or probe is None:
            continue
        raw_entry = raw_source_entries.get(key, {})
        raw_evaluation = raw_entry.get("evaluation") if isinstance(raw_entry, Mapping) else None
        evaluation_reference = raw_evaluation if isinstance(raw_evaluation, Mapping) else evaluation
        source_metrics = _metrics_path_for_entry(source, training)
        source_results = _source_path(source, evaluation_reference.get("results"))
        source_episodes = _source_path(source, evaluation_reference.get("episodes"))
        source_probe = _source_path(source, probe.get("path"))
        training_summary = training.get("summary", {})
        training_runtime = training.get("runtime", {})
        entry["status"] = "reused"
        entry["eligible"] = bool(
            training.get("eligible") is True
            and evaluation.get("eligible") is True
            and probe.get("eligible") is True
        )
        entry["summary"] = _copy_json_value(training_summary)
        entry["runtime"] = _copy_json_value(training_runtime)
        entry["training"] = {
            "metrics_path": (
                relative_path(source_metrics, start=target_path.parent)
                if source_metrics is not None
                else None
            ),
            "summary": _copy_json_value(training_summary),
            "runtime": _copy_json_value(training_runtime),
            "metrics_sha256": (
                sha256_file(source_metrics) if source_metrics is not None else None
            ),
        }
        entry["evaluation"] = {
            "directory": relative_path(
                source_results.parent if source_results is not None else source.parent,
                start=target_path.parent,
            ),
            "results": (
                relative_path(source_results, start=target_path.parent)
                if source_results is not None
                else None
            ),
            "episodes": (
                relative_path(source_episodes, start=target_path.parent)
                if source_episodes is not None
                else None
            ),
            "results_sha256": (
                sha256_file(source_results) if source_results is not None else None
            ),
            "episodes_sha256": (
                sha256_file(source_episodes) if source_episodes is not None else None
            ),
            "summary": _copy_json_value(evaluation_reference.get("summary", {})),
            "status": "reused" if evaluation.get("eligible") is True else "ineligible",
        }
        entry["q_probe"] = {
            "path": (
                relative_path(source_probe, start=target_path.parent)
                if source_probe is not None
                else None
            ),
            "sha256": sha256_file(source_probe) if source_probe is not None else None,
            "summary": _copy_json_value(probe.get("summary", {})),
            "status": "reused" if probe.get("eligible") is True else "ineligible",
        }
        raw_checkpoint = raw_entry.get("checkpoint") if isinstance(raw_entry, Mapping) else None
        entry["checkpoint"] = _copy_json_value(training.get("checkpoint"))
        if isinstance(raw_checkpoint, Mapping):
            entry["checkpoint"] = {
                "path": raw_checkpoint.get("path"),
                "sha256": raw_checkpoint.get("sha256"),
                "step": raw_checkpoint.get("step"),
                "source_manifest": relative_path(source, start=target_path.parent),
                "availability": "historical_external_checkpoint",
            }
        entry["source"] = {
            "kind": "day18_evidence_reuse",
            "manifest": relative_path(source, start=target_path.parent),
            "run_id": training.get("run_id"),
            "run_dir": training.get("run_dir"),
            "historical_git_commit_sha": training.get("runtime", {}).get("git_commit_sha")
            if isinstance(training.get("runtime"), Mapping)
            else None,
        }
        count += 1
    reuse = manifest.setdefault("evidence_reuse", {})
    if isinstance(reuse, dict):
        reuse["decision"] = "compatible_reused"
        reuse["audit"] = _copy_json_value(audit)
    return count


def _entry_summary(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = entry.get("summary")
    if isinstance(summary, Mapping):
        return summary
    training = entry.get("training")
    if isinstance(training, Mapping) and isinstance(training.get("summary"), Mapping):
        return training["summary"]
    return {}


def _entry_runtime(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = entry.get("runtime")
    if isinstance(runtime, Mapping):
        return runtime
    training = entry.get("training")
    if isinstance(training, Mapping) and isinstance(training.get("runtime"), Mapping):
        return training["runtime"]
    return {}


def _evaluation_summary(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = entry.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("summary"), Mapping):
        return evaluation["summary"]
    return {}


def _formal_entry(entry: Mapping[str, Any], *, config: Day20ExperimentConfig) -> bool:
    summary = _entry_summary(entry)
    runtime = _entry_runtime(entry)
    evaluation = _evaluation_summary(entry)
    evaluation_record = entry.get("evaluation")
    probe = entry.get("q_probe")
    return bool(
        entry.get("_artifact_valid", True) is not False
        and entry.get("stage") == config.formal_quality_horizon
        and int(entry.get("target_transitions", -1)) == config.formal_steps
        and entry.get("status") in {"completed", "reused"}
        and entry.get("eligible") is True
        and summary.get("status") == "completed"
        and _summary_proves_budget(summary, target=config.formal_steps)
        and isinstance(evaluation_record, Mapping)
        and evaluation_record.get("status") in {"completed", "reused"}
        and evaluation.get("count") == config.evaluation_config.get("episodes_per_seed", 5) * len(config.evaluation_seeds)
        and isinstance(probe, Mapping)
        and probe.get("status") in {"completed", "reused"}
        and _runtime_proves_formal_cuda(runtime, target=config.formal_steps)
    )


def _complete_stage_entry(
    entry: Mapping[str, Any],
    *,
    config: Day20ExperimentConfig,
    stage: str,
    target: int,
) -> bool:
    if stage == config.formal_quality_horizon and target == config.formal_steps:
        return _formal_entry(entry, config=config)
    summary = _entry_summary(entry)
    runtime = _entry_runtime(entry)
    evaluation = _evaluation_summary(entry)
    evaluation_record = entry.get("evaluation")
    probe = entry.get("q_probe")
    return bool(
        entry.get("_artifact_valid", True) is not False
        and entry.get("stage") == stage
        and int(entry.get("target_transitions", -1)) == target
        and entry.get("status") in {"completed", "reused"}
        and entry.get("eligible") is True
        and summary.get("status") == "completed"
        and _summary_proves_budget(summary, target=target)
        and isinstance(evaluation_record, Mapping)
        and evaluation_record.get("status") in {"completed", "reused"}
        and evaluation.get("count")
        == config.evaluation_config.get("episodes_per_seed", 5) * len(config.evaluation_seeds)
        and isinstance(probe, Mapping)
        and probe.get("status") in {"completed", "reused"}
        and _runtime_proves_formal_cuda(runtime, target=target)
    )


def _finite_metric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _latest_return_metrics(metrics: Sequence[Mapping[str, Any]], *, window: int = 20) -> tuple[float, float] | None:
    values: list[float] = []
    for row in metrics:
        value = _finite_metric(row.get("raw_episode_return"))
        if value is not None:
            values.append(value)
    if len(values) < window * 2:
        return None
    return fmean(values[-window:]), fmean(values[-window * 2 : -window])


def _healthy_growth_unresolved(
    manifest_path: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    config: Day20ExperimentConfig,
    family_ids: Sequence[str],
) -> bool:
    """Detect a top candidate whose recent training return is still rising."""

    candidates = set(family_ids)
    for entry in entries:
        if (
            entry.get("stage") != config.formal_quality_horizon
            or entry.get("family_id") not in candidates
            or not _complete_stage_entry(
                entry,
                config=config,
                stage=config.formal_quality_horizon,
                target=config.formal_steps,
            )
        ):
            continue
        metrics = read_metrics(manifest_path, entry)
        latest = _latest_return_metrics(metrics)
        if latest is not None and latest[0] > latest[1] + 0.5:
            return True
    return False


def _aggregate_family(
    family: DQNFamily,
    entries: Sequence[Mapping[str, Any]],
    *,
    config: Day20ExperimentConfig,
    stage: str | None = None,
    target: int | None = None,
) -> dict[str, Any]:
    selected_stage = stage or config.formal_quality_horizon
    selected_target = config.formal_steps if target is None else int(target)
    main_entries = {
        int(entry["training_seed"]): entry
        for entry in entries
        if entry.get("family_id") == family.family_id
        and entry.get("stage") == selected_stage
        and int(entry.get("target_transitions", -1)) == selected_target
    }
    seed_values: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        entry = main_entries.get(seed)
        complete = (
            _complete_stage_entry(
                entry,
                config=config,
                stage=selected_stage,
                target=selected_target,
            )
            if entry is not None
            else False
        )
        evaluation = _evaluation_summary(entry) if complete and entry is not None else {}
        runtime = _entry_runtime(entry) if complete and entry is not None else {}
        summary = _entry_summary(entry) if entry is not None else {}
        parameter_count = None
        if complete and entry is not None:
            parameter_count = (
                runtime.get("parameter_count")
                or summary.get("parameter_count")
                or (
                    summary.get("model_config", {}).get("parameter_count")
                    if isinstance(summary.get("model_config"), Mapping)
                    else None
                )
                or entry.get("parameter_count")
            )
        seed_values.append(
            {
                "training_seed": seed,
                "mean_return": _finite_metric(evaluation.get("mean_return")),
                "median_return": _finite_metric(evaluation.get("median_return")),
                "std_return": _finite_metric(evaluation.get("std_return")),
                "status": entry.get("status") if entry is not None else "missing",
                "eligible": complete,
                "steps": summary.get("total_transitions"),
                "sps": _finite_metric(runtime.get("steps_per_second")),
                "wall_clock_seconds": _finite_metric(runtime.get("wall_clock_seconds")),
                "peak_allocated_vram_bytes": _finite_metric(
                    runtime.get("cuda_peak_allocated_bytes")
                    or runtime.get("peak_allocated_vram_bytes")
                ),
                "parameter_count": _finite_metric(parameter_count),
                "entry": entry,
            }
        )
    quality_values = [value["mean_return"] for value in seed_values if value["mean_return"] is not None]
    sps_values = [value["sps"] for value in seed_values if value["sps"] is not None]
    wall_values = [value["wall_clock_seconds"] for value in seed_values if value["wall_clock_seconds"] is not None]
    vram_values = [value["peak_allocated_vram_bytes"] for value in seed_values if value["peak_allocated_vram_bytes"] is not None]
    parameter_values = [value["parameter_count"] for value in seed_values if value["parameter_count"] is not None]
    if parameter_values:
        parameter_count: float | list[float] | None = (
            parameter_values[0]
            if len(set(parameter_values)) == 1
            else parameter_values
        )
        parameter_count_source = "run_runtime_or_summary"
    else:
        model = build_q_network(
            family.architecture,
            num_actions=4,
            input_shape=(4, 84, 84),
        )
        parameter_count = float(sum(parameter.numel() for parameter in model.parameters()))
        parameter_count_source = "model_factory_contract"
    return {
        "family_id": family.family_id,
        "label": family.label,
        "algorithm": family.algorithm,
        "architecture": family.architecture,
        "stage": selected_stage,
        "target_transitions": selected_target,
        "formal_entry_count": sum(value["eligible"] for value in seed_values),
        "training_seed_count": len(seed_values),
        "seed_values": [
            {key: value for key, value in item.items() if key != "entry"}
            for item in seed_values
        ],
        "quality_mean": fmean(quality_values) if quality_values else None,
        "quality_median": median(quality_values) if quality_values else None,
        "quality_seed_spread": pstdev(quality_values) if len(quality_values) > 1 else 0.0 if quality_values else None,
        "mean_sps": fmean(sps_values) if sps_values else None,
        "mean_wall_clock_seconds": fmean(wall_values) if wall_values else None,
        "mean_peak_allocated_vram_bytes": fmean(vram_values) if vram_values else None,
        "parameter_count": parameter_count,
        "parameter_count_source": parameter_count_source,
        "all_formal_entries_complete": all(value["eligible"] for value in seed_values),
    }


def _strip_entry(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strip_entry(item) for key, item in value.items() if key not in {"metrics", "entry"}}
    if isinstance(value, list):
        return [_strip_entry(item) for item in value]
    return value


def select_top_candidates(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 2,
) -> list[str]:
    """Return top families by aggregate quality, never by one seed."""

    limit = _positive_int(max_candidates, name="max_candidates")
    ordered = sorted(
        (item for item in aggregates if _finite_metric(item.get("quality_mean")) is not None),
        key=lambda item: (
            -float(item["quality_mean"]),
            (
                -float(item["quality_median"])
                if _finite_metric(item.get("quality_median")) is not None
                else math.inf
            ),
            (
                float(item["quality_seed_spread"])
                if _finite_metric(item.get("quality_seed_spread")) is not None
                else math.inf
            ),
            str(item.get("family_id")),
        ),
    )
    return [str(item["family_id"]) for item in ordered[:limit]]


def _pairwise_seed_differences(
    winner: Mapping[str, Any],
    other: Mapping[str, Any],
) -> list[float | None]:
    other_by_seed = {
        item["training_seed"]: item.get("mean_return")
        for item in other.get("seed_values", [])
        if isinstance(item, Mapping)
    }
    return [
        None
        if item.get("mean_return") is None or other_by_seed.get(item.get("training_seed")) is None
        else float(item["mean_return"]) - float(other_by_seed[item["training_seed"]])
        for item in winner.get("seed_values", [])
        if isinstance(item, Mapping)
    ]


def _selection_for_aggregates(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    config: Day20ExperimentConfig,
    random_mean: float | None,
    required_family_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    required_ids = tuple(required_family_ids or config.family_ids)
    available = [
        item
        for item in aggregates
        if item.get("family_id") in required_ids
        and item.get("all_formal_entries_complete")
    ]
    if len(available) != len(required_ids):
        return {
            "status": "incomplete",
            "final_training_family": None,
            "deployment_candidate": None,
            "ordered_families": select_top_candidates(available, max_candidates=len(required_ids)),
            "reason": "all required families need complete CUDA evaluation evidence at their comparison horizon",
        }
    ordered = sorted(
        available,
        key=lambda item: (
            -float(item["quality_mean"]),
            -float(item["quality_median"]),
            float(item["quality_seed_spread"]),
            -float(item.get("mean_sps") or -math.inf),
            str(item["family_id"]),
        ),
    )
    winner = ordered[0]
    runner_up = ordered[1]
    differences = _pairwise_seed_differences(winner, runner_up)
    finite_differences = [value for value in differences if value is not None]
    direction_consistent = bool(finite_differences) and all(value > 0 for value in finite_differences)
    winner_spread = float(winner.get("quality_seed_spread") or 0.0)
    runner_spread = float(runner_up.get("quality_seed_spread") or 0.0)
    quality_gap = float(winner["quality_mean"]) - float(runner_up["quality_mean"])
    strongly_overlapping = abs(quality_gap) <= max(winner_spread, runner_spread, 0.0)
    extension_triggered = strongly_overlapping or not direction_consistent
    above_random = random_mean is not None and float(winner["quality_mean"]) > random_mean
    return {
        "status": "complete",
        "final_training_family": winner["family_id"],
        "deployment_candidate": winner["family_id"] if above_random else None,
        "deployment_ready": above_random,
        "ordered_families": [item["family_id"] for item in ordered],
        "winner_quality_mean": winner["quality_mean"],
        "runner_up_quality_mean": runner_up["quality_mean"],
        "quality_gap_to_runner_up": quality_gap,
        "winner_vs_runner_up_seed_differences": differences,
        "winner_beats_runner_up_on_every_seed": direction_consistent,
        "random_baseline_mean": random_mean,
        "winner_above_random_baseline": above_random,
        "decision_rule": (
            "rank complete families by multi-seed fixed evaluation mean; break ties "
            "with median, seed spread, correctness/healthy evidence, then measured "
            "engineering cost. Never use best single seed or training peak."
        ),
        "extension": {
            "target_transitions": 1_000_000,
            "triggered": extension_triggered,
            "top_candidates": [item["family_id"] for item in ordered[:2]],
            "reasons": [
                reason
                for reason, active in (
                    ("aggregate evaluation distributions overlap", strongly_overlapping),
                    ("winner direction is not consistent across all paired seeds", not direction_consistent),
                )
                if active
            ],
        },
    }


def build_day20_report(
    manifest_path: str | Path,
    *,
    config: Day20ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Aggregate only completed, eligible evidence into a selection report."""

    source = Path(manifest_path).resolve()
    manifest = read_day20_manifest(source)
    config = config or load_day20_config(require_probe_states=False)
    entries = [dict(entry) for entry in manifest["runs"] if isinstance(entry, Mapping)]
    for entry in entries:
        if (
            entry.get("stage") in {config.formal_quality_horizon, DAY20_EXTENSION_STAGE}
            and entry.get("status") in {"completed", "reused"}
        ):
            valid, computed_summary, validation_error = _validated_entry_artifacts(
                source,
                entry,
                config=config,
            )
            entry["_artifact_valid"] = valid
            entry["artifact_validation"] = {
                "status": "valid" if valid else "invalid",
                "error": validation_error,
            }
            if valid and computed_summary is not None:
                evaluation = dict(entry.get("evaluation", {}))
                evaluation["summary"] = computed_summary
                entry["evaluation"] = evaluation
    aggregates = [
        _aggregate_family(family, entries, config=config)
        for family in config.families
    ]
    random_mean: float | None = None
    random_reference = manifest.get("random_baseline_results")
    random_path = resolve_manifest_reference(source, random_reference) if isinstance(random_reference, str) else None
    random_validation_error: str | None = None
    if random_path is not None and random_path.is_file():
        try:
            random_summary = _validated_evaluation_summary(
                random_path,
                config=config,
                require_cuda=False,
            )
            random_mean = _finite_metric(random_summary.get("mean_return"))
        except (OSError, TypeError, ValueError, KeyError) as error:
            random_validation_error = str(error)
            random_mean = None
    elif random_reference is None:
        random_validation_error = "random baseline reference is missing"
    else:
        random_validation_error = "random baseline results.json is missing"

    training_summary = []
    evaluation_summary = []
    q_summary = []
    for entry in entries:
        training_summary.append(
            {
                "run_id": entry.get("run_id"),
                "family_id": entry.get("family_id"),
                "algorithm": entry.get("algorithm"),
                "architecture": entry.get("architecture"),
                "training_seed": entry.get("training_seed"),
                "stage": entry.get("stage"),
                "target_transitions": entry.get("target_transitions"),
                "status": entry.get("status"),
                "eligible": (
                    entry.get("eligible") is True
                    and entry.get("_artifact_valid", True) is not False
                ),
                "artifact_validation": _strip_entry(entry.get("artifact_validation")),
                "summary": _strip_entry(_entry_summary(entry)),
                "runtime": _strip_entry(_entry_runtime(entry)),
                "metrics_path": (
                    entry.get("training", {}).get("metrics_path")
                    if isinstance(entry.get("training"), Mapping)
                    else None
                ),
                "source": _strip_entry(entry.get("source")),
            }
        )
        evaluation = entry.get("evaluation", {})
        if isinstance(evaluation, Mapping):
            evaluation_summary.append(
                {
                    "run_id": entry.get("run_id"),
                    "family_id": entry.get("family_id"),
                    "training_seed": entry.get("training_seed"),
                    "stage": entry.get("stage"),
                    "target_transitions": entry.get("target_transitions"),
                    "status": evaluation.get("status"),
                    "eligible": (
                        entry.get("eligible") is True
                        and entry.get("_artifact_valid", True) is not False
                        and evaluation.get("summary") is not None
                    ),
                    "summary": _strip_entry(evaluation.get("summary", {})),
                }
            )
        probe = entry.get("q_probe")
        if isinstance(probe, Mapping):
            q_summary.append(
                {
                    "run_id": entry.get("run_id"),
                    "family_id": entry.get("family_id"),
                    "training_seed": entry.get("training_seed"),
                    "stage": entry.get("stage"),
                    "target_transitions": entry.get("target_transitions"),
                    "status": probe.get("status"),
                    "summary": _strip_entry(probe.get("summary", {})),
                }
            )

    base_selection = _selection_for_aggregates(
        aggregates,
        config=config,
        random_mean=random_mean,
    )
    base_extension = base_selection.get("extension", {})
    base_top_candidates = (
        [str(item) for item in base_extension.get("top_candidates", [])]
        if isinstance(base_extension, Mapping)
        and isinstance(base_extension.get("top_candidates"), list)
        else []
    )
    growth_unresolved = (
        base_selection.get("status") == "complete"
        and _healthy_growth_unresolved(
            source,
            entries,
            config=config,
            family_ids=base_top_candidates,
        )
    )
    if growth_unresolved and isinstance(base_extension, Mapping):
        growth_extension = dict(base_extension)
        growth_extension["triggered"] = True
        growth_extension["growth_unresolved"] = True
        reasons = list(growth_extension.get("reasons", []))
        reasons.append("eligible top-candidate training curves still show recent growth")
        growth_extension["reasons"] = reasons
        base_selection["extension"] = growth_extension
    base_extension = base_selection.get("extension", {})
    extension_triggered = (
        isinstance(base_extension, Mapping)
        and base_extension.get("triggered") is True
    )
    extension_top_candidates = (
        [str(item) for item in base_extension.get("top_candidates", [])]
        if isinstance(base_extension, Mapping)
        and isinstance(base_extension.get("top_candidates"), list)
        else []
    )
    extension_entries = [
        entry
        for entry in entries
        if entry.get("stage") == DAY20_EXTENSION_STAGE
        and entry.get("family_id") in extension_top_candidates
    ]
    extension_aggregates = [
        _aggregate_family(
            family,
            extension_entries,
            config=config,
            stage=DAY20_EXTENSION_STAGE,
            target=DAY20_EXTENSION_TARGET,
        )
        for family in config.families
        if family.family_id in extension_top_candidates
    ]
    extension_expected_entry_count = len(extension_top_candidates) * len(config.training_seeds)
    extension_completed_entry_count = sum(
        int(item.get("formal_entry_count", 0)) for item in extension_aggregates
    )
    extension_selection: dict[str, Any] | None = None
    extension_applied = False
    if extension_triggered and extension_top_candidates:
        extension_selection = _selection_for_aggregates(
            extension_aggregates,
            config=config,
            random_mean=random_mean,
            required_family_ids=extension_top_candidates,
        )
        extension_applied = extension_selection.get("status") == "complete"
    extension_report = {
        "status": (
            "complete"
            if extension_applied
            else "pending"
            if extension_triggered
            else "not_triggered"
        ),
        "target_transitions": DAY20_EXTENSION_TARGET,
        "triggered": extension_triggered,
        "growth_unresolved": growth_unresolved,
        "applied": extension_applied,
        "top_candidates": extension_top_candidates,
        "expected_entry_count": extension_expected_entry_count,
        "completed_entry_count": extension_completed_entry_count,
        "aggregates": extension_aggregates,
        "selection": extension_selection,
    }
    if extension_applied and extension_selection is not None:
        selection = dict(extension_selection)
        selection["base_500k"] = base_selection
        selection["selection_horizon"] = DAY20_EXTENSION_TARGET
        selection["extension"] = {
            **extension_report,
            "selection": extension_selection,
        }
    else:
        selection = dict(base_selection)
        selection["selection_horizon"] = config.formal_steps
        selection["extension"] = extension_report
    formal_entries = [entry for entry in entries if entry.get("stage") == config.formal_quality_horizon]
    conditions = {
        "expected_family_count": len(config.families),
        "expected_training_seed_count": len(config.training_seeds),
        "formal_completed_entry_count": sum(_formal_entry(entry, config=config) for entry in formal_entries),
        "formal_expected_entry_count": len(config.families) * len(config.training_seeds),
        "formal_cuda_only": all(
            _entry_runtime(entry).get("requested_device") == "cuda"
            and str(_entry_runtime(entry).get("resolved_device", "")).startswith("cuda:")
            for entry in formal_entries
            if entry.get("status") in {"completed", "reused"}
        ),
        "screening_and_pilot_are_not_final_selection": True,
        "quality_budget": "actual accepted environment transitions",
        "quality_and_engineering_cost_are_separate": True,
        "incomplete_runs_excluded_from_aggregate": True,
        "extension_triggered_by_500k_rule": extension_triggered,
        "extension_completed_entry_count": extension_completed_entry_count,
        "extension_expected_entry_count": extension_expected_entry_count,
        "extension_complete_before_final_selection": extension_applied,
    }
    return {
        "schema_version": DAY20_SCHEMA_VERSION,
        "generated_at_utc": utc_timestamp(),
        "question": (
            "在同一 Contract v2、Day 16 CUDA backend、paired seeds 與 500K "
            "actual environment transitions 下，哪個 DQN family 最值得進入 Final Long Training？"
        ),
        "manifest": relative_path(source, start=config.repository_root),
        "manifest_sha256": sha256_file(source),
        "manifest_status": manifest.get("status"),
        "protocol": config.protocol(),
        "source_of_truth": manifest.get("source_of_truth", {}),
        "provenance": manifest.get("provenance", {}),
        "evidence_reuse": manifest.get("evidence_reuse", {}),
        "training": {
            "entries": training_summary,
            "completed_entry_count": sum(item["eligible"] is True for item in training_summary),
        },
        "evaluation": {
            "entries": evaluation_summary,
            "completed_entry_count": sum(item["eligible"] is True for item in evaluation_summary),
        },
        "q_probe": {
            "entries": q_summary,
            "completed_entry_count": sum(
                item.get("status") in {"completed", "reused"} for item in q_summary
            ),
        },
        "aggregates": aggregates,
        "engineering_cost": {
            "source": "completed formal entries only",
            "families": [
                {
                    "family_id": item["family_id"],
                    "parameter_count": item.get("parameter_count"),
                    "mean_sps": item.get("mean_sps"),
                    "mean_wall_clock_seconds": item.get("mean_wall_clock_seconds"),
                    "mean_peak_allocated_vram_bytes": item.get(
                        "mean_peak_allocated_vram_bytes"
                    ),
                }
                for item in aggregates
            ],
        },
        "random_baseline": {
            "path": random_reference,
            "mean_return": random_mean,
            "status": "valid" if random_mean is not None else "invalid",
            "validation_error": random_validation_error,
        },
        "comparison_conditions": conditions,
        "selection": selection,
        "selection_horizon": (
            DAY20_EXTENSION_TARGET if extension_applied else config.formal_steps
        ),
        "extension": extension_report,
    }


__all__ = [
    "DAY20_COMMON_CONFIG_FIELDS",
    "DAY20_FAMILIES",
    "DAY20_FAMILY_IDS",
    "DAY20_FORMAL_STAGE",
    "DAY20_EXTENSION_STAGE",
    "DAY20_EXTENSION_TARGET",
    "DAY20_MILESTONES",
    "DAY20_REQUIRED_TRAINING_METRICS",
    "DAY20_SCHEMA_VERSION",
    "DAY20_TRAINING_SEEDS",
    "DQNFamily",
    "Day20ExperimentConfig",
    "apply_day18_reuse",
    "audit_day18_evidence_reuse",
    "build_day20_manifest",
    "build_day20_report",
    "config_diff",
    "load_day20_config",
    "read_day20_manifest",
    "read_metrics",
    "relative_path",
    "resolve_manifest_reference",
    "select_top_candidates",
    "sha256_file",
    "utc_timestamp",
    "validate_day20_manifest",
    "write_json",
]
