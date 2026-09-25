"""Run Issue #10 bootstrap uncertainty analysis on Issue #9 artifacts.

The analysis keeps the experimental hierarchy explicit:

* evaluation seeds are paired observations within one trained-model pair;
* training seeds are the independent experimental units for cross-seed claims;
* hierarchical bootstrap resamples both levels without pooling the 150 episodes.

This module intentionally has no training or model-loading path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)


BASELINE_LABEL = "penalty-0.0"
SHAPED_LABEL = "penalty-minus-1.0"
CHECKPOINT_STEP = 1_000_000
REQUIRED_TRAINING_SEEDS = (2022, 2023, 2024)
EXPECTED_EVALUATION_SEEDS = tuple(range(1001, 1051))
DEFAULT_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260916
DEFAULT_CONTRACT_PATH = Path("configs/eval/breakout_contract_v2.json")
EXPECTED_SCORE_DEFINITION = (
    "raw Atari game reward sum; no clipping and no life-loss penalty"
)
ALLOWED_RAW_SCORE_DEFINITIONS = frozenset(
    {
        EXPECTED_SCORE_DEFINITION,
        "raw Atari game reward sum; no training shaping",
    }
)
DEFAULT_EVALUATION_DIRS = {
    2022: Path("experiments/issue-9-reward-shaping/stage3-1m/evaluation"),
    2023: Path(
        "experiments/issue-9-reward-shaping/stage3b-multiseed/seed2023/evaluation"
    ),
    2024: Path(
        "experiments/issue-9-reward-shaping/stage3b-multiseed/seed2024/evaluation"
    ),
}


@dataclass(frozen=True)
class PairedRecord:
    """One paired raw-score observation for one training seed."""

    training_seed: int
    evaluation_seed: int
    baseline_raw_score: float
    shaped_raw_score: float

    @property
    def delta(self) -> float:
        return self.shaped_raw_score - self.baseline_raw_score


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _contract_sha256(contract: BreakoutEvaluationContractV2) -> str:
    canonical = json.dumps(
        contract.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalise_relative_path(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty path")
    normalised = value.strip().replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    return normalised


def _load_contract_metadata(
    contract_path: Path,
) -> tuple[dict[str, Any], str]:
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    canonical_path = contract_path.as_posix()
    return (
        {
            "path": canonical_path,
            "sha256": _contract_sha256(contract),
            "schema_version": contract.schema_version,
            "contract_id": contract.contract_id,
            "environment_id": contract.environment_id,
            "raw_reward_rule": contract.raw_reward_rule,
        },
        _normalise_relative_path(canonical_path, path="contract path"),
    )


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be a finite number")
    return parsed


def _require_raw_score_definition(value: Any, *, path: str) -> None:
    if value not in ALLOWED_RAW_SCORE_DEFINITIONS:
        raise ValueError(f"{path} is not an allowed raw-score definition")


def _seed_from_components(base_seed: int, *components: int) -> int:
    """Derive and return a reproducible uint32 seed for one analysis level."""

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed <= 2**32 - 1
    ):
        raise ValueError("bootstrap seed must be an unsigned 32-bit integer")
    sequence = np.random.SeedSequence([base_seed, *components])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _validate_iterations(iterations: int) -> None:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < DEFAULT_ITERATIONS
    ):
        raise ValueError("Issue #10 requires at least 10000 bootstrap iterations")


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    quantiles = np.percentile(values, [2.5, 97.5])
    return {"lower": float(quantiles[0]), "upper": float(quantiles[1])}


def _effect_summary(
    *,
    deltas: Sequence[float],
    bootstrap_means: np.ndarray,
    bootstrap_medians: np.ndarray,
    iterations: int,
    rng_seed: int,
    sample_size: int,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    wins = int(np.sum(values > 0))
    ties = int(np.sum(values == 0))
    losses = int(np.sum(values < 0))
    return {
        "n_eval": sample_size,
        "point_mean_delta": float(np.mean(values)),
        "point_median_delta": float(np.median(values)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_iterations": iterations,
        "bootstrap_rng_seed": rng_seed,
        "mean_delta_95_ci": _percentile_interval(bootstrap_means),
        "median_delta_95_ci": _percentile_interval(bootstrap_medians),
        "p_bootstrap_mean_effect_gt_zero": float(np.mean(bootstrap_means > 0)),
        "p_bootstrap_median_effect_gt_zero": float(
            np.mean(bootstrap_medians > 0)
        ),
        "bootstrap_mean_std": float(np.std(bootstrap_means, ddof=1)),
        "bootstrap_mean_se": float(np.std(bootstrap_means, ddof=1)),
    }


def validate_paired_records(
    records: Sequence[PairedRecord],
    *,
    training_seed: int,
    expected_evaluation_seeds: Sequence[int] = EXPECTED_EVALUATION_SEEDS,
    source: str = "paired records",
) -> list[PairedRecord]:
    """Validate already-parsed pairs without dropping rows silently."""

    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError(f"{source}: training_seed must be an integer")
    expected = tuple(int(seed) for seed in expected_evaluation_seeds)
    if len(expected) != len(set(expected)):
        raise ValueError(f"{source}: expected evaluation seeds are not unique")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"{source} must be a sequence")

    seen: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, PairedRecord):
            raise ValueError(f"{source}[{index}] must be a PairedRecord")
        if record.training_seed != training_seed:
            raise ValueError(
                f"{source}[{index}]: training_seed does not match its group"
            )
        if isinstance(record.evaluation_seed, bool) or not isinstance(
            record.evaluation_seed, int
        ):
            raise ValueError(
                f"{source}[{index}].evaluation_seed must be an integer"
            )
        if record.evaluation_seed in seen:
            raise ValueError(
                f"{source}: duplicate evaluation seed {record.evaluation_seed}"
            )
        seen.add(record.evaluation_seed)
        _finite_float(
            record.baseline_raw_score,
            path=f"{source}[{index}].baseline_raw_score",
        )
        _finite_float(
            record.shaped_raw_score,
            path=f"{source}[{index}].shaped_raw_score",
        )
        _finite_float(record.delta, path=f"{source}[{index}].delta")

    missing = sorted(set(expected) - seen)
    unexpected = sorted(seen - set(expected))
    if missing:
        raise ValueError(f"{source}: missing paired evaluation seeds {missing}")
    if unexpected:
        raise ValueError(f"{source}: unexpected evaluation seeds {unexpected}")
    if len(records) != len(expected):
        raise ValueError(
            f"{source}: expected {len(expected)} complete pairs, got {len(records)}"
        )
    return sorted(records, key=lambda record: record.evaluation_seed)


def validate_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_seed: int,
    expected_evaluation_seeds: Sequence[int] = EXPECTED_EVALUATION_SEEDS,
    source: str = "paired rows",
) -> list[PairedRecord]:
    """Validate and parse paired raw scores without dropping any row silently."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{source}: paired_rows must be a sequence")
    records: list[PairedRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}[{index}] must be an object")
        raw_seed = row.get("evaluation_seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError(f"{source}[{index}].evaluation_seed must be an integer")
        baseline = _finite_float(
            row.get("baseline_raw_score"),
            path=f"{source}[{index}].baseline_raw_score",
        )
        shaped = _finite_float(
            row.get("shaped_raw_score"),
            path=f"{source}[{index}].shaped_raw_score",
        )
        observed_delta = row.get("score_delta")
        if observed_delta is not None:
            parsed_delta = _finite_float(
                observed_delta,
                path=f"{source}[{index}].score_delta",
            )
            if not math.isclose(parsed_delta, shaped - baseline, abs_tol=1e-9):
                raise ValueError(
                    f"{source}[{index}]: score_delta does not equal shaped-baseline"
                )
        records.append(
            PairedRecord(
                training_seed=training_seed,
                evaluation_seed=raw_seed,
                baseline_raw_score=baseline,
                shaped_raw_score=shaped,
            )
        )
    return validate_paired_records(
        records,
        training_seed=training_seed,
        expected_evaluation_seeds=expected_evaluation_seeds,
        source=source,
    )


def _validate_artifact_metadata(
    payload: Mapping[str, Any],
    *,
    training_seed: int,
    summary_path: Path,
    contract_metadata: Mapping[str, Any],
    canonical_contract_path: str,
) -> None:
    """Validate provenance needed to interpret an Issue #9 evaluation summary."""

    if payload.get("training_transitions") != CHECKPOINT_STEP:
        raise ValueError(
            f"{summary_path}: training_transitions must be {CHECKPOINT_STEP}"
        )
    checkpoint_steps = payload.get("checkpoint_steps")
    if (
        not isinstance(checkpoint_steps, Sequence)
        or isinstance(checkpoint_steps, (str, bytes))
        or CHECKPOINT_STEP not in checkpoint_steps
    ):
        raise ValueError(
            f"{summary_path}: checkpoint_steps must include {CHECKPOINT_STEP}"
        )
    if payload.get("contract_id") != contract_metadata["contract_id"]:
        raise ValueError(f"{summary_path}: contract_id does not match Contract v2")
    artifact_contract_path = _normalise_relative_path(
        payload.get("contract_path"),
        path=f"{summary_path}: contract_path",
    )
    if artifact_contract_path.casefold() != canonical_contract_path.casefold():
        raise ValueError(f"{summary_path}: contract_path does not match Contract v2")
    top_level_hash = payload.get("contract_sha256")
    if top_level_hash is not None and top_level_hash != contract_metadata["sha256"]:
        raise ValueError(f"{summary_path}: contract_sha256 does not match Contract v2")

    variants = payload.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise ValueError(f"{summary_path}: variants are missing")
    for label in (BASELINE_LABEL, SHAPED_LABEL):
        matching = [
            variant
            for variant in variants
            if isinstance(variant, Mapping) and variant.get("label") == label
        ]
        if len(matching) != 1:
            raise ValueError(
                f"{summary_path}: expected exactly one {label} variant"
            )
        variant = matching[0]
        variant_path = _normalise_relative_path(
            variant.get("contract_path"),
            path=f"{summary_path}: {label}.contract_path",
        )
        if variant_path.casefold() != canonical_contract_path.casefold():
            raise ValueError(
                f"{summary_path}: {label} contract_path does not match Contract v2"
            )
        if variant.get("contract_sha256") != contract_metadata["sha256"]:
            raise ValueError(
                f"{summary_path}: {label} contract_sha256 does not match Contract v2"
            )
        config = variant.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"{summary_path}: {label} config is missing")
        expected_penalty = 0.0 if label == BASELINE_LABEL else -1.0
        if config.get("life_loss_penalty") != expected_penalty:
            raise ValueError(
                f"{summary_path}: {label} life_loss_penalty is not "
                f"{expected_penalty}"
            )
        if config.get("seed") != training_seed:
            raise ValueError(
                f"{summary_path}: {label} config seed does not match training_seed"
            )
        if config.get("total_steps") != CHECKPOINT_STEP:
            raise ValueError(
                f"{summary_path}: {label} config total_steps must be "
                f"{CHECKPOINT_STEP}"
            )
        if config.get("algorithm") != "double_dqn" or config.get("architecture") != "dueling":
            raise ValueError(
                f"{summary_path}: {label} is not the expected Dueling Double DQN"
            )


def _record_signature(
    records: Sequence[PairedRecord],
) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (record.evaluation_seed, record.baseline_raw_score, record.shaped_raw_score)
        for record in records
    )


def _validate_paired_evaluation_artifact(
    payload: Mapping[str, Any],
    *,
    summary_path: Path,
    expected_records: Sequence[PairedRecord],
) -> None:
    """Cross-check the standalone 1M paired-evaluation artifact."""

    paired = payload.get(SHAPED_LABEL)
    if not isinstance(paired, Mapping):
        raise ValueError(
            f"{summary_path}: standalone -1.0 paired evaluation is missing"
        )
    if paired.get("evaluation_seed_count") != len(EXPECTED_EVALUATION_SEEDS):
        raise ValueError(f"{summary_path}: standalone paired evaluation count must be 50")
    baseline = paired.get("baseline")
    shaped = paired.get("shaped")
    if not isinstance(baseline, Mapping) or not isinstance(shaped, Mapping):
        raise ValueError(
            f"{summary_path}: standalone paired evaluation is incomplete"
        )
    for label, result in (("baseline", baseline), ("shaped", shaped)):
        if result.get("episode_count") != len(EXPECTED_EVALUATION_SEEDS):
            raise ValueError(
                f"{summary_path}: standalone {label} episode_count must be 50"
            )
        _require_raw_score_definition(
            result.get("score_definition"),
            path=f"{summary_path}: standalone {label} score definition",
        )
    paired_rows = paired.get("paired_rows")
    actual_records = validate_paired_rows(
        paired_rows if isinstance(paired_rows, Sequence) else [],
        training_seed=expected_records[0].training_seed,
        source=f"{summary_path}.{SHAPED_LABEL}.paired_rows",
    )
    if _record_signature(actual_records) != _record_signature(expected_records):
        raise ValueError(
            f"{summary_path}: standalone paired rows do not match sweep summary"
        )


def load_paired_artifacts(
    evaluation_dirs: Mapping[int, Path] = DEFAULT_EVALUATION_DIRS,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[dict[int, list[PairedRecord]], dict[int, str], dict[str, Any]]:
    """Load and validate Issue #9 final paired raw scores and provenance."""

    contract_metadata, canonical_contract_path = _load_contract_metadata(
        Path(contract_path)
    )
    records_by_training_seed: dict[int, list[PairedRecord]] = {}
    sources: dict[int, str] = {}
    expected_eval_seeds = list(EXPECTED_EVALUATION_SEEDS)
    for training_seed, evaluation_dir in sorted(evaluation_dirs.items()):
        summary_path = Path(evaluation_dir) / "sweep-summary.json"
        payload = _read_json(summary_path)
        if payload.get("training_seed") != training_seed:
            raise ValueError(
                f"{summary_path}: training_seed does not match {training_seed}"
            )
        _validate_artifact_metadata(
            payload,
            training_seed=training_seed,
            summary_path=summary_path,
            contract_metadata=contract_metadata,
            canonical_contract_path=canonical_contract_path,
        )
        if payload.get("eval_seeds") != expected_eval_seeds:
            raise ValueError(f"{summary_path}: evaluation seed list is not 1001..1050")
        if payload.get("evaluation_seed_count") != len(expected_eval_seeds):
            raise ValueError(f"{summary_path}: evaluation_seed_count must be 50")
        if payload.get("evaluation_episodes_per_seed") != 1:
            raise ValueError(f"{summary_path}: expected one episode per evaluation seed")
        if payload.get("evaluation_score_definition") != EXPECTED_SCORE_DEFINITION:
            raise ValueError(f"{summary_path}: score is not raw-score-only")
        comparisons = payload.get("comparisons")
        if not isinstance(comparisons, Mapping):
            raise ValueError(f"{summary_path}: comparisons are missing")
        checkpoint = comparisons.get(str(CHECKPOINT_STEP))
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{summary_path}: 1M comparison is missing")
        paired = checkpoint.get(SHAPED_LABEL)
        if not isinstance(paired, Mapping):
            raise ValueError(f"{summary_path}: -1.0 paired comparison is missing")
        baseline = paired.get("baseline")
        shaped = paired.get("shaped")
        if not isinstance(baseline, Mapping) or not isinstance(shaped, Mapping):
            raise ValueError(f"{summary_path}: baseline/shaped summary is incomplete")
        for label, result in (("baseline", baseline), ("shaped", shaped)):
            if result.get("episode_count") != len(expected_eval_seeds):
                raise ValueError(
                    f"{summary_path}: {label} episode_count must be 50"
                )
            raw_score = result.get("raw_score")
            if not isinstance(raw_score, Mapping):
                raise ValueError(f"{summary_path}: {label} raw_score is missing")
            if raw_score.get("count") != len(expected_eval_seeds):
                raise ValueError(
                    f"{summary_path}: {label} raw_score count must be 50"
                )
        _require_raw_score_definition(
            baseline.get("score_definition"),
            path=f"{summary_path}: baseline score definition",
        )
        _require_raw_score_definition(
            shaped.get("score_definition"),
            path=f"{summary_path}: shaped score definition",
        )
        paired_rows = paired.get("paired_rows")
        summary_records = validate_paired_rows(
            paired_rows if isinstance(paired_rows, Sequence) else [],
            training_seed=training_seed,
            expected_evaluation_seeds=expected_eval_seeds,
            source=f"{summary_path}.comparisons.1M.{SHAPED_LABEL}.paired_rows",
        )
        paired_evaluation_path = (
            Path(evaluation_dir) / f"paired-evaluation-{CHECKPOINT_STEP}.json"
        )
        standalone_payload = _read_json(paired_evaluation_path)
        _validate_paired_evaluation_artifact(
            standalone_payload,
            summary_path=paired_evaluation_path,
            expected_records=summary_records,
        )
        records_by_training_seed[training_seed] = summary_records
        sources[training_seed] = summary_path.as_posix()
    if set(records_by_training_seed) != set(evaluation_dirs):
        raise ValueError("loaded training seed set does not match requested seed set")
    contract_metadata["source_artifacts_include_contract_sha256"] = True
    return records_by_training_seed, sources, contract_metadata


def paired_bootstrap(
    records: Sequence[PairedRecord],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    rng_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Bootstrap paired deltas for one trained-model pair."""

    _validate_iterations(iterations)
    if not records:
        raise ValueError("paired bootstrap requires at least one pair")
    deltas = np.asarray([record.delta for record in records], dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    bootstrap_values = deltas[indices]
    means = np.mean(bootstrap_values, axis=1)
    medians = np.median(bootstrap_values, axis=1)
    summary = _effect_summary(
        deltas=deltas,
        bootstrap_means=means,
        bootstrap_medians=medians,
        iterations=iterations,
        rng_seed=rng_seed,
        sample_size=len(records),
    )
    return summary, means, medians


def training_seed_level_bootstrap(
    observed_effects: Mapping[int, float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    rng_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Bootstrap one observed effect per independent training seed."""

    _validate_iterations(iterations)
    seeds = tuple(sorted(observed_effects))
    if len(seeds) < 2:
        raise ValueError(
            "seed-level bootstrap requires at least two training seeds"
        )
    effects = np.asarray([observed_effects[seed] for seed in seeds], dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(effects), size=(iterations, len(effects)))
    values = effects[indices]
    means = np.mean(values, axis=1)
    medians = np.median(values, axis=1)
    summary = _effect_summary(
        deltas=effects,
        bootstrap_means=means,
        bootstrap_medians=medians,
        iterations=iterations,
        rng_seed=rng_seed,
        sample_size=len(seeds),
    )
    summary.update(
        {
            "n_eval": None,
            "n_effects": len(seeds),
            "n_train": len(seeds),
            "training_seed_ids": list(seeds),
            "independent_unit": "training_seed",
            "evaluation_episodes_not_pooled": True,
        }
    )
    return summary, means, medians


def hierarchical_bootstrap(
    records_by_training_seed: Mapping[int, Sequence[PairedRecord]],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    rng_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Bootstrap training seeds, then paired evaluation records within each seed."""

    _validate_iterations(iterations)
    training_seeds = tuple(sorted(records_by_training_seed))
    if len(training_seeds) < 2:
        raise ValueError(
            "hierarchical bootstrap requires at least two training seeds"
        )
    validated_records = {
        seed: validate_paired_records(
            records_by_training_seed[seed],
            training_seed=seed,
            source=f"training seed {seed} paired records",
        )
        for seed in training_seeds
    }
    deltas = []
    for seed in training_seeds:
        records = validated_records[seed]
        deltas.append(np.asarray([record.delta for record in records], dtype=np.float64))
    if len({len(values) for values in deltas}) != 1:
        raise ValueError("all training seeds must have the same paired evaluation count")
    n_train = len(deltas)
    n_eval = len(deltas[0])
    matrix = np.stack(deltas, axis=0)
    rng = np.random.default_rng(rng_seed)
    outer_indices = rng.integers(0, n_train, size=(iterations, n_train))
    inner_indices = rng.integers(0, n_eval, size=(iterations, n_train, n_eval))
    sampled = matrix[outer_indices[:, :, None], inner_indices]
    sampled_seed_effects = np.mean(sampled, axis=2)
    means = np.mean(sampled_seed_effects, axis=1)
    medians = np.median(sampled_seed_effects, axis=1)
    observed_effects = np.mean(matrix, axis=1)
    summary = _effect_summary(
        deltas=observed_effects,
        bootstrap_means=means,
        bootstrap_medians=medians,
        iterations=iterations,
        rng_seed=rng_seed,
        sample_size=n_eval,
    )
    summary.update(
        {
            "n_train": n_train,
            "n_eval_per_train": n_eval,
            "training_seed_ids": list(training_seeds),
            "independent_unit": "training_seed",
            "resampling": {
                "level_1": "training seeds with replacement",
                "level_2": "paired evaluation records within each sampled seed with replacement",
                "pooled_150_episode_bootstrap_used": False,
            },
        }
    )
    return summary, means, medians


def build_analysis(
    records_by_training_seed: Mapping[int, Sequence[PairedRecord]],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    sources: Mapping[int, str] | None = None,
    contract_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build all three bootstrap levels and a combined distribution table."""

    _validate_iterations(iterations)
    if len(records_by_training_seed) < 2:
        raise ValueError(
            "cross-training-seed bootstrap requires at least two training seeds"
        )
    missing_required_seeds = sorted(
        set(REQUIRED_TRAINING_SEEDS) - set(records_by_training_seed)
    )
    if missing_required_seeds:
        raise ValueError(
            "Issue #10 requires current training seeds 2022, 2023, and 2024; "
            f"missing {missing_required_seeds}"
        )
    ordered_seeds = tuple(sorted(records_by_training_seed))
    validated_records = {
        training_seed: validate_paired_records(
            records_by_training_seed[training_seed],
            training_seed=training_seed,
            source=f"training seed {training_seed} paired records",
        )
        for training_seed in ordered_seeds
    }
    per_seed: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    for training_seed in ordered_seeds:
        rng_seed = _seed_from_components(bootstrap_seed, 1, training_seed)
        result, means, medians = paired_bootstrap(
            validated_records[training_seed],
            iterations=iterations,
            rng_seed=rng_seed,
        )
        baseline_scores = [
            record.baseline_raw_score
            for record in validated_records[training_seed]
        ]
        shaped_scores = [
            record.shaped_raw_score
            for record in validated_records[training_seed]
        ]
        result["training_seed"] = training_seed
        result["baseline_mean_raw_score"] = float(np.mean(baseline_scores))
        result["shaped_mean_raw_score"] = float(np.mean(shaped_scores))
        result["baseline_median_raw_score"] = float(np.median(baseline_scores))
        result["shaped_median_raw_score"] = float(np.median(shaped_scores))
        result["source_artifact"] = None if sources is None else sources.get(training_seed)
        per_seed.append(result)
        for iteration, (effect_mean, effect_median) in enumerate(zip(means, medians)):
            distributions.append(
                {
                    "analysis_level": "per_seed_paired",
                    "training_seed": training_seed,
                    "iteration": iteration,
                    "effect_mean": float(effect_mean),
                    "effect_median": float(effect_median),
                    "bootstrap_rng_seed": rng_seed,
                    "n_train": "",
                    "n_eval_per_train": len(validated_records[training_seed]),
                }
            )

    observed_effects = {
        seed: float(np.mean([record.delta for record in validated_records[seed]]))
        for seed in ordered_seeds
    }
    seed_rng_seed = _seed_from_components(bootstrap_seed, 2)
    seed_level, seed_means, seed_medians = training_seed_level_bootstrap(
        observed_effects,
        iterations=iterations,
        rng_seed=seed_rng_seed,
    )
    for iteration, (effect_mean, effect_median) in enumerate(
        zip(seed_means, seed_medians)
    ):
        distributions.append(
            {
                "analysis_level": "training_seed_level",
                "training_seed": "",
                "iteration": iteration,
                "effect_mean": float(effect_mean),
                "effect_median": float(effect_median),
                "bootstrap_rng_seed": seed_rng_seed,
                "n_train": len(observed_effects),
                "n_eval_per_train": "",
            }
        )

    hierarchical_rng_seed = _seed_from_components(bootstrap_seed, 3)
    hierarchical, hierarchical_means, hierarchical_medians = hierarchical_bootstrap(
        validated_records,
        iterations=iterations,
        rng_seed=hierarchical_rng_seed,
    )
    for iteration, (effect_mean, effect_median) in enumerate(
        zip(hierarchical_means, hierarchical_medians)
    ):
        distributions.append(
            {
                "analysis_level": "hierarchical",
                "training_seed": "",
                "iteration": iteration,
                "effect_mean": float(effect_mean),
                "effect_median": float(effect_median),
                "bootstrap_rng_seed": hierarchical_rng_seed,
                "n_train": len(records_by_training_seed),
                "n_eval_per_train": len(next(iter(validated_records.values()))),
            }
        )

    summary = {
        "schema_version": 1,
        "artifact_type": "issue10_reward_shaping_bootstrap_summary",
        "analysis": "paired_per_seed_and_training_seed_hierarchical_bootstrap",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "training_seed_ids": list(ordered_seeds),
        "n_train": len(ordered_seeds),
        "n_eval_per_train": len(next(iter(validated_records.values()))),
        "contract": None if contract_metadata is None else dict(contract_metadata),
        "pairing": {
            "pair_key": "evaluation_seed",
            "baseline_label": BASELINE_LABEL,
            "shaped_label": SHAPED_LABEL,
            "delta_definition": "shaped_raw_score - baseline_raw_score",
            "pairs_validated": True,
            "missing_or_duplicate_pairs_allowed": False,
        },
        "cross_training_seed_inference": {
            "independent_unit": "training_seed",
            "pooled_150_episode_bootstrap_used": False,
            "pseudo_replication_used": False,
            "n_train_limitation": (
                f"Only n_train={len(ordered_seeds)} training seeds are available; "
                "percentile bootstrap intervals are coarse and should not be "
                "treated as definitive population-level evidence."
            ),
            "observed_effect_range": {
                "minimum": float(min(observed_effects.values())),
                "maximum": float(max(observed_effects.values())),
                "span": float(
                    max(observed_effects.values()) - min(observed_effects.values())
                ),
            },
            "observed_effect_variance_population": float(
                np.var(list(observed_effects.values()))
            ),
        },
        "per_seed_paired_bootstrap": per_seed,
        "training_seed_level_bootstrap": {
            **seed_level,
            "observed_training_seed_effects": observed_effects,
        },
        "hierarchical_bootstrap": hierarchical,
        "statistical_limitations": [
            "Evaluation episodes quantify variability for a trained-model pair, not independent training replicates.",
            f"Cross-training-seed inference has n_train={len(ordered_seeds)}.",
            "One evaluation episode is used per evaluation seed.",
            "Bootstrap percentile intervals inherit the discrete/coarse outer resampling distribution at the observed training-seed count.",
            "A CI crossing zero is not evidence of a statistically established improvement.",
        ],
    }
    return summary, distributions


def render_analysis_report(summary: Mapping[str, Any]) -> str:
    """Render a concise, evidence-bearing Markdown report."""

    per_seed = summary["per_seed_paired_bootstrap"]
    seed_level = summary["training_seed_level_bootstrap"]
    hierarchical = summary["hierarchical_bootstrap"]
    cross_seed_mean_effect = seed_level["point_mean_delta"]
    if cross_seed_mean_effect > 0:
        cross_seed_direction = "positive"
        cross_seed_interpretation = (
            "suggestive positive point estimate, but not statistically established "
            "generalization"
        )
    elif cross_seed_mean_effect < 0:
        cross_seed_direction = "negative"
        cross_seed_interpretation = (
            "negative point estimate; no evidence of a positive generalization effect"
        )
    else:
        cross_seed_direction = "zero"
        cross_seed_interpretation = (
            "zero point estimate; no evidence of a positive generalization effect"
        )
    lines = [
        "# Issue #10 - Reward Shaping Bootstrap Analysis",
        "",
        "This report uses Issue #9 structured paired raw-score artifacts only. No training or 2.5M run was started.",
        "",
        "## Design",
        "",
        f"- Bootstrap iterations: `{summary['bootstrap_iterations']}`",
        f"- Bootstrap seed: `{summary['bootstrap_seed']}`",
        f"- Independent cross-seed unit: `training_seed` (`n_train={summary['n_train']}`)",
        f"- Paired evaluation count per training seed: `{summary['n_eval_per_train']}`",
        "- Delta: `shaped_raw_score - baseline_raw_score`",
        "- Pooled 150-episode bootstrap: **not used**",
        "",
        "## Per-training-seed paired bootstrap",
        "",
        "| Training seed | Baseline mean | Shaped mean | Mean Delta | 95% CI mean | Median Delta | 95% CI median | P(Delta mean > 0) | W/T/L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in per_seed:
        mean_ci = result["mean_delta_95_ci"]
        median_ci = result["median_delta_95_ci"]
        lines.append(
            "| {training_seed} | {baseline:.3f} | {shaped:.3f} | {mean:.3f} | "
            "[{lower:.3f}, {upper:.3f}] | {median:.3f} | "
            "[{median_lower:.3f}, {median_upper:.3f}] | {prob:.4f} | "
            "{wins}/{ties}/{losses} |".format(
                training_seed=result["training_seed"],
                baseline=result["baseline_mean_raw_score"],
                shaped=result["shaped_mean_raw_score"],
                mean=result["point_mean_delta"],
                lower=mean_ci["lower"],
                upper=mean_ci["upper"],
                median=result["point_median_delta"],
                median_lower=median_ci["lower"],
                median_upper=median_ci["upper"],
                prob=result["p_bootstrap_mean_effect_gt_zero"],
                wins=result["wins"],
                ties=result["ties"],
                losses=result["losses"],
            )
        )
    seed_ci = seed_level["mean_delta_95_ci"]
    hierarchical_ci = hierarchical["mean_delta_95_ci"]
    seed_ci_crosses_zero = seed_ci["lower"] <= 0 <= seed_ci["upper"]
    hierarchical_ci_crosses_zero = (
        hierarchical_ci["lower"] <= 0 <= hierarchical_ci["upper"]
    )
    cross_seed_ci_crosses_zero = (
        seed_ci_crosses_zero or hierarchical_ci_crosses_zero
    )
    if cross_seed_ci_crosses_zero:
        cross_seed_support = (
            "not established: at least one cross-seed 95% CI crosses zero"
        )
    elif cross_seed_mean_effect > 0:
        cross_seed_support = (
            "positive at this resampling level, subject to the n_train limitation"
        )
    elif cross_seed_mean_effect < 0:
        cross_seed_support = "not supported: the observed cross-seed effect is negative"
    else:
        cross_seed_support = "not supported: the observed cross-seed effect is zero"
    lines.extend(
        [
            "",
            "## Cross-training-seed bootstrap",
            "",
            "- Candidate: `life_loss_penalty=-1.0`",
            f"- Seed-level point mean effect: `{seed_level['point_mean_delta']:.3f}`",
            f"- Seed-level point median effect: `{seed_level['point_median_delta']:.3f}`",
            f"- Seed-level 95% CI for mean: `[{seed_ci['lower']:.3f}, {seed_ci['upper']:.3f}]`",
            f"- Seed-level 95% CI for median: `[{seed_level['median_delta_95_ci']['lower']:.3f}, {seed_level['median_delta_95_ci']['upper']:.3f}]`",
            f"- Seed-level P(effect > 0): `{seed_level['p_bootstrap_mean_effect_gt_zero']:.4f}`",
            f"- Seed-level P(median effect > 0): `{seed_level['p_bootstrap_median_effect_gt_zero']:.4f}`",
            f"- Hierarchical point mean effect: `{hierarchical['point_mean_delta']:.3f}`",
            f"- Hierarchical point median effect: `{hierarchical['point_median_delta']:.3f}`",
            f"- Hierarchical 95% CI for mean: `[{hierarchical_ci['lower']:.3f}, {hierarchical_ci['upper']:.3f}]`",
            f"- Hierarchical 95% CI for median: `[{hierarchical['median_delta_95_ci']['lower']:.3f}, {hierarchical['median_delta_95_ci']['upper']:.3f}]`",
            f"- Hierarchical P(effect > 0): `{hierarchical['p_bootstrap_mean_effect_gt_zero']:.4f}`",
            f"- Hierarchical P(median effect > 0): `{hierarchical['p_bootstrap_median_effect_gt_zero']:.4f}`",
            f"- Hierarchical bootstrap std/SE: `{hierarchical['bootstrap_mean_std']:.3f}`",
            f"- Cross-seed point-estimate direction: `{cross_seed_direction}`",
            f"- Seed-level CI crosses zero: `{'yes' if seed_ci_crosses_zero else 'no'}`",
            f"- Hierarchical CI crosses zero: `{'yes' if hierarchical_ci_crosses_zero else 'no'}`",
            f"- Cross-training-seed inference: **{cross_seed_interpretation}**",
            f"- Does the evidence support cross-training-seed improvement? **{cross_seed_support}**",
            "",
            "## Interpretation",
            "",
            "The per-seed paired results describe uncertainty for each already-trained baseline/shaped pair. They do not establish cross-training-seed generalization.",
            "",
            f"The seed-level and hierarchical results are the relevant cross-training-seed analyses. With n_train={summary['n_train']}, their percentile CIs are coarse; a CI crossing zero must be reported as inconclusive rather than statistically significant.",
            "",
            "No naive pooled bootstrap of 150 evaluation episodes was used, and no pooled result may be used as promotion evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flat_per_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        output = {
            key: value for key, value in row.items() if not isinstance(value, Mapping)
        }
        for prefix in ("mean_delta_95_ci", "median_delta_95_ci"):
            interval = row.get(prefix)
            if isinstance(interval, Mapping):
                output[f"{prefix}_lower"] = interval.get("lower")
                output[f"{prefix}_upper"] = interval.get("upper")
        flattened.append(output)
    return flattened


def write_analysis_artifacts(
    output_dir: Path,
    summary: Mapping[str, Any],
    distributions: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "bootstrap-summary.json",
        "per-seed-bootstrap.csv",
        "cross-seed-bootstrap.json",
        "hierarchical-bootstrap.json",
        "bootstrap-distribution.csv",
        "analysis-report.md",
    )
    paths = {name: output_dir / name for name in filenames}
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "bootstrap output already exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    paths["bootstrap-summary.json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(
        paths["per-seed-bootstrap.csv"],
        _flat_per_seed_rows(summary["per_seed_paired_bootstrap"]),
    )
    paths["cross-seed-bootstrap.json"].write_text(
        json.dumps(
            summary["training_seed_level_bootstrap"],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    paths["hierarchical-bootstrap.json"].write_text(
        json.dumps(
            summary["hierarchical_bootstrap"],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_csv(paths["bootstrap-distribution.csv"], distributions)
    paths["analysis-report.md"].write_text(
        render_analysis_report(summary),
        encoding="utf-8",
    )
    return paths
