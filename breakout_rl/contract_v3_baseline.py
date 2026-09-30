"""Run, evaluate, and report the canonical Issue 21 Contract v3 baseline."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ale_py
import gymnasium
import numpy as np
import torch

from breakout_rl.artifacts import sha256_file
from breakout_env import make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_COMPLETION_DETECTOR_ID,
    read_ale_episode_frame,
)
from breakout_rl.evaluation import evaluate_policy, load_evaluation_config
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V3_ID,
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.config import DQNConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MILESTONES = (400_000, 1_000_000, 2_000_000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _resolve_path(value: str | Path, *, root: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _portable_path(path: Path, *, root: Path = REPOSITORY_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_plan(
    config_path: Path,
) -> tuple[dict[str, Any], DQNConfig, BreakoutEvaluationContractV2, Path, Path]:
    payload = _read_json_object(config_path)
    if payload.get("issue") != 21:
        raise ValueError("baseline config must identify Issue 21")
    if payload.get("training_lifecycle") != "continuous_single_run":
        raise ValueError("Issue 21 requires one continuous training lifecycle")
    raw_training = payload.get("training_config")
    if not isinstance(raw_training, Mapping):
        raise ValueError("baseline config training_config must be an object")
    training = DQNConfig.from_dict(raw_training)
    milestones = payload.get("milestones")
    if not isinstance(milestones, Sequence) or isinstance(milestones, (str, bytes)):
        raise ValueError("baseline config milestones must be a sequence")
    parsed_milestones = tuple(int(step) for step in milestones)
    if parsed_milestones != EXPECTED_MILESTONES:
        raise ValueError("Issue 21 milestones must be 400K, 1M, and 2M transitions")
    if training.checkpoint_steps != EXPECTED_MILESTONES:
        raise ValueError("training checkpoint_steps must match the Issue 21 milestones")
    if training.total_steps != 2_000_000 or training.seed != 11:
        raise ValueError("Issue 21 training must use seed 11 through 2M transitions")
    if (
        training.algorithm != "double_dqn"
        or training.architecture != "dueling"
        or training.replay_sampling != "uniform"
        or training.life_loss_penalty != 0.0
    ):
        raise ValueError(
            "Issue 21 requires unshaped Dueling Double DQN with Uniform Replay"
        )
    if training.replay_backend != "gpu" or training.device != "cuda":
        raise ValueError(
            "Issue 21 baseline must use the configured CUDA/GPU Replay path"
        )
    if training.num_envs != 2 or not training.strict_action_selection_parity:
        raise ValueError(
            "Issue 21 baseline requires the validated two-environment schedule"
        )

    contract_value = payload.get("contract")
    if not isinstance(contract_value, str):
        raise ValueError("baseline config must name its environment contract")
    contract_path = _resolve_path(contract_value)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    if contract.contract_id != BREAKOUT_CONTRACT_V3_ID or contract.frame_skip != 1:
        raise ValueError(
            "Issue 21 requires the canonical Contract v3 frame-skip-1 runtime"
        )

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("baseline config evaluation section must be an object")
    evaluation_config_path = _resolve_path(str(evaluation.get("config", "")))
    evaluation_contract_path = _resolve_path(str(evaluation.get("contract", "")))
    if evaluation_contract_path != contract_path:
        raise ValueError("training and evaluation must use the same Contract v3 file")
    evaluation_config = load_evaluation_config(evaluation_config_path)
    if evaluation_config.environment_id != contract.environment_id:
        raise ValueError("evaluation environment does not match Contract v3")
    if evaluation_config.epsilon != contract.evaluation_epsilon:
        raise ValueError("evaluation epsilon does not match Contract v3")
    if (
        expand_concrete_episode_seeds(
            evaluation_config.seeds,
            episodes_per_seed=evaluation_config.episodes_per_seed,
        )
        != contract.concrete_episode_seeds
    ):
        raise ValueError("evaluation seeds do not match the Contract v3 seed list")
    if int(evaluation.get("episodes_per_checkpoint", 0)) != len(
        contract.concrete_episode_seeds
    ):
        raise ValueError(
            "each checkpoint must use the complete canonical evaluation set"
        )
    if float(evaluation.get("epsilon", -1.0)) != contract.evaluation_epsilon:
        raise ValueError(
            "baseline config evaluation epsilon does not match Contract v3"
        )

    control = payload.get("control_diagnostics")
    if not isinstance(control, Mapping):
        raise ValueError("baseline config control_diagnostics must be an object")
    return (
        payload,
        training,
        contract,
        contract_path,
        evaluation_config_path,
    )


def _runtime_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Issue 21 training config requires an available CUDA device")
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gymnasium": gymnasium.__version__,
        "ale_py": ale_py.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_index": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
    }


def _validated_evaluation_device(value: str) -> str:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"unsupported evaluation device: {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("evaluation device must be cpu or cuda[:index]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA evaluation device is unavailable")
    return str(device)


def _smoke_environment(
    contract: BreakoutEvaluationContractV2,
    *,
    seed: int,
) -> dict[str, Any]:
    env = make_breakout_env(
        **breakout_environment_kwargs(contract, allow_contract_v3=True)
    )
    try:
        observation, _info = env.reset(seed=seed)
        if not isinstance(observation, np.ndarray):
            raise ValueError("Contract v3 reset observation must be a NumPy array")
        if observation.shape != (4, 84, 84) or observation.dtype != np.uint8:
            raise ValueError(
                "Contract v3 observation must retain shape (4, 84, 84) and uint8 dtype"
            )
        frame_before = read_ale_episode_frame(env)
        next_observation, reward, _terminated, _truncated, _step_info = env.step(0)
        frame_after = read_ale_episode_frame(env)
        if frame_before is None or frame_after is None:
            raise ValueError(
                "Contract v3 smoke did not expose the ALE-native frame counter"
            )
        frame_delta = frame_after - frame_before
        if frame_delta != 1:
            raise ValueError(
                "Contract v3 must advance exactly one ALE frame per agent decision; "
                f"observed delta={frame_delta}"
            )
        if not isinstance(next_observation, np.ndarray):
            raise ValueError("Contract v3 step observation must be a NumPy array")
        if next_observation.shape != (4, 84, 84) or next_observation.dtype != np.uint8:
            raise ValueError("Contract v3 step observation shape/dtype changed")
        if not math.isfinite(float(reward)):
            raise ValueError("Contract v3 smoke produced a non-finite reward")
        return {
            "status": "passed",
            "frame_skip": contract.frame_skip,
            "ale_frame_before": frame_before,
            "ale_frame_after": frame_after,
            "ale_frame_delta": frame_delta,
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
        }
    finally:
        env.close()


def _validate_evaluation_summary(
    summary: Mapping[str, Any],
    *,
    source: str | Path,
    expected_episode_count: int,
) -> None:
    if summary.get("clear_detection_available") is not True:
        raise ValueError(f"{source}: clear detection is unavailable")
    if summary.get("total_episodes") != expected_episode_count:
        raise ValueError(f"{source}: summary episode count is inconsistent")
    clear_count = summary.get("clear_count")
    clear_rate = summary.get("clear_rate")
    if (
        not isinstance(clear_count, int)
        or isinstance(clear_count, bool)
        or not 0 <= clear_count <= expected_episode_count
    ):
        raise ValueError(f"{source}: clear_count is invalid")
    if (
        isinstance(clear_rate, bool)
        or not isinstance(clear_rate, (int, float))
        or not math.isfinite(float(clear_rate))
        or not 0.0 <= float(clear_rate) <= 1.0
    ):
        raise ValueError(f"{source}: clear_rate must be a finite probability")
    if not math.isclose(
        float(clear_rate),
        clear_count / expected_episode_count,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{source}: clear_rate does not match clear_count")

    for field in ("mean_return", "median_return", "max_return"):
        value = summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{source}: raw score field {field} is missing or invalid")

    timing_fields = (
        "best_clear_steps",
        "median_clear_steps",
        "best_clear_emulator_frame",
        "median_clear_emulator_frame",
    )
    step_samples = summary.get("clear_time_sample_count")
    frame_samples = summary.get("clear_emulator_frame_sample_count")
    if clear_count == 0:
        if step_samples != 0 or frame_samples != 0:
            raise ValueError(f"{source}: zero-clear result has clear-time samples")
        if any(summary.get(field) is not None for field in timing_fields):
            raise ValueError(f"{source}: zero-clear result has clear-time values")
    else:
        if not isinstance(step_samples, int) or step_samples <= 0:
            raise ValueError(f"{source}: clear-time agent-step metrics are missing")
        if not isinstance(frame_samples, int) or frame_samples <= 0:
            raise ValueError(f"{source}: clear-time emulator-frame metrics are missing")
        if any(summary.get(field) is None for field in timing_fields):
            raise ValueError(f"{source}: clear-time metrics are incomplete")

    for field in (
        "mean_life_loss_count",
        "mean_score_per_life",
        "mean_frames_between_life_losses",
        "mean_time_to_first_life_loss",
        "life_losses_per_1000_steps",
    ):
        if field not in summary:
            raise ValueError(f"{source}: survival metric {field} is missing")


def _evaluation_smoke(
    contract: BreakoutEvaluationContractV2,
    *,
    contract_path: Path,
) -> dict[str, Any]:
    """Run the complete canonical seed set through the Contract v3 evaluator."""

    env_factory = lambda: make_breakout_env(
        **breakout_environment_kwargs(contract, allow_contract_v3=True)
    )
    result = evaluate_policy(
        None,
        episodes=1,
        seeds=contract.concrete_episode_seeds,
        device="cpu",
        epsilon=contract.evaluation_epsilon,
        env_factory=env_factory,
        max_steps_per_episode=contract.time_limit_semantics["agent_step_limit"],
        model_id="issue21-contract-v3-evaluation-smoke",
        evaluation_id="issue21-contract-v3-evaluation-smoke",
        metadata={
            "evaluation_contract_path": _portable_path(contract_path),
            "evaluation_contract": contract.to_dict(),
        },
    ).to_dict()
    summary = result["summary"]
    _validate_evaluation_summary(
        summary,
        source="Contract v3 evaluation smoke",
        expected_episode_count=len(contract.concrete_episode_seeds),
    )
    if result.get("total_emulator_frames") is None:
        raise ValueError("Contract v3 evaluation smoke omitted native ALE frame counts")
    provenance = result.get("contract_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Contract v3 evaluation smoke omitted contract provenance")
    if provenance.get("validation_status") != "canonical_contract_v3":
        raise ValueError("Contract v3 evaluation smoke failed runtime binding")
    detector = result.get("completion_detector")
    if not isinstance(detector, Mapping):
        raise ValueError(
            "Contract v3 evaluation smoke omitted completion detector identity"
        )
    if detector.get("detector_id") != BREAKOUT_COMPLETION_DETECTOR_ID:
        raise ValueError("Contract v3 evaluation smoke used an unexpected detector")
    return {
        "status": "passed",
        "evaluation_seeds": list(contract.concrete_episode_seeds),
        "episodes_per_seed": 1,
        "total_episodes": result["total_episodes"],
        "total_agent_steps": result["total_agent_steps"],
        "total_emulator_frames": result["total_emulator_frames"],
        "clear_count": summary["clear_count"],
        "clear_rate": summary["clear_rate"],
        "mean_return": summary["mean_return"],
        "median_return": summary["median_return"],
        "max_return": summary["max_return"],
        "clear_time_sample_count": summary["clear_time_sample_count"],
        "survival_metrics": {
            field: summary[field]
            for field in (
                "mean_life_loss_count",
                "mean_score_per_life",
                "mean_frames_between_life_losses",
                "mean_time_to_first_life_loss",
                "life_losses_per_1000_steps",
            )
        },
        "clear_detection_available": summary["clear_detection_available"],
        "contract_validation_status": provenance["validation_status"],
        "completion_detector_id": detector["detector_id"],
    }


def _training_smoke(
    payload: Mapping[str, Any],
    *,
    contract_path: Path,
    contract: BreakoutEvaluationContractV2,
) -> dict[str, Any]:
    raw_training = payload.get("training_config")
    if not isinstance(raw_training, Mapping):
        raise ValueError("baseline config training_config must be an object")
    smoke_training = dict(raw_training)
    smoke_training.update(
        {
            "total_steps": 256,
            "epsilon_decay_steps": 256,
            "replay_capacity": 512,
            "learning_starts": 32,
            "target_update_interval": 128,
            "checkpoint_interval": 256,
            "checkpoint_steps": [256],
            "diagnostics_interval": 128,
            "metrics_flush_interval": 256,
            "profile_stages": False,
        }
    )
    DQNConfig.from_dict(smoke_training)
    smoke_config = dict(payload)
    smoke_config["milestones"] = [256]
    smoke_config["training_config"] = smoke_training

    with tempfile.TemporaryDirectory(prefix="issue21-v3-smoke-") as temporary:
        temporary_root = Path(temporary)
        smoke_config_path = temporary_root / "smoke-config.json"
        smoke_config_path.write_text(
            json.dumps(smoke_config, indent=2) + "\n",
            encoding="utf-8",
        )
        run_id = "issue21-preflight"
        run_dir = temporary_root / run_id
        summary_path = run_dir / "training_summary.json"
        command = [
            sys.executable,
            "-m",
            "scripts.training.train_vectorized_dqn",
            "--config",
            str(smoke_config_path),
            "--contract",
            _portable_path(contract_path),
            "--run-dir",
            str(temporary_root),
            "--run-id",
            run_id,
            "--output",
            str(summary_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            output = (completed.stdout + "\n" + completed.stderr)[-6000:]
            raise RuntimeError(f"real-ALE training smoke failed:\n{output}")
        summary = _read_json_object(summary_path)
        if summary.get("total_agent_steps") != 256:
            raise ValueError("real-ALE training smoke did not complete 256 transitions")
        if summary.get("total_emulator_frames") is None:
            raise ValueError("real-ALE training smoke did not record native ALE frames")
        if int(summary.get("optimizer_updates", 0)) <= 0:
            raise ValueError(
                "Dueling Double DQN smoke did not perform optimizer updates"
            )
        if (
            summary.get("algorithm") != "double_dqn"
            or summary.get("architecture") != "dueling"
        ):
            raise ValueError("training smoke did not use Dueling Double DQN")
        if summary.get("replay_sampling") != "uniform":
            raise ValueError("training smoke did not use Uniform Replay")
        if summary.get("strict_action_selection_parity_satisfied") is not True:
            raise ValueError("training smoke failed vectorized action-selection parity")
        if summary.get("model_config", {}).get("input_shape") != [4, 84, 84]:
            raise ValueError("training smoke changed the Contract v3 observation shape")

        checkpoint_path = run_dir / "checkpoints" / "step-00000256.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("training smoke did not write its checkpoint")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        contract_metadata = checkpoint.get("environment_contract")
        if not isinstance(contract_metadata, Mapping):
            raise ValueError(
                "training checkpoint omitted environment contract provenance"
            )
        if contract_metadata.get("contract_id") != contract.contract_id:
            raise ValueError("training checkpoint recorded the wrong contract id")
        if contract_metadata.get("contract_sha256") != sha256_file(contract_path):
            raise ValueError("training checkpoint recorded the wrong contract hash")
        if checkpoint.get("global_step") != 256:
            raise ValueError("training smoke checkpoint has the wrong transition count")
        if checkpoint.get("total_emulator_frames") is None:
            raise ValueError("training smoke checkpoint omitted native ALE frame count")
        return {
            "status": "passed",
            "total_agent_steps": summary["total_agent_steps"],
            "total_emulator_frames": summary["total_emulator_frames"],
            "optimizer_updates": summary["optimizer_updates"],
            "replay_sampling": summary["replay_sampling"],
            "observation_shape": summary["model_config"]["input_shape"],
            "contract_sha256": contract_metadata["contract_sha256"],
        }


def run_preflight(config_path: Path) -> dict[str, Any]:
    payload, training, contract, contract_path, evaluation_config_path = _validate_plan(
        config_path
    )
    runtime = _runtime_metadata()
    environment = _smoke_environment(contract, seed=training.seed)
    evaluation = _evaluation_smoke(
        contract,
        contract_path=contract_path,
    )
    smoke = _training_smoke(
        payload,
        contract_path=contract_path,
        contract=contract,
    )
    return {
        "status": "passed",
        "runtime": runtime,
        "environment_smoke": environment,
        "evaluation_smoke": evaluation,
        "training_smoke": smoke,
        "contract_path": _portable_path(contract_path),
        "evaluation_config_path": _portable_path(evaluation_config_path),
    }


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint must contain a mapping")
    return payload


def _validate_training_result(
    summary: Mapping[str, Any],
    *,
    training: DQNConfig,
    contract: BreakoutEvaluationContractV2,
    contract_path: Path,
    run_dir: Path,
) -> None:
    if summary.get("status") != "completed":
        raise ValueError("formal training did not complete successfully")
    if summary.get("total_agent_steps") != training.total_steps:
        raise ValueError("formal training transition count does not equal 2M")
    if summary.get("seed") != 11:
        raise ValueError("formal training summary does not report seed 11")
    if summary.get("contract_id") != contract.contract_id:
        raise ValueError("formal training summary does not identify Contract v3")
    if summary.get("total_emulator_frames") is None:
        raise ValueError("formal training summary omitted native ALE frame count")
    if summary.get("replay_sampling") != "uniform":
        raise ValueError("formal training summary does not report Uniform Replay")
    if int(summary.get("optimizer_updates", 0)) <= 0:
        raise ValueError("formal training completed without optimizer updates")
    if summary.get("strict_action_selection_parity_satisfied") is not True:
        raise ValueError("formal training did not satisfy action-selection parity")
    contract_metadata = summary.get("environment_contract")
    if not isinstance(contract_metadata, Mapping):
        raise ValueError("formal training summary omitted contract hash provenance")
    if contract_metadata.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("formal training contract hash does not match the source file")
    if contract_metadata.get("contract_id") != contract.contract_id:
        raise ValueError("formal training contract id does not match the source file")
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise ValueError(
            "formal training did not preserve per-transition action metrics"
        )


def _validate_evaluation_result(
    path: Path,
    *,
    checkpoint: Mapping[str, Any],
    expected_checkpoint_step: int,
    expected_checkpoint_sha256: str,
    expected_contract_hash: str,
    expected_contract_path: Path,
    expected_source_commit: str,
    expected_seeds: Sequence[int],
    expected_episode_count: int,
) -> dict[str, Any]:
    result = _read_json_object(path)
    if result.get("evaluation_status") != "completed":
        raise ValueError(f"{path}: evaluation did not complete")
    if result.get("evaluation_seeds") != list(expected_seeds):
        raise ValueError(f"{path}: evaluation seed list does not match the protocol")
    if result.get("total_episodes") != expected_episode_count:
        raise ValueError(f"{path}: evaluation episode count is incomplete")
    if result.get("episodes_per_seed") != 1:
        raise ValueError(f"{path}: each concrete Contract v3 seed must run once")
    training = result.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(f"{path}: checkpoint training metadata is missing")
    if training.get("total_agent_steps") != expected_checkpoint_step:
        raise ValueError(f"{path}: evaluation loaded the wrong checkpoint transition")
    if training.get("contract_id") != checkpoint.get("contract_id"):
        raise ValueError(f"{path}: checkpoint contract id is inconsistent")
    recorded_contract_path = Path(str(training.get("contract_path", "")))
    if not recorded_contract_path.is_absolute():
        recorded_contract_path = REPOSITORY_ROOT / recorded_contract_path
    if recorded_contract_path.resolve() != expected_contract_path.resolve():
        raise ValueError(f"{path}: checkpoint contract path is inconsistent")
    contract = result.get("contract_provenance")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{path}: evaluation contract provenance is missing")
    if contract.get("contract_id") != BREAKOUT_CONTRACT_V3_ID:
        raise ValueError(f"{path}: evaluation is not labeled Contract v3")
    if contract.get("contract_sha256") != expected_contract_hash:
        raise ValueError(f"{path}: evaluation recorded the wrong contract hash")
    if contract.get("validation_status") != "canonical_contract_v3":
        raise ValueError(f"{path}: Contract v3 runtime binding was not validated")
    if result.get("total_emulator_frames") is None:
        raise ValueError(f"{path}: evaluation omitted native ALE frame counts")
    source = result.get("source_provenance")
    if (
        not isinstance(source, Mapping)
        or source.get("source_commit") != expected_source_commit
    ):
        raise ValueError(f"{path}: source commit does not match the training run")
    detector = result.get("completion_detector")
    if not isinstance(detector, Mapping):
        raise ValueError(f"{path}: completion detector identity is missing")
    if detector.get("detector_id") != BREAKOUT_COMPLETION_DETECTOR_ID:
        raise ValueError(f"{path}: unexpected completion detector identity")

    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path}: evaluation summary is missing")
    _validate_evaluation_summary(
        summary,
        source=path,
        expected_episode_count=expected_episode_count,
    )
    checkpoint_result = result.get("checkpoint")
    if not isinstance(checkpoint_result, Mapping):
        raise ValueError(f"{path}: checkpoint identity is missing")
    if checkpoint_result.get("sha256") != expected_checkpoint_sha256:
        raise ValueError(
            f"{path}: checkpoint SHA-256 does not match the evaluated file"
        )
    return result


def _markdown_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.{digits}f}"


def _build_screening_assessment(
    milestone_records: Sequence[Mapping[str, Any]],
    *,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not milestone_records:
        raise ValueError("screening assessment requires evaluated milestones")
    clear_checkpoints = [
        int(item["agent_steps"])
        for item in milestone_records
        if int(item["summary"]["clear_count"]) > 0
    ]
    score_progression = []
    for item in milestone_records:
        summary = item["summary"]
        score_progression.append(
            {
                "agent_steps": int(item["agent_steps"]),
                "mean_raw_score": float(summary["mean_return"]),
                "median_raw_score": float(summary["median_return"]),
                "max_raw_score": float(summary["max_return"]),
                "clear_count": int(summary["clear_count"]),
                "clear_rate": float(summary["clear_rate"]),
                "survival_metrics": {
                    key: float(summary[key])
                    for key in (
                        "mean_life_loss_count",
                        "life_losses_per_1000_steps",
                        "mean_score_per_life",
                        "mean_frames_between_life_losses",
                        "mean_time_to_first_life_loss",
                    )
                },
            }
        )
    reference = score_progression[0]
    latest = score_progression[-1]
    score_metric_keys = ("mean_raw_score", "median_raw_score", "max_raw_score")
    consistent_score_progression = len(score_progression) > 1 and all(
        all(
            earlier[key] < later[key]
            for earlier, later in zip(score_progression, score_progression[1:])
        )
        for key in score_metric_keys
    )
    higher_clear_count_or_rate = any(
        later["clear_count"] > reference["clear_count"]
        or later["clear_rate"] > reference["clear_rate"]
        for later in score_progression[1:]
    )
    best_later = max(
        score_progression[1:] or [reference], key=lambda item: item["max_raw_score"]
    )
    higher_max_score = best_later["max_raw_score"] > reference["max_raw_score"]
    survival_directions = {
        "mean_life_loss_count": "lower",
        "life_losses_per_1000_steps": "lower",
        "mean_score_per_life": "higher",
        "mean_frames_between_life_losses": "higher",
        "mean_time_to_first_life_loss": "higher",
    }
    survival_comparison = {
        key: {
            "reference": reference["survival_metrics"][key],
            "latest": latest["survival_metrics"][key],
            "favorable_at_latest": (
                latest["survival_metrics"][key] < reference["survival_metrics"][key]
                if direction == "lower"
                else latest["survival_metrics"][key]
                > reference["survival_metrics"][key]
            ),
        }
        for key, direction in survival_directions.items()
    }
    favorable_survival_metrics = sum(
        item["favorable_at_latest"] for item in survival_comparison.values()
    )
    if favorable_survival_metrics == len(survival_comparison):
        survival_status = "positive_signal"
    elif favorable_survival_metrics:
        survival_status = "mixed"
    else:
        survival_status = "not_observed"
    high_score_samples = (
        controls.get("high_score", {}).get("sample_count")
        if controls is not None
        else None
    )
    first_clear = {
        "status": "met" if clear_checkpoints else "not_met",
        "clear_checkpoints": clear_checkpoints,
    }
    criteria = {
        "first_verified_clear": first_clear,
        "higher_clear_count_or_rate": {
            "status": "observed" if higher_clear_count_or_rate else "not_observed",
            "checkpoint_values": [
                {
                    "agent_steps": item["agent_steps"],
                    "clear_count": item["clear_count"],
                    "clear_rate": item["clear_rate"],
                }
                for item in score_progression
            ],
        },
        "materially_higher_max_score": {
            "status": "observed_increase" if higher_max_score else "not_observed",
            "reference_checkpoint_agent_steps": reference["agent_steps"],
            "reference_max_raw_score": reference["max_raw_score"],
            "best_later_checkpoint_agent_steps": best_later["agent_steps"],
            "best_later_max_raw_score": best_later["max_raw_score"],
            "increase_points": best_later["max_raw_score"] - reference["max_raw_score"],
            "interpretation": (
                "within-run checkpoint change; materiality is judged from the full "
                "raw-score progression and is not an independent control comparison"
            ),
        },
        "consistent_raw_score_progression": {
            "status": "observed" if consistent_score_progression else "not_observed",
            "required_metrics": list(score_metric_keys),
            "checkpoints": score_progression,
        },
        "materially_better_late_game_survival": {
            "status": survival_status,
            "reference_checkpoint_agent_steps": reference["agent_steps"],
            "latest_checkpoint_agent_steps": latest["agent_steps"],
            "latest_vs_reference": survival_comparison,
            "interpretation": (
                "descriptive endpoint checkpoint comparison; one survival metric "
                "moving against the reference keeps this signal mixed"
                if survival_status == "mixed"
                else "descriptive endpoint checkpoint comparison"
            ),
        },
        "visibly_improved_high_speed_paddle_control": {
            "status": "not_assessed",
            "reason": (
                "no high-score training transitions were available"
                if high_score_samples == 0
                else "no checkpoint-specific paddle-contact control comparison is available"
            ),
            "high_score_training_sample_count": high_score_samples,
        },
        "better_performance_at_matched_emulator_frame_budget": {
            "status": "not_assessed",
            "reason": "no historical matched-frame comparator is included in this run",
        },
        "independent_matched_contract_v3_control": {
            "status": "not_assessed",
            "reason": "this issue runs one training lifecycle and no separate control",
        },
    }
    promising = bool(clear_checkpoints) or consistent_score_progression
    if clear_checkpoints:
        basis = "at least one canonical Contract v3 evaluation episode recorded a verified clear"
    elif consistent_score_progression:
        baseline_values = " / ".join(
            f"{reference[key]:.2f}" for key in score_metric_keys
        )
        latest_values = " / ".join(f"{latest[key]:.2f}" for key in score_metric_keys)
        basis = (
            "raw score mean / median / max improved at each of the three canonical "
            "Contract v3 checkpoints on the same 15 evaluation seeds "
            f"({reference['agent_steps']:,} steps: {baseline_values}; "
            f"{latest['agent_steps']:,} steps: {latest_values}); no verified clear "
            "was recorded"
        )
    else:
        basis = (
            "no verified clear or consistent raw-score progression across the "
            "canonical checkpoints was observed; the single-run results do not "
            "establish comparative survival, control, or matched-frame evidence"
        )
    if promising:
        return {
            "status": "promising",
            "promising": True,
            "decision": "promote_for_follow_up",
            "basis": basis,
            "criteria": criteria,
            "clear_checkpoints": clear_checkpoints,
            "score_progression": score_progression,
            "second_training_seed_started": False,
            "next_algorithm_started": False,
        }
    return {
        "status": "inconclusive",
        "promising": None,
        "decision": "hold_for_more_evidence",
        "basis": basis,
        "criteria": criteria,
        "clear_checkpoints": [],
        "score_progression": score_progression,
        "second_training_seed_started": False,
        "next_algorithm_started": False,
    }


def _build_report(
    *,
    manifest: Mapping[str, Any],
    evaluation_results: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> str:
    checkpoint_rows = []
    observed_clears: list[tuple[int, int]] = []
    for item, result in zip(manifest["milestones"], evaluation_results, strict=True):
        summary = result["summary"]
        checkpoint_step = int(item["agent_steps"])
        clear_count = int(summary["clear_count"])
        if clear_count:
            observed_clears.append((checkpoint_step, clear_count))
        checkpoint_rows.append(
            "| {steps} | {frames} | {mean} / {median} / {maximum} | {clears} / "
            "{rate} | {best_step} / {median_step} | {best_frame} / {median_frame} | "
            "{clear_samples} | {survival} |".format(
                steps=f"{checkpoint_step:,}",
                frames=_markdown_number(item["total_emulator_frames"], 0),
                mean=_markdown_number(summary.get("mean_return")),
                median=_markdown_number(summary.get("median_return")),
                maximum=_markdown_number(summary.get("max_return"), 0),
                clears=f"{clear_count} / {summary['total_episodes']}",
                rate=f"{float(summary['clear_rate']):.1%}",
                best_step=_markdown_number(summary.get("best_clear_steps"), 0),
                median_step=_markdown_number(summary.get("median_clear_steps"), 0),
                best_frame=_markdown_number(
                    summary.get("best_clear_emulator_frame"), 0
                ),
                median_frame=_markdown_number(
                    summary.get("median_clear_emulator_frame"),
                    0,
                ),
                clear_samples=_markdown_number(
                    summary.get("clear_time_sample_count"),
                    0,
                ),
                survival=(
                    f"losses={_markdown_number(summary.get('mean_life_loss_count'))}; "
                    f"per 1K={_markdown_number(summary.get('life_losses_per_1000_steps'))}; "
                    f"score/life={_markdown_number(summary.get('mean_score_per_life'))}; "
                    f"frames between={_markdown_number(summary.get('mean_frames_between_life_losses'))}; "
                    f"steps to first={_markdown_number(summary.get('mean_time_to_first_life_loss'))}"
                ),
            )
        )

    assessment = manifest["screening_assessment"]
    if assessment["status"] == "promising" and observed_clears:
        decision = (
            "Promising for follow-up: canonical Contract v3 evaluation recorded a "
            f"verified clear at {observed_clears[0][0]:,} agent steps."
        )
    elif assessment["status"] == "promising":
        decision = (
            "Promising enough to advance to an explicit follow-up algorithm screen: "
            f"yes. {assessment['basis']}. The 2M evaluation still recorded no "
            "verified clear (0 of 15 episodes); this is a within-run learning-curve "
            "signal, not an independent matched-control or statistical superiority "
            "claim. No extra seed or n-step run was started automatically."
        )
    else:
        decision = (
            "Screening assessment: inconclusive; hold for more evidence. "
            f"{assessment['basis']}. This is not evidence that the baseline performed "
            "worse. The run does not automatically launch another seed or the n-step "
            "experiment."
        )

    runtime = manifest["runtime"]
    contract = manifest["contract"]
    control_lines = [
        f"- Transitions inspected: {_markdown_number(controls['transition_count'], 0)}",
        f"- NOOP use: {_markdown_number(controls['noop_rate'] * 100, 1)}% "
        f"({_markdown_number(controls['noop_count'], 0)} transitions)",
        f"- Paddle-control cadence (LEFT or RIGHT): "
        f"{_markdown_number(controls['paddle_control_cadence'] * 100, 1)}%",
        f"- Action-switch rate: {_markdown_number(controls['action_switch_rate'] * 100, 1)}% "
        f"({controls['action_switch_count']:,} of "
        f"{controls['adjacent_action_pairs']:,} adjacent pairs)",
        f"- Direct LEFT/RIGHT reversals: "
        f"{_markdown_number(controls['left_right_reversal_count'], 0)} "
        f"({_markdown_number(controls['left_right_reversal_rate'] * 100, 1)}% of pairs)",
        f"- Late-episode action distribution: "
        f"`{json.dumps(controls['late_episode']['action_distribution'], sort_keys=True)}` "
        f"across {controls['late_episode']['sample_count']:,} decisions.",
        f"- High-score action distribution (raw score >= "
        f"{controls['high_score']['threshold_raw_score']:g}): "
        f"`{json.dumps(controls['high_score']['action_distribution'], sort_keys=True)}` "
        f"across {controls['high_score']['sample_count']:,} decisions.",
    ]
    score_progression = assessment["score_progression"]
    score_progression_text = "; ".join(
        "{steps:,}: {mean:.2f} / {median:.2f} / {maximum:.2f}".format(
            steps=item["agent_steps"],
            mean=item["mean_raw_score"],
            median=item["median_raw_score"],
            maximum=item["max_raw_score"],
        )
        for item in score_progression
    )
    screening_criteria = assessment["criteria"]
    survival_comparison = screening_criteria["materially_better_late_game_survival"]
    favorable_survival_count = sum(
        item["favorable_at_latest"]
        for item in survival_comparison["latest_vs_reference"].values()
    )
    screening_criteria_lines = [
        "- Screening assessment implementation commit: "
        f"`{assessment.get('assessment_source_commit', manifest['source_commit'])}`.",
        f"- First verified clear: `{screening_criteria['first_verified_clear']['status']}`.",
        f"- Higher clear count or rate after 400K: `{screening_criteria['higher_clear_count_or_rate']['status']}`.",
        f"- Raw score mean / median / max by checkpoint: {score_progression_text}.",
        "- Late-game survival endpoint vs 400K: "
        f"`{survival_comparison['status']}` "
        f"({favorable_survival_count} of "
        f"{len(survival_comparison['latest_vs_reference'])} indicators moved favorably).",
        "- High-speed paddle-control comparison: "
        f"`{screening_criteria['visibly_improved_high_speed_paddle_control']['status']}` "
        f"({screening_criteria['visibly_improved_high_speed_paddle_control']['reason']}).",
        "- Matched emulator-frame comparison: "
        f"`{screening_criteria['better_performance_at_matched_emulator_frame_budget']['status']}` "
        f"({screening_criteria['better_performance_at_matched_emulator_frame_budget']['reason']}).",
    ]
    return "\n".join(
        [
            "# Contract v3 Dueling Double DQN baseline",
            "",
            "## Training",
            "",
            f"- Training seed: 11; continuous lifecycle: 0 to 2,000,000 transitions.",
            f"- Replay: Uniform; algorithm: Dueling Double DQN; frame skip: 1.",
            f"- Contract: `{contract['contract_id']}` "
            f"(SHA-256 `{contract['sha256']}`).",
            f"- Source commit: `{manifest['source_commit']}`.",
            f"- Runtime: Python {runtime['python']}, PyTorch {runtime['torch']}, "
            f"CUDA {runtime['torch_cuda']}, {runtime['cuda_device_name']}.",
            f"- Actual training emulator frames at 2M transitions: "
            f"{_markdown_number(manifest['training']['total_emulator_frames'], 0)} "
            "from `ALEInterface.getEpisodeFrameNumber`; reset-time serve actions are excluded.",
            "",
            "## Canonical Contract v3 evaluation",
            "",
            "Each checkpoint used the same 15 evaluation episodes and raw rewards. "
            "Mean, median, and max are raw Breakout scores.",
            "",
            "| Agent steps | Training ALE frames | Mean / median / max raw score | Clears / episodes; clear rate | Best / median clear agent step | Best / median clear ALE frame | Clear-time samples | Survival metrics |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *checkpoint_rows,
            "",
            "Clear-time fields are `n/a` when no clears were observed; zero clears are "
            "reported as `clear_count = 0`, `clear_rate = 0`.",
            "",
            "## Control diagnostics",
            "",
            *control_lines,
            "",
            "The late-episode distribution covers the final 25% of completed training "
            "episodes by agent-transition count. The high-score view starts when raw "
            "episode score reaches 500. These are descriptive action summaries and do "
            "not change training rewards.",
            "",
            "## Screening assessment",
            "",
            decision,
            "",
            *screening_criteria_lines,
            "",
            "Contract v2 uses a different frame-skip contract and different agent-step "
            "semantics. The step counts above are not v2-equivalent budgets; use the "
            "recorded ALE-native frame counts for any historical reference. This is a "
            "single-seed screen and does not establish statistical superiority.",
            "",
        ]
    )


def run_formal_experiment(
    *,
    config_path: Path,
    run_dir: Path,
    evaluation_device: str,
    control_summarizer: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    if not callable(control_summarizer):
        raise TypeError("control_summarizer must be callable")
    evaluation_device = _validated_evaluation_device(evaluation_device)
    payload, training, contract, contract_path, evaluation_config_path = _validate_plan(
        config_path
    )
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists; refusing to overwrite or continue a run "
            "without replay-state resume"
        )
    preflight = run_preflight(config_path)
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    milestone_records: list[dict[str, Any]] = []
    evaluation_results: list[dict[str, Any]] = []
    control_threshold = float(payload["control_diagnostics"]["high_score_threshold"])
    late_episode_fraction = float(
        payload["control_diagnostics"]["late_episode_fraction"]
    )
    eval_config = load_evaluation_config(evaluation_config_path)
    expected_seeds = contract.concrete_episode_seeds
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": payload["experiment_id"],
        "issue": 21,
        "status": "running",
        "started_at_utc": _utc_now(),
        "source_commit": source,
        "source_working_tree_dirty": dirty,
        "config_path": _portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "contract": {
            "contract_id": contract.contract_id,
            "path": _portable_path(contract_path),
            "sha256": sha256_file(contract_path),
            "frame_skip": contract.frame_skip,
        },
        "runtime": preflight["runtime"],
        "preflight": preflight,
        "training_config": training.to_dict(),
        "evaluation_protocol": {
            "config_path": _portable_path(evaluation_config_path),
            "contract_path": _portable_path(contract_path),
            "configured_seed_groups": list(eval_config.seeds),
            "configured_episodes_per_seed": eval_config.episodes_per_seed,
            "evaluation_seeds": list(expected_seeds),
            "episodes_per_seed": 1,
            "expanded_episode_seeds": list(expected_seeds),
            "epsilon": eval_config.epsilon,
            "device": evaluation_device,
        },
        "completion_detector": {
            "detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "path": "breakout_rl/completion.py",
            "sha256": sha256_file(REPOSITORY_ROOT / "breakout_rl/completion.py"),
        },
        "milestone_steps": list(EXPECTED_MILESTONES),
        "milestones": milestone_records,
        "training": None,
        "control_diagnostics": None,
        "screening_assessment": None,
    }
    _write_json(manifest_path, manifest)

    try:
        training_summary_path = run_dir / "training_summary.json"
        training_command = [
            sys.executable,
            "-m",
            "scripts.training.train_vectorized_dqn",
            "--config",
            _portable_path(config_path),
            "--run-dir",
            str(run_dir.parent),
            "--run-id",
            run_dir.name,
            "--output",
            str(training_summary_path),
        ]
        completed = subprocess.run(training_command, cwd=REPOSITORY_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"continuous training returned exit code {completed.returncode}"
            )
        training_summary = _read_json_object(training_summary_path)
        _validate_training_result(
            training_summary,
            training=training,
            contract=contract,
            contract_path=contract_path,
            run_dir=run_dir,
        )
        manifest["training"] = {
            "status": "completed",
            "summary_path": "training_summary.json",
            "summary_sha256": sha256_file(training_summary_path),
            "total_agent_steps": training_summary["total_agent_steps"],
            "total_emulator_frames": training_summary["total_emulator_frames"],
            "emulator_frame_count_source": training_summary.get(
                "emulator_frame_count_source"
            ),
            "optimizer_updates": training_summary["optimizer_updates"],
            "replay_sampling": training_summary["replay_sampling"],
            "wall_clock_seconds": training_summary["runtime"].get("wall_clock_seconds"),
        }
        manifest["status"] = "evaluating_checkpoints"
        _write_json(manifest_path, manifest)

        evaluation_root = run_dir / "evaluations"
        for step in EXPECTED_MILESTONES:
            checkpoint_path = run_dir / "checkpoints" / f"step-{step:08d}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"missing Issue 21 checkpoint: {checkpoint_path}"
                )
            checkpoint = _checkpoint_payload(checkpoint_path)
            if checkpoint.get("global_step") != step:
                raise ValueError(
                    f"{checkpoint_path}: wrong checkpoint transition count"
                )
            checkpoint_sha256 = sha256_file(checkpoint_path)
            if checkpoint.get("total_emulator_frames") is None:
                raise ValueError(f"{checkpoint_path}: missing native ALE frame count")
            checkpoint_contract = checkpoint.get("environment_contract")
            if not isinstance(checkpoint_contract, Mapping):
                raise ValueError(f"{checkpoint_path}: missing contract provenance")
            if checkpoint_contract.get("contract_sha256") != sha256_file(contract_path):
                raise ValueError(f"{checkpoint_path}: contract hash mismatch")

            output_dir = evaluation_root / f"step-{step:08d}"
            evaluation_id = f"issue21-contract-v3-seed11-step-{step:08d}"
            evaluation_command = [
                sys.executable,
                "-m",
                "scripts.evaluation.evaluate_dqn",
                "--policy",
                "dqn",
                "--checkpoint",
                str(checkpoint_path),
                "--config",
                _portable_path(evaluation_config_path),
                "--contract",
                _portable_path(contract_path),
                "--device",
                evaluation_device,
                "--output-dir",
                str(output_dir),
                "--evaluation-id",
                evaluation_id,
            ]
            evaluation_process = subprocess.run(
                evaluation_command,
                cwd=REPOSITORY_ROOT,
                check=False,
            )
            if evaluation_process.returncode != 0:
                raise RuntimeError(
                    f"evaluation at {step:,} transitions returned exit code "
                    f"{evaluation_process.returncode}"
                )
            results_path = output_dir / "results.json"
            result = _validate_evaluation_result(
                results_path,
                checkpoint=checkpoint,
                expected_checkpoint_step=step,
                expected_checkpoint_sha256=checkpoint_sha256,
                expected_contract_hash=sha256_file(contract_path),
                expected_contract_path=contract_path,
                expected_source_commit=source,
                expected_seeds=contract.concrete_episode_seeds,
                expected_episode_count=len(expected_seeds),
            )
            checkpoint_record = {
                "agent_steps": step,
                "total_emulator_frames": checkpoint["total_emulator_frames"],
                "checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_id": evaluation_id,
                "evaluation_results_path": results_path.relative_to(run_dir).as_posix(),
                "evaluation_results_sha256": sha256_file(results_path),
                "evaluation_total_agent_steps": result["total_agent_steps"],
                "evaluation_total_emulator_frames": result["total_emulator_frames"],
                "summary": dict(result["summary"]),
            }
            milestone_records.append(checkpoint_record)
            evaluation_results.append(result)
            _write_json(manifest_path, manifest)

        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open("r", newline="", encoding="utf-8") as stream:
            controls = control_summarizer(
                csv.DictReader(stream),
                high_score_threshold=control_threshold,
                late_episode_fraction=late_episode_fraction,
            )
        manifest["control_diagnostics"] = controls
        screening_assessment = _build_screening_assessment(
            milestone_records, controls=controls
        )
        screening_assessment["assessment_source_commit"] = source
        manifest["screening_assessment"] = screening_assessment
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        report_path = run_dir / "RESULTS.md"
        report_path.write_text(
            _build_report(
                manifest=manifest,
                evaluation_results=evaluation_results,
                controls=controls,
            ),
            encoding="utf-8",
        )
        manifest["report_path"] = "RESULTS.md"
        manifest["report_sha256"] = sha256_file(report_path)
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise
