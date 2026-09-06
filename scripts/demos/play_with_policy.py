"""Run a short Contract v2 Breakout smoke with a PyTorch or ONNX policy."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.artifacts import (
    repository_relative_path as _relative_path,
    repository_root as _repository_root,
    sha256_file as _sha256_file,
)
from breakout_rl.evaluation import load_dqn_checkpoint
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.inference import (
    ONNXRuntimePolicy,
    PyTorchPolicy,
    default_inference_spec,
    load_inference_spec,
    validate_action_meanings,
)


DEFAULT_CHECKPOINT = Path("assets/day21/models/final_model/model.pt")
DEFAULT_ONNX_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_CONTRACT = Path("configs/eval/breakout_contract_v2.json")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_OUTPUT = Path("assets/day23/onnx-gameplay-smoke.json")


def _runtime_metadata(*, device: torch.device, device_index: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_model = None
    if cuda_available:
        try:
            gpu_model = str(torch.cuda.get_device_name(device_index))
        except (RuntimeError, TypeError, ValueError):
            gpu_model = None
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_index": device_index,
        "gpu_model": gpu_model,
    }


def run_smoke(
    *,
    runtime: str,
    provider: str,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    onnx_model: str | Path = DEFAULT_ONNX_MODEL,
    contract_path: str | Path = DEFAULT_CONTRACT,
    spec_path: str | Path = DEFAULT_SPEC,
    output: str | Path = DEFAULT_OUTPUT,
    seed: int = 101,
    steps: int = 256,
    device: str = "cuda",
    device_index: int = 0,
) -> dict[str, Any]:
    if runtime not in {"pytorch", "onnx"}:
        raise ValueError("runtime must be pytorch or onnx")
    if provider not in {"cpu", "cuda"}:
        raise ValueError("provider must be cpu or cuda")
    if steps < 1:
        raise ValueError("steps must be positive")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")

    root = _repository_root()
    contract_file = Path(contract_path).resolve()
    contract = load_evaluation_contract(contract_file)
    validate_breakout_runtime_contract(contract)
    spec = load_inference_spec(spec_path) if spec_path else default_inference_spec()

    pytorch_runtime: dict[str, Any] | None = None
    if runtime == "pytorch":
        resolved_device = torch.device(device)
        loaded = load_dqn_checkpoint(
            checkpoint,
            device=resolved_device,
            env_factory=lambda: make_breakout_env(
                **breakout_environment_kwargs(contract)
            ),
        )
        policy: Any = PyTorchPolicy(
            loaded.model,
            device=resolved_device,
            spec=spec,
        )
        checkpoint_path = Path(checkpoint).resolve()
        model_metadata = {
            "path": _relative_path(checkpoint_path, root=root),
            "sha256": _sha256_file(checkpoint_path),
            "model_id": loaded.model_id,
            "checkpoint_metadata": dict(loaded.checkpoint_metadata),
        }
        pytorch_runtime = _runtime_metadata(
            device=resolved_device,
            device_index=device_index,
        )
    else:
        onnx_path = Path(onnx_model).resolve()
        policy = ONNXRuntimePolicy(
            onnx_path,
            provider=provider,
            device_index=device_index,
            spec=spec,
        )
        model_metadata = {
            "path": _relative_path(onnx_path, root=root),
            "sha256": _sha256_file(onnx_path),
        }

    env = make_breakout_env(
        render_mode=None,
        **breakout_environment_kwargs(contract),
    )
    requested_actions: Counter[str] = Counter()
    executed_actions: Counter[str] = Counter()
    auto_fire_reasons: Counter[str] = Counter()
    episode_return = 0.0
    completed_steps = 0
    terminated = False
    truncated = False
    try:
        observation, _ = env.reset(seed=seed)
        action_meanings = validate_action_meanings(
            env.unwrapped.get_action_meanings(),
            spec=spec,
        )
        for _step in range(steps):
            action = int(policy.select_action(observation))
            requested_actions[action_meanings[action]] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            executed_action = int(info.get("fire_reset_executed_action", action))
            executed_actions[action_meanings[executed_action]] += 1
            if info.get("fire_reset_auto"):
                reason = str(info.get("fire_reset_reason") or "unknown")
                auto_fire_reasons[reason] += 1
            episode_return += float(reward)
            completed_steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    runtime_metadata = (
        dict(policy.runtime_metadata)
        if runtime == "onnx"
        else {
            "requested_provider": "PyTorch",
            "actual_provider": "PyTorch",
            "pytorch": pytorch_runtime,
        }
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "day23_policy_gameplay_smoke",
        "status": "completed",
        "runtime": runtime,
        "requested_onnx_provider": provider if runtime == "onnx" else None,
        "runtime_metadata": runtime_metadata,
        "model": model_metadata,
        "inference_spec": {
            "contract_id": spec.contract_id,
            "input_name": spec.input_name,
            "output_name": spec.output_name,
            "action_meanings": list(spec.action_meanings),
            "normalization_divisor": spec.normalization_divisor,
            "path": _relative_path(Path(spec_path), root=root),
            "sha256": _sha256_file(Path(spec_path).resolve(), normalize_text=True),
        },
        "environment_contract": {
            "contract_id": contract.contract_id,
            "path": _relative_path(contract_file, root=root),
            "sha256": _sha256_file(contract_file),
            "semantics": contract.to_dict(),
        },
        "seed": seed,
        "max_steps": steps,
        "steps_completed": completed_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "raw_episode_return": episode_return,
        "requested_action_distribution": dict(sorted(requested_actions.items())),
        "executed_action_distribution": dict(sorted(executed_actions.items())),
        "auto_fire_reason_counts": dict(sorted(auto_fire_reasons.items())),
        "smoke_question": (
            "Can the selected policy execute Contract v2 Breakout steps and return "
            "valid actions without an inference integration error?"
        ),
        "reproduction_command": (
            "python -m scripts.demos.play_with_policy "
            f"--runtime {runtime} --provider {provider} --steps {steps} --seed {seed}"
        ),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            runtime=args.runtime,
            provider=args.provider,
            checkpoint=args.checkpoint,
            onnx_model=args.onnx_model,
            contract_path=args.contract,
            spec_path=args.spec,
            output=args.output,
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            device_index=args.device_index,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 23 gameplay smoke failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "runtime": result["runtime"],
                "requested_onnx_provider": result["requested_onnx_provider"],
                "steps_completed": result["steps_completed"],
                "raw_episode_return": result["raw_episode_return"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
