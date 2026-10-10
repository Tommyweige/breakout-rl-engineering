"""Frozen Laya preflight, smoke gate and matched-contract evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.completion import read_ale_episode_frame
from breakout_rl.evaluation import (
    evaluate_policy, write_evaluation_artifacts,
    _resolved_action_from_info, _capture_source_provenance,
)
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs, load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.laya_vision_agent import (
    LAYA_CODE_REVISION, LAYA_MODEL_ID, LAYA_MODEL_REVISION,
    LayaVisionPolicy, load_laya_agent,
)
from breakout_rl.inference import ONNXRuntimePolicy, load_inference_spec

PILOT_SEEDS = (101, 202, 303)


def _unsupported_cuda(error: RuntimeError, device: str) -> bool:
    return device == "cuda" and any(message in str(error).lower() for message in (
        "no kernel image", "not compiled with cuda", "cuda is not supported", "invalid device function",
    ))


class _DQNBaseline:
    policy_type = "dqn-onnx"

    def __init__(self, policy):
        self.policy = policy

    def select_action(self, observation, *, rng):
        return self.policy.select_action(observation)


def latency_summary(decisions: list[dict]) -> dict:
    result = {"decisions": len(decisions)}
    for name in ("rgb_conversion_seconds", "model_seconds", "predict_non_model_seconds", "decision_seconds"):
        values = [row[name] for row in decisions if row[name] is not None]
        result[name] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None
    elapsed = sum(row["decision_seconds"] for row in decisions)
    result["decisions_per_second"] = len(decisions) / elapsed if elapsed else None
    entropies = [row["entropy_nats"] for row in decisions if row["entropy_nats"] is not None]
    result["mean_entropy_nats"] = float(np.mean(entropies)) if entropies else None
    confidences = [row["confidence"] for row in decisions if row["confidence"] is not None]
    result["mean_reported_confidence"] = float(np.mean(confidences)) if confidences else None
    result["requested_choice_distribution"] = {
        name: sum(row["choice"] == name for row in decisions) for name in ("NOOP", "FIRE", "RIGHT", "LEFT")
    }
    result["left_right_oscillations"] = sum(
        a.get("seed") == b.get("seed") and
        (a["choice"], b["choice"]) in (("LEFT", "RIGHT"), ("RIGHT", "LEFT"))
        for a, b in zip(decisions, decisions[1:])
    )
    return result


def evaluate_dqn_baseline(path: Path, contract_path: Path, seeds: tuple[int, ...], output_dir: Path) -> dict:
    contract = load_evaluation_contract(contract_path)
    if contract.frame_skip != 4:
        return {"status": "blocked", "reason": "v3 predictive-controller comparison is optional and unrun."}
    if not path.is_file():
        return {"status": "blocked", "reason": f"Missing DQN ONNX asset: {path}"}
    spec = load_inference_spec()
    model_metadata = json.loads(path.with_suffix(".onnx.metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != model_metadata["model_sha256"] or spec.environment_contract_sha256 != hashlib.sha256(contract_path.read_bytes()).hexdigest():
        raise ValueError("DQN model or inference contract hash does not match its provenance")
    onnx_policy = ONNXRuntimePolicy(path, provider="cpu", intra_op_num_threads=1, spec=spec)
    baseline_metadata = {"evaluation_contract_path": contract_path.as_posix(), "evaluation_contract": contract.to_dict(),
                         "evaluation_seed_subset": list(seeds), "onnx_runtime": onnx_policy.runtime_metadata}
    result = evaluate_policy(None, episodes=1, seeds=seeds, device="cpu",
                             env_factory=lambda: make_breakout_env(**breakout_environment_kwargs(contract)),
                             model_id="day21-final-long-training-seed2022-onnx",
                             policy_factory=lambda env: _DQNBaseline(onnx_policy),
                             metadata=baseline_metadata, training_metadata=model_metadata["source_model"],
                             checkpoint_metadata=model_metadata)
    write_evaluation_artifacts(result, output_dir)
    return {"status": "complete", "results": result.to_dict(), "model_sha256": digest}


def run_laya_evaluation(*, output_dir: Path, mode: str = "pilot", device: str = "cuda",
                        contract_path: Path = Path("configs/eval/breakout_contract_v2.json"),
                        baseline_onnx: Path = Path("web/public/models/final_model/model.onnx"),
                        max_pilot_hours: float = 1.0) -> dict:
    if mode not in ("smoke", "seed101", "pilot", "full"):
        raise ValueError("Unknown Laya evaluation mode")
    if not np.isfinite(max_pilot_hours) or max_pilot_hours <= 0:
        raise ValueError("max_pilot_hours must be positive and finite")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing research artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    seeds = (101,) if mode == "seed101" else contract.concrete_episode_seeds if mode == "full" else PILOT_SEEDS
    kwargs = breakout_environment_kwargs(contract, allow_contract_v3=True)
    factory = lambda: make_breakout_env(render_mode="rgb_array", **kwargs)
    resolved_device = device
    if device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
    metadata = {
        "evaluation_contract_path": contract_path.as_posix(),
        "evaluation_contract": contract.to_dict(),
        "evaluation_seed_subset": list(seeds),
        "model_id": LAYA_MODEL_ID, "model_revision": LAYA_MODEL_REVISION,
        "laya_code_revision": LAYA_CODE_REVISION,
        "policy_input": "one ALE-rendered RGB image; no RAM, frame stack or detector input",
        "requested_device": device, "resolved_device": resolved_device,
        "cpu_fallback": device != resolved_device,
        "dependencies": {name: version(name) for name in ("torch", "torchvision", "laya", "transformers", "huggingface_hub", "gymnasium", "ale-py", "numpy", "pillow")},
        "platform": platform.platform(),
        "max_pilot_hours": max_pilot_hours,
        "confidence_caveat": "Upstream probabilities are not established as calibrated on this ALE domain.",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "source_provenance": _capture_source_provenance(Path(__file__).resolve().parents[1]),
        "policy_file_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "breakout_rl/laya_vision_agent.py", "breakout_rl/laya_evaluation.py",
            "scripts/evaluation/evaluate_laya_vision.py", "requirements-laya.txt")},
    }
    evidence = {"status": "preflight", "metadata": metadata, "smoke": None, "evaluation": None,
                "baseline": {"status": "unrun", "reason": "Laya evaluation must pass before matched DQN comparison."}}
    trace_path = output_dir / "trace.jsonl"

    def save():
        evidence["wall_clock_seconds"] = time.perf_counter() - started
        (output_dir / "run.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        _write_report(output_dir, evidence)

    save()
    try:
        try:
            agent = load_laya_agent(resolved_device)
        except RuntimeError as error:
            if not _unsupported_cuda(error, resolved_device):
                raise
            metadata["gpu_load_failure"] = str(error)
            resolved_device = "cpu"
            metadata.update(resolved_device="cpu", cpu_fallback=True)
            agent = load_laya_agent("cpu")
        metadata["checkpoint_source"] = agent.source
        metadata["model_config"] = agent.cfg
        metadata["parameters"] = sum(p.numel() for p in agent.model.parameters())
        metadata["dtype"] = str(next(agent.model.parameters()).dtype)
        if resolved_device.startswith("cuda"):
            metadata["gpu"] = torch.cuda.get_device_name(resolved_device)
            metadata["gpu_total_bytes"] = torch.cuda.get_device_properties(resolved_device).total_memory
            torch.cuda.reset_peak_memory_stats(resolved_device)
        policy = None
        with trace_path.open("w", encoding="utf-8") as trace:
            def observe(row):
                decision = policy.decisions[-1]
                decision.update(row)
                trace.write(json.dumps(decision) + "\n")

            env = factory()
            try:
                observation, _ = env.reset(seed=101)
                from ale_py import roms
                rom_path = Path(roms.get_rom_path("breakout"))
                metadata["rom_sha256"] = hashlib.sha256(rom_path.read_bytes()).hexdigest()
                metadata["rom_checksum_source"] = "ale_py.roms.get_rom_path('breakout')"
                origin = read_ale_episode_frame(env)
                policy = LayaVisionPolicy(agent, env.render, env.unwrapped.get_action_meanings())
                from PIL import Image
                from laya.vlm import vlm_prefix
                preprocessing_times = []
                frame = Image.fromarray(env.render())
                for _ in range(10):
                    prep_started = time.perf_counter()
                    vlm_prefix(agent.processor, [frame], agent.prep)
                    preprocessing_times.append(time.perf_counter() - prep_started)
                metadata["image_preprocessing_seconds"] = {
                    "scope": "pinned upstream vlm_prefix: image processor and prefix preparation, 10 fixed-frame calls",
                    "p50": float(np.percentile(preprocessing_times, 50)),
                    "p95": float(np.percentile(preprocessing_times, 95)),
                }
                # Warm-up is excluded from measured smoke latency.
                try:
                    policy.select_action(observation, rng=np.random.default_rng(101))
                except RuntimeError as error:
                    if not _unsupported_cuda(error, resolved_device):
                        raise
                    metadata["gpu_inference_failure"] = str(error)
                    resolved_device = "cpu"
                    agent = load_laya_agent("cpu")
                    metadata.update(resolved_device="cpu", cpu_fallback=True,
                                    dtype=str(next(agent.model.parameters()).dtype))
                    policy = LayaVisionPolicy(agent, env.render, env.unwrapped.get_action_meanings())
                    policy.select_action(observation, rng=np.random.default_rng(101))
                policy.decisions.clear()
                policy.model_latencies.clear()
                for step in range(100):
                    if step < 3:
                        from PIL import Image
                        Image.fromarray(env.render()).save(output_dir / f"smoke-frame-{step:03d}.png")
                    action = policy.select_action(observation, rng=np.random.default_rng(101))
                    observation, _, terminated, truncated, info = env.step(action)
                    executed, auto_fire, _ = _resolved_action_from_info(info, requested_action=action, action_count=env.action_space.n)
                    observe({"phase": "smoke", "seed": 101, "agent_step": step + 1,
                             "requested_action": action, "executed_action": executed,
                             "auto_fire": auto_fire, "emulator_frame": read_ale_episode_frame(env)})
                    if terminated or truncated:
                        raise RuntimeError("Smoke episode ended before 100 decisions; no complete smoke gate")
                evidence["smoke"] = {"status": "passed", "complete_episode": False,
                                     "emulator_frames": read_ale_episode_frame(env) - origin,
                                     "latency": latency_summary(policy.decisions)}
            finally:
                env.close()
            save()
            p95 = evidence["smoke"]["latency"]["decision_seconds"]["p95"]
            estimated_seconds = p95 * int(contract.time_limit_semantics["agent_step_limit"]) * len(seeds)
            evidence["gate"] = {"estimated_max_episode_budget_seconds": estimated_seconds,
                                "budget_seconds": max_pilot_hours * 3600,
                                "uses_actual_episode_limit": True}
            if mode == "smoke":
                evidence["status"] = "smoke_only"
            elif estimated_seconds > max_pilot_hours * 3600:
                evidence["status"] = "blocked"
                evidence["blocker"] = "Measured p95 decision latency exceeds the predeclared evaluation compute budget. Frame skip unchanged."
            else:
                def policy_factory(live_env):
                    nonlocal policy
                    policy = LayaVisionPolicy(agent, live_env.render, live_env.unwrapped.get_action_meanings())
                    return policy

                result = evaluate_policy(None, episodes=1, seeds=seeds, device=resolved_device,
                                         env_factory=factory, policy_factory=policy_factory,
                                         step_callback=lambda row: observe(dict(row, phase="evaluation")),
                                         model_id=LAYA_MODEL_ID, metadata=metadata,
                                         evaluation_id=f"issue69-laya-{mode}")
                write_evaluation_artifacts(result, output_dir / "laya")
                evidence["evaluation"] = {"results": result.to_dict(), "latency": latency_summary(policy.decisions)}
                evidence["status"] = "evaluation_complete"
        if mode != "smoke":
            evidence["baseline"] = evaluate_dqn_baseline(baseline_onnx, contract_path, seeds, output_dir / "dqn")
            if evidence["status"] == "evaluation_complete" and evidence["baseline"]["status"] != "complete" and contract.frame_skip == 4:
                evidence["status"] = "blocked"
                evidence["blocker"] = evidence["baseline"]["reason"]
        if resolved_device.startswith("cuda"):
            evidence["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(resolved_device)
            evidence["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(resolved_device)
    except Exception as error:
        evidence["status"] = "blocked"
        evidence["blocker"] = f"{type(error).__name__}: {error}"
        save()
        raise
    save()
    return evidence


def _write_report(output_dir: Path, evidence: dict) -> None:
    meta = evidence["metadata"]
    lines = ["# Laya Vision frozen-policy feasibility", "", f"Status: {evidence['status']}", "",
             f"Contract: {meta['evaluation_contract']['contract_id']}; declared seeds: {meta['evaluation_seed_subset']}.",
             f"Model revision: `{LAYA_MODEL_REVISION}`; code revision: `{LAYA_CODE_REVISION}`.",
             "", "| Policy | Contract | Episodes | Raw mean | Clear status |", "|---|---|---:|---:|---|"]
    for name, value in (("Laya", evidence.get("evaluation")), ("DQN", evidence.get("baseline"))):
        results = (value or {}).get("results")
        if results:
            rows = results["per_episode"]
            cleared = sum(row["cleared"] is True for row in rows)
            unavailable = sum(row["cleared"] is None for row in rows)
            mean = sum(row["episode_return"] for row in rows) / len(rows)
            lines.append(f"| {name} | {meta['evaluation_contract']['contract_id']} | {len(rows)} | {mean:.2f} | {cleared}/{len(rows)}; unknown={unavailable} |")
        else:
            lines.append(f"| {name} | {meta['evaluation_contract']['contract_id']} | 0 | unavailable | blocked/unrun |")
    lines.extend(["", f"Blocker: {evidence.get('blocker', (evidence.get('baseline') or {}).get('reason', 'none'))}",
                  "", "Smoke is a truncated 100-decision diagnostic, not a complete episode. A budget gate uses measured p95 times the unchanged episode limit; it is an estimate, not a runtime timeout.",
                  "Timing: rgb_conversion measures render/copy/PIL; model_seconds includes vision tower, connector and decision forward hooks. A separate fixed-frame vlm_prefix benchmark measures upstream image preprocessing; predict_non_model includes all remaining API work.",
                  "Confidence is unvalidated on this ALE domain. Upstream normalized score 0.20 is not a clear rate; emulator frames and inference wall time are distinct from human timed records.",
                  "No claim of zero-shot unseen-game transfer: upstream weights include Breakout training data. Weights: CC BY-NC-SA 4.0; code: Apache 2.0.",
                  "", "Next experiment: complete any blocked matched-contract baseline/gated seeds before considering a separately specified temporal ablation."])
    if evidence.get("smoke"):
        latency = evidence["smoke"]["latency"]
        timing = latency["decision_seconds"]
        lines.extend(["", f"Measured smoke: {latency['decisions']} decisions; p50={timing['p50']:.4f}s, p95={timing['p95']:.4f}s; {latency['decisions_per_second']:.2f} decisions/s.",
                      f"Device={meta['resolved_device']}; dtype={meta.get('dtype')}; peak allocated VRAM={evidence.get('peak_gpu_allocated_bytes', 'unavailable')} bytes.",
                      f"Gate: {evidence.get('gate', {})}.",
                      "One-frame control remains wall-clock constrained; the evaluator never changes the contract frame skip."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
