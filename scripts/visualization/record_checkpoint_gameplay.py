"""Record real Breakout gameplay from a saved DQN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.models.factory import build_q_network, checkpoint_architecture
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import resolve_device


def checkpoint_step(payload: dict[str, Any], checkpoint_path: Path) -> int:
    raw_step = payload.get("global_step")
    if isinstance(raw_step, int) and raw_step >= 0:
        return raw_step
    match = re.search(r"step-(\d+)", checkpoint_path.stem)
    if match:
        return int(match.group(1))
    raise ValueError("checkpoint does not contain a recoverable global_step")


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _frame_image(frame: np.ndarray, *, width: int, header: str) -> Image.Image:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise TypeError("rendered frame must have dtype uint8")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("rendered frame must be grayscale, RGB, or RGBA")

    image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    scale = width / image.width
    image = image.resize(
        (width, max(1, round(image.height * scale))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (image.width, image.height + 24), color=(12, 12, 12))
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), header, fill=(245, 245, 245), font=_font(12))
    return canvas


def _save_gif(
    frames: list[Image.Image],
    *,
    output: Path,
    fps: int,
    artifact_path: str,
) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("no gameplay frames were recorded")
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / fps)
    palette_frames = [
        frame.quantize(
            colors=96,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    palette_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "path": artifact_path,
        "frame_count": len(frames),
        "fps": fps,
        "frame_duration_ms": duration_ms,
        "duration_seconds": len(frames) * duration_ms / 1000.0,
        "frame_size": list(frames[0].size),
        "file_size_bytes": output.stat().st_size,
    }


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip()).resolve()


def _git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _portable_path(path: Path, *, repo_root: Path | None) -> str:
    resolved = path.resolve()
    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_checkpoint(
    checkpoint: str | Path,
    *,
    output: str | Path,
    metadata_path: str | Path,
    device: str,
    evaluation_seed: int,
    episodes: int,
    max_steps: int,
    record_every: int,
    fps: int,
    max_width: int,
    contract_path: str | Path = "configs/eval/breakout_contract_v2.json",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if episodes < 1 or max_steps < 1 or record_every < 1 or fps < 1:
        raise ValueError("episodes, max_steps, record_every, and fps must be positive")

    resolved_device = resolve_device(device)
    payload = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    repo_root = _git_root(checkpoint_path.parent)
    checkpoint_artifact_path = _portable_path(checkpoint_path, repo_root=repo_root)
    output_path = Path(output)
    metadata_file = Path(metadata_path)
    output_artifact_path = _portable_path(output_path, repo_root=repo_root)
    metadata_artifact_path = _portable_path(metadata_file, repo_root=repo_root)
    saved_config = payload.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}
    config = DQNConfig.from_dict(saved_config)
    actual_step = checkpoint_step(payload, checkpoint_path)

    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    checkpoint_contract_id = payload.get("contract_id", saved_config.get("contract_id"))
    if (
        checkpoint_contract_id is not None
        and checkpoint_contract_id != contract.contract_id
    ):
        raise ValueError(
            "checkpoint Contract v2 id does not match the supplied contract"
        )
    env = make_breakout_env(
        render_mode="rgb_array",
        **breakout_environment_kwargs(contract),
    )
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}
    architecture = checkpoint_architecture(payload)
    hidden_dim_value = model_config.get("hidden_dim")
    hidden_dim = 512 if hidden_dim_value is None else int(hidden_dim_value)
    model = build_q_network(
        architecture,
        num_actions=int(env.action_space.n),
        input_shape=tuple(int(value) for value in env.observation_space.shape),
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    model.load_state_dict(payload["online_network"])
    model.eval()

    frames: list[Image.Image] = []
    completed_episodes = 0
    total_steps = 0
    episode_returns: list[float] = []
    try:
        with torch.no_grad():
            for episode_index in range(episodes):
                observation, _ = env.reset(seed=evaluation_seed + episode_index)
                episode_return = 0.0
                for episode_step in range(1, max_steps + 1):
                    state = observation_to_tensor(observation, device=resolved_device)
                    action = int(torch.argmax(model(state)[0]).item())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    total_steps += 1
                    if (
                        total_steps == 1
                        or total_steps % record_every == 0
                        or terminated
                        or truncated
                    ):
                        frame = env.render()
                        if frame is None:
                            raise RuntimeError("environment did not return rgb_array render")
                        frames.append(
                            _frame_image(
                                frame,
                                width=max_width,
                                header=(
                                    f"ckpt {actual_step:,}  eval step {total_steps:,}  "
                                    f"episode return {episode_return:.0f}"
                                ),
                            )
                        )
                    if terminated or truncated:
                        break
                completed_episodes += 1
                episode_returns.append(episode_return)
    finally:
        env.close()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)

    gif_metadata = _save_gif(
        frames,
        output=output_path,
        fps=fps,
        artifact_path=output_artifact_path,
    )
    metadata: dict[str, Any] = {
        "checkpoint": checkpoint_artifact_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_commit": _git_commit(repo_root),
        "checkpoint_step": actual_step,
        "training_run_id": checkpoint_path.parent.parent.name,
        "training_seed": saved_config.get("seed"),
        "algorithm": payload.get("algorithm", saved_config.get("algorithm", "dqn")),
        "architecture": architecture,
        "checkpoint_contract_id": checkpoint_contract_id,
        "replay_backend": saved_config.get("replay_backend", "cpu"),
        "training_config": saved_config,
        "model_config": {
            "num_actions": int(env.action_space.n),
            "input_shape": [int(value) for value in env.observation_space.shape],
            "hidden_dim": int(model.hidden_dim),
            "architecture": architecture,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "environment": ENVIRONMENT_ID,
        "contract_id": contract.contract_id,
        "contract_path": Path(contract_path).as_posix(),
        "contract": contract.to_dict(),
        "evaluation_seed": evaluation_seed,
        "evaluation_epsilon": 0.0,
        "device": str(resolved_device),
        "episodes": completed_episodes,
        "max_steps_per_episode": max_steps,
        "evaluation_steps": total_steps,
        "episode_returns": episode_returns,
        "record_every": record_every,
        "source_script": "record_checkpoint_gameplay.py",
        "render_source": (
            "make_breakout_env(render_mode='rgb_array', **breakout_environment_kwargs(contract)).render()"
        ),
        "reproduction_command": (
            "python -m scripts.visualization.record_checkpoint_gameplay "
            f"--checkpoint {checkpoint_artifact_path} --output {output_artifact_path} "
            f"--metadata {metadata_artifact_path} --device {device} "
            f"--contract {Path(contract_path).as_posix()} "
            f"--evaluation-seed {evaluation_seed} --episodes {episodes} "
            f"--max-steps {max_steps} --record-every {record_every} --fps {fps} "
            f"--max-width {max_width}"
        ),
        "gif": gif_metadata,
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-width", type=int, default=240)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = record_checkpoint(
        args.checkpoint,
        output=args.output,
        metadata_path=args.metadata,
        device=args.device,
        evaluation_seed=args.evaluation_seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        record_every=args.record_every,
        fps=args.fps,
        max_width=args.max_width,
        contract_path=args.contract,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
