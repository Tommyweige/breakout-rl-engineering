"""Record real Breakout gameplay from a saved DQN checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.models.dqn import DQNNetwork
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


def _save_gif(frames: list[Image.Image], *, output: Path, fps: int) -> dict[str, Any]:
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
        "path": output.as_posix(),
        "frame_count": len(frames),
        "fps": fps,
        "frame_duration_ms": duration_ms,
        "duration_seconds": len(frames) * duration_ms / 1000.0,
        "frame_size": list(frames[0].size),
        "file_size_bytes": output.stat().st_size,
    }


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
    saved_config = payload.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}
    config = DQNConfig.from_dict(saved_config)
    actual_step = checkpoint_step(payload, checkpoint_path)

    env = make_breakout_env(render_mode="rgb_array")
    model = DQNNetwork(
        int(env.action_space.n),
        input_shape=tuple(int(value) for value in env.observation_space.shape),
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

    output_path = Path(output)
    metadata_file = Path(metadata_path)
    gif_metadata = _save_gif(frames, output=output_path, fps=fps)
    metadata: dict[str, Any] = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_step": actual_step,
        "training_run_id": checkpoint_path.parent.parent.name,
        "training_seed": saved_config.get("seed"),
        "replay_backend": saved_config.get("replay_backend", "cpu"),
        "training_config": saved_config,
        "model_config": {
            "num_actions": int(env.action_space.n),
            "input_shape": [int(value) for value in env.observation_space.shape],
            "hidden_dim": int(model.hidden_dim),
        },
        "environment": ENVIRONMENT_ID,
        "evaluation_seed": evaluation_seed,
        "evaluation_epsilon": 0.0,
        "device": str(resolved_device),
        "episodes": completed_episodes,
        "max_steps_per_episode": max_steps,
        "evaluation_steps": total_steps,
        "episode_returns": episode_returns,
        "record_every": record_every,
        "source_script": "record_checkpoint_gameplay.py",
        "render_source": "make_breakout_env(render_mode='rgb_array').render()",
        "reproduction_command": (
            "python record_checkpoint_gameplay.py "
            f"--checkpoint {checkpoint_path} --output {output_path} "
            f"--metadata {metadata_file} --device {device} "
            f"--evaluation-seed {evaluation_seed} --episodes {episodes} "
            f"--max-steps {max_steps} --record-every {record_every} --fps {fps}"
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
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
