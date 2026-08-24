"""Record a real Breakout DQN smoke run as a compact evidence GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.training import DQNConfig, DQNTrainer, TrainingStepSnapshot


class TrainingGifRecorder:
    """Sample raw environment renders and annotate them with live run state."""

    def __init__(
        self,
        *,
        total_steps: int,
        record_every: int,
        fps: int,
        max_width: int = 240,
        palette_colors: int = 96,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be greater than zero")
        if record_every < 1:
            raise ValueError("record_every must be greater than zero")
        if fps < 1:
            raise ValueError("fps must be greater than zero")
        if max_width < 1:
            raise ValueError("max_width must be greater than zero")
        if not 2 <= palette_colors <= 256:
            raise ValueError("palette_colors must be between 2 and 256")

        self.total_steps = total_steps
        self.record_every = record_every
        self.fps = fps
        self.max_width = max_width
        self.palette_colors = palette_colors
        self.frames: list[Image.Image] = []
        self.frame_steps: list[int] = []
        self._font = self._load_font(12)

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        candidates: list[Path] = []
        windows_root = os.environ.get("WINDIR")
        if windows_root:
            candidates.extend(
                [
                    Path(windows_root) / "Fonts" / "consola.ttf",
                    Path(windows_root) / "Fonts" / "arial.ttf",
                ]
            )
        candidates.append(Path("DejaVuSansMono.ttf"))
        for candidate in candidates:
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _rgb_image(frame: np.ndarray) -> Image.Image:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            raise TypeError("rendered frame must have dtype uint8")
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]
        elif array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                "rendered frame must have shape (H, W), (H, W, 3), or (H, W, 4)"
            )
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")

    def _annotate(
        self,
        frame: np.ndarray,
        snapshot: TrainingStepSnapshot,
    ) -> Image.Image:
        game = self._rgb_image(frame)
        scale = self.max_width / game.width
        resized_size = (
            self.max_width,
            max(1, round(game.height * scale)),
        )
        game = game.resize(resized_size, Image.Resampling.NEAREST)

        overlay_height = 42
        canvas = Image.new(
            "RGB",
            (game.width, game.height + overlay_height),
            color=(12, 12, 12),
        )
        canvas.paste(game, (0, overlay_height))
        draw = ImageDraw.Draw(canvas)
        if not snapshot.warmup_complete:
            phase = "warm-up"
        elif snapshot.optimizer_updated:
            phase = "optimizer update"
        else:
            phase = "collect"
        draw.text(
            (6, 4),
            f"step {snapshot.global_step}/{self.total_steps}  eps {snapshot.epsilon:.3f}",
            fill=(245, 245, 245),
            font=self._font,
        )
        draw.text(
            (6, 22),
            f"ep {snapshot.episode}  raw score {snapshot.current_raw_episode_return:.0f}  {phase}",
            fill=(245, 245, 245),
            font=self._font,
        )
        return canvas

    def on_step(
        self,
        snapshot: TrainingStepSnapshot,
        frame: np.ndarray | None,
    ) -> None:
        """Receive a frame after each real environment step."""

        if snapshot.global_step % self.record_every != 0:
            return
        if frame is None:
            raise RuntimeError(
                "the training environment did not return a render frame; "
                "a real gameplay GIF cannot be produced"
            )
        self.frames.append(self._annotate(frame, snapshot))
        self.frame_steps.append(snapshot.global_step)

    def save(self, output_path: Path) -> dict[str, Any]:
        """Write an optimized GIF and return measured artifact metadata."""

        if not self.frames:
            raise RuntimeError("no frames were recorded")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = round(1000 / self.fps)
        palette_frames = [
            frame.quantize(
                colors=self.palette_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            for frame in self.frames
        ]
        palette_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=palette_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        return {
            "path": output_path.as_posix(),
            "frame_count": len(self.frames),
            "frame_steps": [self.frame_steps[0], self.frame_steps[-1]],
            "fps": self.fps,
            "frame_duration_ms": duration_ms,
            "duration_seconds": len(self.frames) * duration_ms / 1000.0,
            "frame_size": list(self.frames[0].size),
            "palette_colors": self.palette_colors,
            "file_size_bytes": output_path.stat().st_size,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real Breakout DQN smoke loop and record a gameplay GIF."
    )
    parser.add_argument("--run-id", default="day12-gif-seed42")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day12/training-smoke.gif"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("assets/day12/training-smoke-gif.json"),
    )
    parser.add_argument("--total-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def _reproduction_command(args: argparse.Namespace) -> str:
    return (
        "python record_training_gif.py "
        f"--run-id {args.run_id} --total-steps {args.total_steps} "
        f"--seed {args.seed} --record-every {args.record_every} "
        f"--fps {args.fps} --device {args.device}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_path = args.run_dir / args.run_id
    if (run_path / "metrics.csv").exists():
        raise FileExistsError(
            f"run already exists: {run_path}; choose a new --run-id "
            "to avoid appending to an existing metrics file"
        )

    config = DQNConfig.smoke(
        total_steps=args.total_steps,
        device=args.device,
    ).with_overrides(seed=args.seed)
    recorder = TrainingGifRecorder(
        total_steps=config.total_steps,
        record_every=args.record_every,
        fps=args.fps,
    )
    env = make_breakout_env(render_mode="rgb_array")
    try:
        trainer = DQNTrainer(
            env,
            config,
            run_dir=run_path,
            on_step=recorder.on_step,
        )
        summary = trainer.train()
    finally:
        env.close()

    gif_metadata = recorder.save(args.output)
    metadata = {
        "run_id": args.run_id,
        "seed": args.seed,
        "environment": ENVIRONMENT_ID,
        "device": args.device,
        "total_steps": config.total_steps,
        "record_every": args.record_every,
        "source_render": "env.render() from make_breakout_env(render_mode='rgb_array')",
        "source_script": "record_training_gif.py",
        "trainer": "breakout_rl/training/dqn_trainer.py",
        "run_artifacts": {
            "config": (run_path / "config.json").as_posix(),
            "metrics": (run_path / "metrics.csv").as_posix(),
            "summary": (run_path / "summary.json").as_posix(),
        },
        "reproduction_command": _reproduction_command(args),
        "training_summary": summary,
        "gif": gif_metadata,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
