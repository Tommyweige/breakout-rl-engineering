"""Create article figures from the real Day 4 Breakout environment."""

from __future__ import annotations

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from breakout_env import make_breakout_env, make_breakout_raw_env


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SEED = 42


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB observation to an OpenCV BGR image."""

    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)


def gray_to_bgr(frame: np.ndarray, scale: int = 4) -> np.ndarray:
    """Convert a grayscale observation to a readable enlarged image."""

    image = np.asarray(frame, dtype=np.uint8)
    image = cv2.resize(
        image,
        (image.shape[1] * scale, image.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def add_title(canvas: np.ndarray, title: str, origin: tuple[int, int]) -> None:
    """Draw an ASCII title so the figure works without font setup."""

    cv2.putText(
        canvas,
        title,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )


def add_caption(canvas: np.ndarray, caption: str, origin: tuple[int, int]) -> None:
    """Draw a small caption below a panel."""

    cv2.putText(
        canvas,
        caption,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )


def place(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    """Place an image on a white canvas."""

    height, width = image.shape[:2]
    canvas[y : y + height, x : x + width] = image


def save_observation_pipeline() -> None:
    """Save raw, preprocessed, and stacked observations side by side."""

    raw_env = make_breakout_raw_env(render_mode="rgb_array")
    stacked_env = make_breakout_env(render_mode="rgb_array")
    try:
        raw_observation, _ = raw_env.reset(seed=SEED)
        stacked_observation, _ = stacked_env.reset(seed=SEED)
        for action in (1, 2, 2, 3):
            stacked_observation, _, terminated, truncated, _ = stacked_env.step(action)
            if terminated or truncated:
                stacked_observation, _ = stacked_env.reset(seed=SEED)

        raw_image = cv2.resize(
            rgb_to_bgr(raw_observation), (320, 420), interpolation=cv2.INTER_NEAREST
        )
        processed_image = gray_to_bgr(stacked_observation[-1])
        stack_image = np.hstack(
            [gray_to_bgr(frame, scale=2) for frame in stacked_observation]
        )

        canvas = np.full((640, 1600, 3), 248, dtype=np.uint8)
        add_title(canvas, "From raw pixels to a usable observation", (42, 48))
        place(canvas, raw_image, 58, 110)
        place(canvas, processed_image, 470, 152)
        place(canvas, stack_image, 850, 214)
        add_caption(canvas, "RAW: 210 x 160 x 3", (58, 570))
        add_caption(canvas, "GRAY + RESIZE: 84 x 84", (470, 570))
        add_caption(canvas, "4 RECENT FRAMES: (4, 84, 84)", (850, 570))
        cv2.imwrite(str(ASSETS / "day04-observation-pipeline.png"), canvas)
    finally:
        raw_env.close()
        stacked_env.close()


def save_frame_skip() -> None:
    """Save four raw frames while holding one action."""

    env = make_breakout_raw_env(render_mode="rgb_array")
    try:
        env.reset(seed=SEED)
        env.step(1)  # FIRE, so the ball is moving before the sampled action.

        frames: list[np.ndarray] = []
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(2)  # RIGHT
            frames.append(rgb_to_bgr(env.render()))
            if terminated or truncated:
                break

        while len(frames) < 4:
            frames.append(frames[-1].copy())

        panel_width, panel_height = 240, 315
        canvas = np.full((460, 1080, 3), 248, dtype=np.uint8)
        add_title(canvas, "One action held for four game updates", (38, 46))
        for index, frame in enumerate(frames):
            panel = cv2.resize(
                frame,
                (panel_width, panel_height),
                interpolation=cv2.INTER_NEAREST,
            )
            x = 38 + index * 260
            place(canvas, panel, x, 88)
            add_caption(canvas, f"RIGHT - update {index + 1}", (x, 432))
        cv2.imwrite(str(ASSETS / "day04-frame-skip.png"), canvas)
    finally:
        env.close()


def save_frame_stack() -> None:
    """Save the four frames contained in one real stacked observation."""

    env = make_breakout_env(render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=SEED)
        for action in (1, 2, 2, 3, 3, 2):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=SEED)

        canvas = np.full((410, 1060, 3), 248, dtype=np.uint8)
        add_title(canvas, "The observation keeps four recent frames", (38, 46))
        for index, frame in enumerate(np.asarray(observation)):
            panel = gray_to_bgr(frame, scale=3)
            x = 38 + index * 255
            place(canvas, panel, x, 88)
            add_caption(canvas, f"frame {index + 1}", (x + 8, 380))
        cv2.imwrite(str(ASSETS / "day04-frame-stack.png"), canvas)
    finally:
        env.close()


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    save_observation_pipeline()
    save_frame_skip()
    save_frame_stack()
    print("Created Day 4 figures:")
    for name in (
        "day04-observation-pipeline.png",
        "day04-frame-skip.png",
        "day04-frame-stack.png",
    ):
        print(f"- {ASSETS / name}")


if __name__ == "__main__":
    main()
