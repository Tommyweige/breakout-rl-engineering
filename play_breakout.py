"""Play two Atari Breakout environments side by side.

The left game is controlled by the existing random agent. The right game is
controlled by the human player through the keyboard.
"""

from __future__ import annotations

import tkinter as tk

import ale_py
import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)

FRAME_SCALE = 2
FRAME_DELAY_MS = 20


class HumanInput:
    """Track keyboard state and turn it into one Atari action per frame."""

    def __init__(self, action_indices: dict[str, int]) -> None:
        self.action_indices = action_indices
        self.left_pressed = False
        self.right_pressed = False
        self.fire_pending = False
        self.reset_pending = False
        self.quit_requested = False

    def handle_key(self, keysym: str, pressed: bool) -> None:
        key = keysym.lower()

        if key in {"left", "a"}:
            self.left_pressed = pressed
        elif key in {"right", "d"}:
            self.right_pressed = pressed
        elif pressed and key in {"space", "f"}:
            self.fire_pending = True
        elif pressed and key == "r":
            self.reset_pending = True
        elif pressed and key in {"escape", "q"}:
            self.quit_requested = True

    def next_action(self) -> int:
        """Return the human action for the next environment step."""

        if self.fire_pending:
            self.fire_pending = False
            return self.action_indices["FIRE"]

        if self.left_pressed and not self.right_pressed:
            return self.action_indices["LEFT"]
        if self.right_pressed and not self.left_pressed:
            return self.action_indices["RIGHT"]
        return self.action_indices["NOOP"]


def frame_to_photoimage(frame: np.ndarray, scale: int = FRAME_SCALE) -> tk.PhotoImage:
    """Convert an RGB NumPy frame into a Tkinter image without extra packages."""

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame, got shape {frame.shape}")

    scaled_frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    scaled_frame = np.ascontiguousarray(scaled_frame, dtype=np.uint8)
    height, width, _ = scaled_frame.shape
    ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + scaled_frame.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")


def action_indices(env: gym.Env) -> dict[str, int]:
    """Return action IDs by their ALE names instead of relying on magic numbers."""

    return {
        meaning: index
        for index, meaning in enumerate(env.unwrapped.get_action_meanings())
    }


def play_side_by_side(ai_env: gym.Env, human_env: gym.Env) -> None:
    """Run a random-agent game beside a keyboard-controlled game."""

    ai_observation, _ = ai_env.reset(seed=42)
    human_observation, _ = human_env.reset(seed=43)
    human_actions = action_indices(human_env)
    human_input = HumanInput(human_actions)
    action_meanings = ai_env.unwrapped.get_action_meanings()

    root = tk.Tk()
    root.title("Atari Breakout — AI vs Human")
    root.configure(bg="#171717")
    root.resizable(False, False)
    root.focus_force()

    title = tk.Label(
        root,
        text="Atari Breakout    |    AI / Random Agent ←       → Human",
        bg="#171717",
        fg="white",
        font=("Segoe UI", 14, "bold"),
        pady=8,
    )
    title.pack()

    games = tk.Frame(root, bg="#171717")
    games.pack(padx=8, pady=(0, 8))

    ai_panel = tk.Frame(games, bg="#262626")
    ai_panel.pack(side=tk.LEFT, padx=(0, 4))
    human_panel = tk.Frame(games, bg="#262626")
    human_panel.pack(side=tk.LEFT, padx=(4, 0))

    tk.Label(
        ai_panel,
        text="AI / Random Agent",
        bg="#262626",
        fg="#78b7ff",
        font=("Segoe UI", 11, "bold"),
        pady=5,
    ).pack()
    tk.Label(
        human_panel,
        text="Human",
        bg="#262626",
        fg="#ffcc66",
        font=("Segoe UI", 11, "bold"),
        pady=5,
    ).pack()

    ai_image_label = tk.Label(ai_panel, bg="black")
    ai_image_label.pack()
    human_image_label = tk.Label(human_panel, bg="black")
    human_image_label.pack()

    status = tk.Label(
        root,
        text="←/A and →/D: move    Space/F: fire    R: reset human game    Esc/Q: quit",
        bg="#171717",
        fg="#dddddd",
        font=("Segoe UI", 9),
        pady=8,
    )
    status.pack()

    def on_key_press(event: tk.Event) -> None:
        human_input.handle_key(event.keysym, pressed=True)

    def on_key_release(event: tk.Event) -> None:
        human_input.handle_key(event.keysym, pressed=False)

    root.bind_all("<KeyPress>", on_key_press)
    root.bind_all("<KeyRelease>", on_key_release)

    def close_window() -> None:
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)

    def update() -> None:
        nonlocal ai_observation, human_observation

        if human_input.quit_requested:
            close_window()
            return

        if human_input.reset_pending:
            human_observation, _ = human_env.reset()
            human_input.reset_pending = False

        ai_action = ai_env.action_space.sample()
        (
            ai_observation,
            ai_reward,
            ai_terminated,
            ai_truncated,
            _,
        ) = ai_env.step(ai_action)

        human_action = human_input.next_action()
        (
            human_observation,
            human_reward,
            human_terminated,
            human_truncated,
            _,
        ) = human_env.step(human_action)

        if ai_terminated or ai_truncated:
            ai_observation, _ = ai_env.reset()
        if human_terminated or human_truncated:
            human_observation, _ = human_env.reset()

        ai_image = frame_to_photoimage(ai_observation)
        human_image = frame_to_photoimage(human_observation)
        ai_image_label.configure(image=ai_image)
        human_image_label.configure(image=human_image)
        ai_image_label.image = ai_image
        human_image_label.image = human_image

        status.configure(
            text=(
                "AI: "
                f"{action_meanings[ai_action]} (reward {ai_reward:g})    |    "
                "Human: "
                f"{action_meanings[human_action]} (reward {human_reward:g})    |    "
                "←/A  →/D  Space/F  R reset  Esc/Q quit"
            )
        )
        root.after(FRAME_DELAY_MS, update)

    initial_ai_image = frame_to_photoimage(ai_observation)
    initial_human_image = frame_to_photoimage(human_observation)
    ai_image_label.configure(image=initial_ai_image)
    human_image_label.configure(image=initial_human_image)
    ai_image_label.image = initial_ai_image
    human_image_label.image = initial_human_image
    root.after(FRAME_DELAY_MS, update)
    root.mainloop()


def main() -> None:
    ai_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    human_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    try:
        observation, _ = ai_env.reset(seed=42)
        print("Observation shape:", observation.shape)
        print("Observation space:", ai_env.observation_space)
        print("Action space:", ai_env.action_space)
        print("Action meanings:", ai_env.unwrapped.get_action_meanings())
        print("\nLeft: AI random agent | Right: human player")
        print("Controls: Left/A, Right/D, Space/F to fire, R to reset, Esc/Q to quit")
        play_side_by_side(ai_env, human_env)
    finally:
        ai_env.close()
        human_env.close()


if __name__ == "__main__":
    main()
