"""Regenerate the genuine ALE Breakout clear fixtures.

Run from the repository root:

    python -m tests.fixtures.breakout_clear.generate_fixtures

Each fixture is a requested-action sequence that clears both walls in the
pinned ALE runtime when replayed from ``env.reset(seed=...)`` through the
repository's own evaluation environment factory. A search-based driver finds
the actions: it reads ALE RAM to track the ball and uses ALE state cloning
(including the sticky-action RNG) to discard any segment that loses a life.
That driver is fixture tooling only; it never touches the completion detector.

The ground truth stored with each fixture comes from the replay itself:
raw rewards, ALE lives and frame counters, and the number of bricks visible in
the RGB screen. The detector under test is never consulted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.completion import count_visible_bricks
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
)
from breakout_rl.vision_evaluation import make_vision_breakout_env

FIXTURE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIXTURE_DIR.parents[2]
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
# Breakout RAM fields read only by the fixture driver.
BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
BALL_X_RANGE = (56, 200)
PADDLE_CONTACT_Y = 178

FIXTURES: dict[str, dict[str, Any]] = {
    # Contract v2 through the DQN evaluation path (frame skip 4, no-op reset).
    "v2_full_clear": {
        "contract": "configs/eval/breakout_contract_v2.json",
        "factory": "make_breakout_env",
        "episode_seed": 101,
        "planner_seed": 0,
    },
    # Contract v3 through the pixel-controller evaluation path (one frame per action).
    "v3_full_clear": {
        "contract": "configs/eval/breakout_contract_v3.json",
        "factory": "make_vision_breakout_env",
        "episode_seed": 101,
        "planner_seed": 0,
    },
}


def make_fixture_env(spec: dict[str, Any]) -> Any:
    contract = load_evaluation_contract(REPOSITORY_ROOT / spec["contract"])
    if spec["factory"] == "make_breakout_env":
        return make_breakout_env(
            **breakout_environment_kwargs(contract, allow_contract_v3=True)
        )
    if spec["factory"] == "make_vision_breakout_env":
        return make_vision_breakout_env(contract)
    raise ValueError(f"unknown fixture factory {spec['factory']!r}")


def encode_actions(actions: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])
    return runs


def decode_actions(runs: list[list[int]]) -> list[int]:
    return [int(action) for action, count in runs for _ in range(int(count))]


def _tracker(ram: np.ndarray, previous: tuple[int, int] | None, offset: int) -> int:
    """Move the paddle under the predicted landing point plus an aiming offset."""

    ball_x, ball_y, paddle_x = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])
    if ball_y == 0:
        return FIRE
    landing = float(ball_x)
    if previous is not None and previous[1] and ball_y > previous[1]:
        velocity_x = ball_x - previous[0]
        velocity_y = ball_y - previous[1]
        landing = ball_x + velocity_x * max(0.0, PADDLE_CONTACT_Y - ball_y) / velocity_y
        span = BALL_X_RANGE[1] - BALL_X_RANGE[0]
        folded = (landing - BALL_X_RANGE[0]) % (2 * span)
        landing = BALL_X_RANGE[0] + (folded if folded <= span else 2 * span - folded)
    goal = paddle_x + offset
    return RIGHT if landing > goal + 1 else LEFT if landing < goal - 1 else NOOP


def _rebuild(spec: dict[str, Any], actions: list[int]) -> tuple[Any, float]:
    """Return a fresh environment that has executed ``actions`` from reset."""

    env = make_fixture_env(spec)
    env.reset(seed=spec["episode_seed"])
    score = 0.0
    for action in actions:
        score += float(env.step(action)[1])
    return env, score


def plan_full_clear(spec: dict[str, Any], log: Callable[[str], None]) -> list[int]:
    """Search for requested actions that clear both walls without losing a life.

    ALE ``restoreState`` does not reproduce every sticky-action detail, so the
    search environment only proposes segments. A separate committed
    environment, which only ever steps forward from reset, accepts a segment
    after actually executing it. Its result is therefore exactly what a replay
    from ``env.reset(seed)`` produces.
    """

    rng = np.random.default_rng(spec["planner_seed"])
    frame_skip = load_evaluation_contract(REPOSITORY_ROOT / spec["contract"]).frame_skip
    horizon = 1000 // frame_skip  # emulator frames allowed between paddle bounces
    actions: list[int] = []
    committed, score = _rebuild(spec, [])
    # The FIRE wrapper owns the initial serve; record what was requested.
    while committed._needs_fire:
        committed.step(NOOP)
        actions.append(NOOP)
    lives = committed.unwrapped.ale.lives()
    search = make_fixture_env(spec)
    search.reset(seed=spec["episode_seed"])
    ale = search.unwrapped.ale
    boundaries: list[int] = [len(actions)]  # action count at each accepted reward
    failures = 0

    def propose(state: Any) -> list[int] | None:
        """Chain surviving paddle bounces in the search env until a reward."""

        segment: list[int] = []
        for _ in range(60):
            for _ in range(24):
                ale.restoreState(state)
                search._needs_fire = False
                search._last_lives = lives
                offset = int(rng.integers(0, 15))
                part: list[int] = []
                previous: tuple[int, int] | None = None
                outcome = None
                for _ in range(horizon):
                    ram = ale.getRAM()
                    ball = (int(ram[BALL_X]), int(ram[BALL_Y]))
                    action = _tracker(ram, previous, offset)
                    _, reward, terminated, truncated, _ = search.step(action)
                    part.append(action)
                    if ale.lives() < lives or terminated or truncated:
                        break
                    after = int(ale.getRAM()[BALL_Y])
                    if reward:
                        outcome = "reward"
                        break
                    if previous is not None and after < ball[1] and ball[1] > 150:
                        outcome = "bounce"
                        break
                    previous = ball
                if outcome is not None:
                    break
            else:
                return None
            segment.extend(part)
            if outcome == "reward":
                return segment
            state = ale.cloneState(include_rng=True)
        return None

    try:
        while score < 864:
            if failures >= 30 and len(boundaries) > 1:
                boundaries.pop()  # backtrack one reward
                del actions[boundaries[-1]:]
                committed.close()
                committed, score = _rebuild(spec, actions)
                failures = 0
                log(f"backtracked to score {score:.0f}")
            segment = propose(committed.unwrapped.ale.cloneState(include_rng=True))
            if segment is None:
                failures += 1
                continue
            committed_ale = committed.unwrapped.ale
            gained, lost = 0.0, False
            for action in segment:
                _, reward, terminated, truncated, _ = committed.step(action)
                gained += float(reward)
                if committed_ale.lives() < lives or terminated or truncated:
                    lost = True
                    break
            if lost:
                failures += 1
                committed.close()
                committed, score = _rebuild(spec, actions)
                continue
            actions.extend(segment)
            score += gained
            if gained:
                boundaries.append(len(actions))
                failures = 0
                if len(boundaries) % 25 == 0:
                    log(f"score {score:.0f} after {len(actions)} agent steps")
        log(f"planned {len(actions)} agent steps to score {score:.0f}")
        return actions
    finally:
        committed.close()
        search.close()


def evade_action(ram: np.ndarray) -> int:
    """Filler policy after a scenario prefix: steer away from the ball."""

    return LEFT if int(ram[BALL_X]) >= int(ram[PADDLE_X]) + 8 else RIGHT


def replay_scenario(spec: dict[str, Any], prefix: list[int]) -> dict[str, Any]:
    """Replay ``prefix`` from reset, then evade until the episode ends.

    Returns detector-independent ground truth. Walls are counted from the RGB
    screen: a wall is destroyed when the brick count either jumps back up
    (the ROM redraws a fresh wall) or reaches zero. Rewards and lives come
    from the environment step results and ``ale.lives()``.
    """

    env = make_fixture_env(spec)
    try:
        env.reset(seed=spec["episode_seed"])
        ale = env.unwrapped.ale
        origin = ale.getEpisodeFrameNumber()
        bricks = count_visible_bricks(ale.getScreenRGB())
        if bricks != 108:
            raise RuntimeError("reset screen does not show a full wall")
        score = 0.0
        walls = 0
        wall_events: list[dict[str, Any]] = []
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            step += 1
            action = prefix[step - 1] if step <= len(prefix) else evade_action(ale.getRAM())
            _, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            previous, bricks = bricks, count_visible_bricks(ale.getScreenRGB())
            wall_done = (previous > 0 and bricks == 0) or (bricks > previous + 50)
            # A redraw right after an empty frame belongs to the same wall.
            if wall_done and not (bricks > previous + 50 and previous == 0):
                walls += 1
                wall_events.append(
                    {
                        "wall": walls,
                        "agent_step": step,
                        "emulator_frame": ale.getEpisodeFrameNumber() - origin,
                        "cumulative_raw_reward": score,
                        "lives": ale.lives(),
                        "screen_bricks_after": bricks,
                    }
                )
        second = next((event for event in wall_events if event["wall"] == 2), None)
        return {
            "prefix_agent_steps": len(prefix),
            "total_agent_steps": step,
            "total_emulator_frames": ale.getEpisodeFrameNumber() - origin,
            "final_cumulative_raw_reward": score,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_lives": ale.lives(),
            "walls_destroyed_on_screen": walls,
            "screen_bricks_final": bricks,
            "screen_wall_events": wall_events,
            "two_wall_clear": second is not None,
            "two_wall_clear_agent_step": second["agent_step"] if second else None,
            "two_wall_clear_emulator_frame": second["emulator_frame"] if second else None,
            "lives_at_two_wall_clear": second["lives"] if second else None,
        }
    finally:
        env.close()


def build_scenarios(spec: dict[str, Any], actions: list[int]) -> dict[str, Any]:
    full = replay_scenario(spec, actions)
    if not full["two_wall_clear"] or full["final_cumulative_raw_reward"] != 864.0:
        raise RuntimeError("full plan does not clear both walls on replay")
    # Reward steps of the plan, from a prefix-only replay.
    env = make_fixture_env(spec)
    try:
        env.reset(seed=spec["episode_seed"])
        reward_steps, score = [], 0.0
        for step, action in enumerate(actions, start=1):
            reward = float(env.step(action)[1])
            score += reward
            if reward:
                reward_steps.append((step, score))
    finally:
        env.close()
    first_wall_step = full["screen_wall_events"][0]["agent_step"]
    # Latest prefix cut that still leaves brick(s) after evasion: search the
    # window between the last two rewards first, then earlier windows.
    last_brick = None
    for k in range(2, 6):
        start_step, start_score = reward_steps[-k]
        for cut in range(reward_steps[-k + 1][0] - 1, start_step - 1, -5):
            candidate = replay_scenario(spec, actions[:cut])
            if candidate["final_cumulative_raw_reward"] == start_score:
                last_brick = candidate
                break
        if last_brick is not None:
            break
    if last_brick is None:
        raise RuntimeError("no prefix cut leaves exactly the last brick")
    scenarios = {
        "full_clear_then_game_over": full,
        "first_wall_only_then_game_over": replay_scenario(spec, actions[:first_wall_step]),
        "last_brick_left_then_game_over": last_brick,
    }
    first_wall = scenarios["first_wall_only_then_game_over"]
    last_brick = scenarios["last_brick_left_then_game_over"]
    if first_wall["walls_destroyed_on_screen"] != 1 or first_wall["two_wall_clear"]:
        raise RuntimeError("first-wall scenario did not stop after exactly one wall")
    if last_brick["two_wall_clear"] or last_brick["final_cumulative_raw_reward"] >= 864:
        raise RuntimeError("last-brick scenario accidentally cleared the second wall")
    return scenarios


def main() -> int:
    replan = "--replan" in sys.argv[1:]
    names = [arg for arg in sys.argv[1:] if arg != "--replan"] or list(FIXTURES)
    for name in names:
        spec = FIXTURES[name]
        path = FIXTURE_DIR / f"{name}.json"
        if path.is_file() and not replan:
            # Reuse the committed plan; only the ground truth is recomputed.
            actions = decode_actions(json.loads(path.read_text())["requested_actions_rle"])
        else:
            actions = plan_full_clear(spec, log=lambda line: print(f"{name}: {line}", flush=True))
        payload = {
            "schema_version": 1,
            "name": name,
            "description": (
                "Genuine ALE trajectory that clears both Breakout walls. Each scenario "
                "replays a prefix of requested_actions from env.reset(seed=episode_seed), "
                "then steers away from the ball until the episode ends."
            ),
            "environment": spec,
            "generator": "python -m tests.fixtures.breakout_clear.generate_fixtures",
            "requested_actions_rle": encode_actions(actions),
            "scenarios": build_scenarios(spec, actions),
        }
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        summary = {
            key: (value["final_cumulative_raw_reward"], value["two_wall_clear_agent_step"])
            for key, value in payload["scenarios"].items()
        }
        print(f"{name}: wrote {path.relative_to(REPOSITORY_ROOT)} {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
