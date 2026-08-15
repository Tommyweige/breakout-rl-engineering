"""Launch a visible ALE/Breakout-v5 preview with a random policy."""

import gymnasium as gym
import ale_py


gym.register_envs(ale_py)


def main() -> None:
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    observation, info = env.reset(seed=42)

    print("Observation shape:", observation.shape)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Action meanings:", env.unwrapped.get_action_meanings())

    try:
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()

