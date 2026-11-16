"""Record one deterministic demonstration episode from the saved model."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO


ARTIFACT_DIR = Path("artifacts")
VIDEO_DIR = Path("demo_video")


def main() -> None:
    summary_path = ARTIFACT_DIR / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run train.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    algorithm = summary.get("algorithm")
    if algorithm not in {"dqn", "ppo"}:
        raise ValueError("A trained DQN or PPO model is required.")

    base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        base_env,
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix="landing",
    )
    model_class = DQN if algorithm == "dqn" else PPO
    model = model_class.load(ARTIFACT_DIR / "best_model.zip", env=env)

    observation, _ = env.reset(seed=2026)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
    env.close()
    print(f"Video saved in {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()

