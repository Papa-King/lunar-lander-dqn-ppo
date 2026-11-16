"""Beginner-friendly Lunar Lander training entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.utils import set_random_seed


ARTIFACT_DIR = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["random", "dqn", "ppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args()


def run_random_baseline(seed: int, episodes: int = 20) -> list[float]:
    env = gym.make("LunarLander-v3")
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        del observation
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        returns.append(total_reward)
    env.close()
    return returns


def main() -> None:
    args = parse_args()
    if args.timesteps < 0:
        raise ValueError("--timesteps must be zero or positive")
    if args.timesteps > 200_000:
        raise ValueError("This lab caps each run at 200,000 timesteps.")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    set_random_seed(args.seed)
    started = time.perf_counter()

    if args.algorithm == "random":
        returns = run_random_baseline(args.seed)
        summary = {
            "algorithm": "random",
            "seed": args.seed,
            "total_timesteps": 0,
            "training_seconds": 0.0,
            "baseline_episode_returns": returns,
        }
    else:
        env = Monitor(
            gym.make("LunarLander-v3"),
            filename=str(ARTIFACT_DIR / "training.monitor.csv"),
        )
        model_class = DQN if args.algorithm == "dqn" else PPO
        learning_rate = args.learning_rate or (1e-4 if args.algorithm == "dqn" else 3e-4)
        model = model_class(
            "MlpPolicy", env, verbose=1, seed=args.seed, learning_rate=learning_rate
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        training_seconds = time.perf_counter() - started
        model.save(ARTIFACT_DIR / "best_model")
        env.close()
        monitor_data = load_results(str(ARTIFACT_DIR))
        summary = {
            "algorithm": args.algorithm,
            "seed": args.seed,
            "learning_rate": learning_rate,
            "total_timesteps": args.timesteps,
            "training_seconds": round(training_seconds, 3),
            "episodes_completed": int(len(monitor_data)),
            "mean_training_episode_reward": round(float(monitor_data["r"].mean()), 3),
            "environment": "LunarLander-v3",
        }

    with (ARTIFACT_DIR / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
