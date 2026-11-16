"""Evaluate one saved model using fixed classroom practice seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO


ARTIFACT_DIR = Path("artifacts")
OFFICIAL_SEEDS = [
    1101, 1129, 1151, 1171, 1201, 1229, 1259, 1289, 1301, 1327,
    1361, 1381, 1409, 1433, 1459, 1481, 1511, 1543, 1571, 1601,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, choices=[20])
    return parser.parse_args()


def load_model(algorithm: str, env: gym.Env):
    model_path = ARTIFACT_DIR / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError("Run train.py first; artifacts/best_model.zip is missing.")
    model_class = DQN if algorithm == "dqn" else PPO
    return model_class.load(model_path, env=env)


def main() -> None:
    parse_args()
    summary_path = ARTIFACT_DIR / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("artifacts/training_summary.json is missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    algorithm = summary.get("algorithm")
    if algorithm not in {"dqn", "ppo"}:
        raise ValueError("Train a DQN or PPO model before practice evaluation.")

    env = gym.make("LunarLander-v3")
    model = load_model(algorithm, env)
    episodes = []

    for seed in OFFICIAL_SEEDS:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        switches = 0
        previous_action = None

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            action_value = int(np.asarray(action).item())
            if previous_action is not None and action_value != previous_action:
                switches += 1
            previous_action = action_value
            observation, reward, terminated, truncated, _ = env.step(action_value)
            total_reward += float(reward)
            steps += 1

        two_leg_landing = bool(observation[6] > 0.5 and observation[7] > 0.5)
        episodes.append({
            "seed": seed,
            "reward": round(total_reward, 3),
            "steps": steps,
            "score_at_least_200": total_reward >= 200.0,
            "two_leg_landing": two_leg_landing,
            "action_switch_rate": round(switches / max(steps - 1, 1), 4),
        })

    env.close()
    rewards = np.array([item["reward"] for item in episodes])
    switch_rates = np.array([item["action_switch_rate"] for item in episodes])
    result = {
        "environment": "LunarLander-v3",
        "algorithm": algorithm,
        "evaluation_episodes": len(episodes),
        "mean_reward": round(float(rewards.mean()), 3),
        "reward_std": round(float(rewards.std()), 3),
        "score_200_rate": round(float((rewards >= 200).mean()), 3),
        "two_leg_landing_rate": round(
            float(np.mean([item["two_leg_landing"] for item in episodes])), 3
        ),
        "mean_action_switch_rate": round(float(switch_rates.mean()), 4),
        "training": summary,
        "episodes": episodes,
    }
    Path("evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
