"""Inspect validation episode endings without changing model selection or test data."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from metrics import ENV_ID, seeds_for, write_json


ROOT = Path(__file__).resolve().parent.parent


def diagnose(model: DQN, seeds: list[int]) -> dict:
    env = gym.make(ENV_ID)
    episodes = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            done = False
            rewards = []
            steps = 0
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
                rewards.append(float(reward))
                steps += 1
                done = terminated or truncated
            final_reward = rewards[-1]
            # Gymnasium LunarLander-v3 sets the terminal reward to exactly
            # +100 for rest/success and -100 for a crash or leaving the view.
            if truncated:
                outcome = "time_limit"
            elif final_reward == 100.0:
                outcome = "safe_rest"
            elif abs(float(observation[0])) >= 1.0:
                outcome = "out_of_bounds"
            elif final_reward == -100.0:
                outcome = "crash"
            else:
                outcome = "other_termination"
            episodes.append({
                "seed": seed, "outcome": outcome, "steps": steps,
                "reward": sum(rewards), "last_reward": final_reward,
                "last_observation": [float(v) for v in observation],
                "two_leg_contact_at_end": bool(observation[6] > 0.5 and observation[7] > 0.5),
            })
    finally:
        env.close()
    kinds = ("safe_rest", "crash", "out_of_bounds", "time_limit", "other_termination")
    return {"counts": {kind: sum(e["outcome"] == kind for e in episodes) for kind in kinds},
            "episodes": episodes}


def main() -> None:
    result = {"split": "validation", "count_per_model": 30, "models": {}}
    for train_seed in (42, 7, 123):
        path = ROOT / "research" / "runs" / f"dqn_zoo_seed{train_seed}" / "best_landing.zip"
        model = DQN.load(path, device="cpu")
        item = diagnose(model, seeds_for("validation"))
        result["models"][str(train_seed)] = item
        misleading = [e["seed"] for e in item["episodes"]
                      if e["two_leg_contact_at_end"] and e["outcome"] != "safe_rest"]
        print(train_seed, item["counts"], "two-leg but not safe:", misleading)
    output = ROOT / "research" / "diagnostics" / "validation_outcomes.json"
    write_json(output, result)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
