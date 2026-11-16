"""Validation-only probe of a fixed action hold time for the selected PPO."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from metrics import ENV_ID, PAD_MARGIN_WORLD, gentle_contact, seeds_for, write_json


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "research" / "selected" / "centered_ppo_v2_seed123.zip"
OUTPUT = ROOT / "research" / "diagnostics" / "action_hold_validation.json"


def evaluate(model: PPO, hold_frames: int, seeds: list[int]) -> dict:
    env = gym.make(ENV_ID)
    episodes = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            first_contact = None
            previous_action = None
            action = 0
            switches = steps = 0
            reward_total = 0.0
            done = False
            while not done:
                if steps % hold_frames == 0:
                    predicted, _ = model.predict(observation, deterministic=True)
                    action = int(np.asarray(predicted).item())
                if previous_action is not None and action != previous_action:
                    switches += 1
                previous_action = action
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
                steps += 1
                if first_contact is None and (observation[6] > 0.5 or observation[7] > 0.5):
                    first_contact = {"vx": float(observation[2]),
                                     "vy": float(observation[3]),
                                     "angle": float(observation[4]),
                                     "omega": float(observation[5])}
                done = terminated or truncated
            core = env.unwrapped
            x_world = float(core.lander.position.x)
            safe = bool(terminated and float(reward) == 100.0)
            centered = bool(core.helipad_x1 + PAD_MARGIN_WORLD <= x_world <=
                            core.helipad_x2 - PAD_MARGIN_WORLD)
            episodes.append({"seed": seed, "reward": reward_total,
                             "safe": safe, "centered": safe and centered,
                             "gentle": safe and centered and gentle_contact(first_contact),
                             "truncated": bool(truncated),
                             "action_switch_rate": switches / max(steps - 1, 1)})
    finally:
        env.close()
    return {"hold_frames": hold_frames, "episodes": episodes,
            "mean_reward": float(np.mean([e["reward"] for e in episodes])),
            "safe_rate": float(np.mean([e["safe"] for e in episodes])),
            "centered_rate": float(np.mean([e["centered"] for e in episodes])),
            "gentle_rate": float(np.mean([e["gentle"] for e in episodes])),
            "mean_action_switch_rate": float(np.mean([e["action_switch_rate"] for e in episodes]))}


def main() -> None:
    model = PPO.load(MODEL, device="cpu")
    results = [evaluate(model, frames, seeds_for("validation")) for frames in (1, 2, 3, 4)]
    write_json(OUTPUT, {"split": "validation", "model": str(MODEL), "results": results})
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
