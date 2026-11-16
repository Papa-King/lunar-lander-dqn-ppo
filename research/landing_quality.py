"""Measure pad placement and touchdown state on unmodified LunarLander-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO

from metrics import ENV_ID, seeds_for, write_json


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "research" / "runs" / "dqn_zoo_lr0003_seed123" / "best_landing.zip"
# 0.75 world units covers the 20-pixel-wide legs (20/30 world units) with margin.
PAD_MARGIN_WORLD = 0.75


def run(model: DQN, *, continuous: bool, count: int, start_seed: int = 314159265) -> dict:
    env = gym.make(ENV_ID)
    rows = []
    try:
        for i in range(count):
            if continuous:
                observation, _ = env.reset(seed=start_seed if i == 0 else None)
                seed = None if i else start_seed
            else:
                seed = seeds_for("validation")[i]
                observation, _ = env.reset(seed=seed)
            first_contact = None
            reward_total = 0.0
            steps = 0
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
                reward_total += float(reward)
                steps += 1
                if first_contact is None and (observation[6] > 0.5 or observation[7] > 0.5):
                    first_contact = {
                        "step": steps, "x_normalized": float(observation[0]),
                        "vx_normalized": float(observation[2]),
                        "vy_normalized": float(observation[3]),
                        "angle_radians": float(observation[4]),
                        "angular_velocity_normalized": float(observation[5]),
                    }
                done = terminated or truncated
            core = env.unwrapped
            x_world = float(core.lander.position.x)
            safe = bool(terminated and float(reward) == 100.0)
            inside = bool(core.helipad_x1 + PAD_MARGIN_WORLD <= x_world <=
                          core.helipad_x2 - PAD_MARGIN_WORLD)
            rows.append({
                "episode": i + 1, "seed": seed, "steps": steps, "reward": reward_total,
                "safe_rest": safe, "time_limit": bool(truncated),
                "final_x_world": x_world,
                "pad_x1_world": float(core.helipad_x1),
                "pad_x2_world": float(core.helipad_x2),
                "pad_with_leg_margin": inside,
                "safe_and_inside_pad": safe and inside,
                "first_leg_contact": first_contact,
            })
    finally:
        env.close()
    return {
        "mode": "consecutive" if continuous else "validation",
        "model": str(model._model_path) if hasattr(model, "_model_path") else None,
        "pad_margin_world": PAD_MARGIN_WORLD,
        "count": count,
        "safe_rest_count": sum(r["safe_rest"] for r in rows),
        "safe_and_inside_pad_count": sum(r["safe_and_inside_pad"] for r in rows),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--mode", choices=["validation", "consecutive"], required=True)
    parser.add_argument("--start-seed", type=int, default=314159265)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_class = DQN if args.algorithm == "dqn" else PPO
    model = model_class.load(args.model, device="cpu")
    result = run(model, continuous=args.mode == "consecutive",
                 count=10 if args.mode == "consecutive" else 30,
                 start_seed=args.start_seed)
    result["model"] = str(args.model.resolve())
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("mode", "count", "safe_rest_count",
                                               "safe_and_inside_pad_count")}, indent=2))
    for row in result["episodes"]:
        if not row["safe_and_inside_pad"]:
            print(row["episode"], row["seed"], round(row["reward"], 1),
                  "safe", row["safe_rest"], "x", round(row["final_x_world"], 2),
                  "contact", row["first_leg_contact"])


if __name__ == "__main__":
    main()
