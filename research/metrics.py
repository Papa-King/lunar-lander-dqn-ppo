"""Shared, explicit evaluation for the extended Lunar Lander experiments."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np


SEED_FILE = Path(__file__).with_name("seeds.json")
ENV_ID = "LunarLander-v3"
PAD_MARGIN_WORLD = 0.75


def gentle_contact(state: dict | None) -> bool:
    """A fixed, interpretable touchdown threshold (normalized observation units)."""
    return bool(state and abs(state["vx"]) <= 0.1 and abs(state["vy"]) <= 0.1
                and abs(state["angle"]) <= 0.1 and abs(state["omega"]) <= 0.25)


def seeds_for(split: str, count: int | None = None) -> list[int]:
    groups = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    seeds = groups[split]
    if count is not None:
        if count < 1 or count > len(seeds):
            raise ValueError(f"{split} count must be between 1 and {len(seeds)}")
        seeds = seeds[:count]
    return seeds


def evaluate_model(model, seeds: list[int]) -> dict:
    """Evaluate a fixed policy on fixed starts; retain termination evidence."""
    env = gym.make(ENV_ID)
    episodes = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            reward_total = 0.0
            steps = switches = 0
            previous_action = None
            final_reward = 0.0
            first_contact = None
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                action = int(np.asarray(action).item())
                if previous_action is not None and action != previous_action:
                    switches += 1
                previous_action = action
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
                final_reward = float(reward)
                steps += 1
                if first_contact is None and (observation[6] > 0.5 or observation[7] > 0.5):
                    first_contact = {"step": steps, "vx": float(observation[2]),
                                     "vy": float(observation[3]),
                                     "angle": float(observation[4]),
                                     "omega": float(observation[5])}
            # In Gymnasium LunarLander-v3, the environment replaces the
            # terminal reward with +100 when the lander comes to rest safely.
            safe_landing = bool(terminated and final_reward == 100.0)
            base = env.unwrapped
            x_world = float(base.lander.position.x)
            centered = bool(base.helipad_x1 + PAD_MARGIN_WORLD <= x_world <=
                            base.helipad_x2 - PAD_MARGIN_WORLD)
            gentle = gentle_contact(first_contact)
            episodes.append({
                "seed": seed,
                "reward": reward_total,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_reward": final_reward,
                "safe_landing": safe_landing,
                "final_x_world": x_world,
                "pad_centered_with_leg_margin": centered,
                "gentle_first_contact": gentle,
                "safe_centered": safe_landing and centered,
                "safe_centered_gentle": safe_landing and centered and gentle,
                "first_leg_contact": first_contact,
                "two_leg_contact_at_end": bool(observation[6] > 0.5 and observation[7] > 0.5),
                "score_at_least_200": reward_total >= 200.0,
                "action_switch_rate": switches / max(steps - 1, 1),
            })
    finally:
        env.close()

    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=float)
    switches = np.asarray([episode["action_switch_rate"] for episode in episodes], dtype=float)
    centered_episodes = [e for e in episodes if e["safe_centered"] and e["first_leg_contact"]]
    centered_steps = [e["steps"] for e in centered_episodes]
    contact_vy = [abs(e["first_leg_contact"]["vy"]) for e in centered_episodes]
    contact_omega = [abs(e["first_leg_contact"]["omega"]) for e in centered_episodes]
    return {
        "environment": ENV_ID,
        "evaluation_episodes": len(episodes),
        "mean_reward": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "two_leg_landing_rate": float(np.mean([e["two_leg_contact_at_end"] for e in episodes])),
        "safe_landing_rate": float(np.mean([e["safe_landing"] for e in episodes])),
        "safe_centered_rate": float(np.mean([e["safe_centered"] for e in episodes])),
        "safe_centered_gentle_rate": float(np.mean([e["safe_centered_gentle"] for e in episodes])),
        "median_safe_centered_duration_seconds": float(np.median(centered_steps) / 50)
        if centered_steps else None,
        "median_safe_centered_abs_touchdown_vy": float(np.median(contact_vy))
        if contact_vy else None,
        "median_safe_centered_abs_touchdown_omega": float(np.median(contact_omega))
        if contact_omega else None,
        "score_200_rate": float(np.mean([e["score_at_least_200"] for e in episodes])),
        "mean_action_switch_rate": float(switches.mean()),
        "truncation_rate": float(np.mean([e["truncated"] for e in episodes])),
        "episodes": episodes,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
