"""Test a frozen policy on 200 fresh seeds, then film its ten calmest landings."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MODEL = RESEARCH / "selected" / "slow_ppo_v3_seed42.zip"
PLAN = RESEARCH / "phase10_top10_from_200_plan.json"
RESULT = RESEARCH / "comparisons" / "phase10_seed42_200_test.json"
VIDEO = RESEARCH / "videos" / "top10_from_200_slow_ppo_v3_seed42.mp4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_seeds() -> set[int]:
    used: set[int] = set()
    for path in RESEARCH.glob("*seeds.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for values in data.values():
            if isinstance(values, list):
                used.update(v for v in values if isinstance(v, int))
    phase9 = RESEARCH / "phase9_five_models_plan.json"
    used.update(json.loads(phase9.read_text(encoding="utf-8"))["seeds"])
    return used


def calmness_score(episode: dict) -> float:
    """Lower is calmer: touchdown motion dominates, switches are a small tie breaker."""
    c = episode["first_leg_contact"]
    return float(
        (c["vx"] / 0.1) ** 2 + (c["vy"] / 0.1) ** 2
        + (c["angle"] / 0.1) ** 2 + (c["omega"] / 0.25) ** 2
        + 0.1 * episode["action_switch_rate"] ** 2
    )


def label(frame: np.ndarray, rank: int, test_index: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    caption = f"Selected {rank}/10  |  Test {test_index}/200"
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), caption, font=font)
    draw.rectangle((8, 8, 18 + box[2], 17 + box[3]), fill="black")
    draw.text((13, 12), caption, font=font, fill="white")
    return np.asarray(image)


def main() -> None:
    for path in (PLAN, RESULT, VIDEO):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    selected = json.loads((RESEARCH / "selected" / "models.json").read_text())
    expected_sha = selected["slow_ppo_v3_seed42"]["sha256"]
    if sha256(MODEL) != expected_sha:
        raise ValueError("Selected checkpoint hash changed")
    rng = random.Random(20261105)
    used = prior_seeds()
    seeds: list[int] = []
    while len(seeds) < 200:
        seed = rng.randrange(10_000_000, 2_000_000_000)
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    write_json(PLAN, {
        "model": str(MODEL), "model_sha256": expected_sha, "seeds": seeds,
        "selection_rule": "Among safe centered gentle landings lasting at least 7.5 s, "
                          "choose the ten lowest normalized first-contact squared "
                          "motion scores; action switching contributes only 0.1 * rate^2.",
        "selection_formula": "(vx/.1)^2 + (vy/.1)^2 + (angle/.1)^2 + "
                             "(omega/.25)^2 + .1 * action_switch_rate^2",
        "evaluation": "unmodified LunarLander-v3, deterministic policy",
    })
    model = PPO.load(MODEL, device="cpu")
    result = evaluate_model(model, seeds)
    eligible = [e for e in result["episodes"]
                if e["safe_centered_gentle"] and e["steps"] >= 375]
    if len(eligible) < 10:
        write_json(RESULT, {"plan_sha256": sha256(PLAN), "evaluation": result,
                            "selection_error": f"Only {len(eligible)} eligible episodes"})
        raise RuntimeError("Fewer than ten landings meet the predeclared rule")
    ranked = sorted(eligible, key=lambda e: (calmness_score(e), e["seed"]))[:10]
    by_seed = {seed: i + 1 for i, seed in enumerate(seeds)}
    selected_episodes = [{"rank": i, "test_episode": by_seed[e["seed"]],
                          "calmness_score": calmness_score(e), **e}
                         for i, e in enumerate(ranked, 1)]
    write_json(RESULT, {"plan_sha256": sha256(PLAN), "model_sha256": expected_sha,
                        "evaluation_seeds": seeds, "evaluation": result,
                        "eligible_count": len(eligible),
                        "selected_episodes": selected_episodes})
    print(f"200 tested: safe+center={result['safe_centered_rate']:.1%}, "
          f"gentle={result['safe_centered_gentle_rate']:.1%}, "
          f"eligible={len(eligible)}; saved {RESULT}", flush=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    fps = int(env.metadata["render_fps"])
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    temporary = VIDEO.with_name(VIDEO.stem + ".tmp.mp4")
    filmed = []
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for e in selected_episodes:
                observation, _ = env.reset(seed=e["seed"])
                writer.append_data(label(env.render(), e["rank"], e["test_episode"]))
                steps = 0
                total_reward = 0.0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(
                        int(np.asarray(action).item()))
                    writer.append_data(label(env.render(), e["rank"], e["test_episode"]))
                    steps += 1
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                if steps != e["steps"] or abs(total_reward - e["reward"]) > 1e-4:
                    raise RuntimeError(f"Filmed episode differs from test: seed {e['seed']}")
                filmed.append({"rank": e["rank"], "test_episode": e["test_episode"],
                               "seed": e["seed"], "steps": steps,
                               "duration_seconds": (steps + 1) / fps,
                               "reward": total_reward})
                print(f"Filmed selected {e['rank']}/10 (test {e['test_episode']}/200, "
                      f"{(steps + 1) / fps:.2f}s)", flush=True)
        temporary.replace(VIDEO)
    finally:
        env.close()
        temporary.unlink(missing_ok=True)
    write_json(VIDEO.with_suffix(".json"), {
        "source_result": str(RESULT), "video": str(VIDEO), "fps": fps,
        "duration_seconds": sum(e["duration_seconds"] for e in filmed),
        "selection": "10 calmest eligible episodes from 200, not consecutive",
        "filmed": filmed,
    })
    print(f"Saved {VIDEO}", flush=True)


if __name__ == "__main__":
    main()
