"""Add 200 fresh tests, rank all 400 landings, and film the calmest 20 in order."""

from __future__ import annotations

import json
import random
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from metrics import evaluate_model, write_json
from phase10_top10_from_200 import MODEL, RESEARCH, calmness_score, prior_seeds, sha256


OLD_RESULT = RESEARCH / "comparisons" / "phase10_seed42_200_test.json"
PLAN = RESEARCH / "phase11_top20_from_400_plan.json"
RESULT = RESEARCH / "comparisons" / "phase11_seed42_400_combined.json"
VIDEO = RESEARCH / "videos" / "top20_from_400_slow_ppo_v3_seed42.mp4"


def label(frame: np.ndarray, rank: int, round_number: int, episode_index: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    caption = f"Rank {rank}/20  |  Round {round_number}, test {episode_index}/200"
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), caption, font=font)
    draw.rectangle((8, 8, 18 + box[2], 17 + box[3]), fill="black")
    draw.text((13, 12), caption, font=font, fill="white")
    return np.asarray(image)


def main() -> None:
    for path in (PLAN, RESULT, VIDEO):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    old = json.loads(OLD_RESULT.read_text(encoding="utf-8"))
    old_seeds = old["evaluation_seeds"]
    if len(old_seeds) != 200 or len(set(old_seeds)) != 200:
        raise ValueError("First round must contain 200 unique starts")
    model_hash = sha256(MODEL)
    if model_hash != old["model_sha256"]:
        raise ValueError("First round used a different model checkpoint")

    used = prior_seeds() | set(old_seeds)
    rng = random.Random(20261106)
    new_seeds: list[int] = []
    while len(new_seeds) < 200:
        seed = rng.randrange(10_000_000, 2_000_000_000)
        if seed not in used:
            new_seeds.append(seed)
            used.add(seed)
    write_json(PLAN, {
        "purpose": "Second fresh 200 tests and top 20 across all 400",
        "model": str(MODEL), "model_sha256": model_hash,
        "first_result": str(OLD_RESULT), "first_result_sha256": sha256(OLD_RESULT),
        "first_seeds": old_seeds, "second_seeds": new_seeds,
        "selection_rule": "Reuse phase10 rule: safe centered gentle, duration >= 7.5 s; "
                          "sort all 400 by the same lowest calmness score",
        "selection_formula": "(vx/.1)^2 + (vy/.1)^2 + (angle/.1)^2 + "
                             "(omega/.25)^2 + .1 * action_switch_rate^2",
    })
    model = PPO.load(MODEL, device="cpu")
    second = evaluate_model(model, new_seeds)
    print(f"New 200: safe+center={second['safe_centered_rate']:.1%}, "
          f"gentle={second['safe_centered_gentle_rate']:.1%}", flush=True)

    old_episodes = old["evaluation"]["episodes"]
    if len(old_episodes) != 200 or len(second["episodes"]) != 200:
        raise ValueError("Expected exactly 400 total episodes")
    all_episodes = []
    for round_number, episodes in ((1, old_episodes), (2, second["episodes"])):
        for test_index, episode in enumerate(episodes, 1):
            all_episodes.append({"round": round_number, "test_episode": test_index,
                                 **episode})
    eligible = [e for e in all_episodes
                if e["safe_centered_gentle"] and e["steps"] >= 375]
    if len(eligible) < 20:
        write_json(RESULT, {"plan_sha256": sha256(PLAN), "second_evaluation": second,
                            "selection_error": f"Only {len(eligible)} eligible"})
        raise RuntimeError("Fewer than 20 episodes meet the frozen criteria")
    ranked = sorted(eligible, key=lambda e: (calmness_score(e), e["seed"]))[:20]
    chosen = [{"rank": i, "calmness_score": calmness_score(e), **e}
              for i, e in enumerate(ranked, 1)]
    write_json(RESULT, {
        "plan_sha256": sha256(PLAN), "model_sha256": model_hash,
        "round1_result_sha256": sha256(OLD_RESULT),
        "round1_summary": {k: v for k, v in old["evaluation"].items() if k != "episodes"},
        "round2_evaluation": second,
        "combined_episodes": 400, "combined_eligible_count": len(eligible),
        "combined_safe_centered_count": sum(e["safe_centered"] for e in all_episodes),
        "combined_gentle_count": sum(e["safe_centered_gentle"] for e in all_episodes),
        "selected_episodes": chosen,
    })
    print(f"Combined 400: eligible={len(eligible)}, selected 20; saved {RESULT}",
          flush=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    fps = int(env.metadata["render_fps"])
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    temporary = VIDEO.with_name(VIDEO.stem + ".tmp.mp4")
    filmed = []
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for e in chosen:
                observation, _ = env.reset(seed=e["seed"])
                writer.append_data(label(env.render(), e["rank"], e["round"],
                                         e["test_episode"]))
                steps = 0
                total_reward = 0.0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(
                        int(np.asarray(action).item()))
                    writer.append_data(label(env.render(), e["rank"], e["round"],
                                             e["test_episode"]))
                    steps += 1
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                if steps != e["steps"] or abs(total_reward - e["reward"]) > 1e-4:
                    raise RuntimeError(f"Filmed episode differs from test: seed {e['seed']}")
                filmed.append({"rank": e["rank"], "round": e["round"],
                               "test_episode": e["test_episode"], "seed": e["seed"],
                               "steps": steps, "duration_seconds": (steps + 1) / fps,
                               "reward": total_reward,
                               "calmness_score": e["calmness_score"]})
                print(f"Filmed rank {e['rank']}/20 from round {e['round']} "
                      f"test {e['test_episode']}/200", flush=True)
        temporary.replace(VIDEO)
    finally:
        env.close()
        temporary.unlink(missing_ok=True)
    write_json(VIDEO.with_suffix(".json"), {
        "source_result": str(RESULT), "video": str(VIDEO), "fps": fps,
        "duration_seconds": sum(e["duration_seconds"] for e in filmed),
        "selection": "Top 20 calmest eligible from all 400; ordered rank 1 to 20",
        "filmed": filmed,
    })
    print(f"Saved {VIDEO}", flush=True)


if __name__ == "__main__":
    main()
