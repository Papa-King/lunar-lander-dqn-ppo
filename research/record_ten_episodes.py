"""Record ten consecutive, unselected deterministic episodes in one video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "research" / "runs" / "dqn_zoo_lr0003_seed123" / "checkpoints" / "step_000100000.zip"
OUTPUT = ROOT / "research" / "videos" / "ten_consecutive_seed123_start314159265.mp4"
START_SEED = 314159265
EPISODES = 10


def labelled(frame: np.ndarray, index: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    caption = f"Episode {index}/{EPISODES}"
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), caption, font=font)
    draw.rectangle((8, 8, 18 + box[2], 17 + box[3]), fill="black")
    draw.text((13, 12), caption, font=font, fill="white")
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Already recorded: {output}")
    model_class = DQN if args.algorithm == "dqn" else PPO
    model = model_class.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    fps = int(env.metadata["render_fps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    records = []
    total_frames = 0
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8) as video:
            for index in range(1, EPISODES + 1):
                # Seed the RNG once. Subsequent reset() calls continue the same
                # environment RNG sequence with no re-seeding or selection.
                observation, _ = env.reset(seed=args.start_seed if index == 1 else None)
                video.append_data(labelled(env.render(), index))
                total_frames += 1
                reward_total = 0.0
                steps = 0
                done = False
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(
                        int(np.asarray(action).item())
                    )
                    video.append_data(labelled(env.render(), index))
                    total_frames += 1
                    reward_total += float(reward)
                    steps += 1
                    done = terminated or truncated
                if truncated:
                    outcome = "time_limit"
                elif float(reward) == 100.0:
                    outcome = "safe_rest"
                elif abs(float(observation[0])) >= 1.0:
                    outcome = "out_of_bounds"
                elif float(reward) == -100.0:
                    outcome = "crash"
                else:
                    outcome = "other"
                record = {"episode": index, "steps": steps,
                          "reward": reward_total, "outcome": outcome,
                          "final_two_leg_contact": bool(observation[6] > 0.5 and observation[7] > 0.5)}
                records.append(record)
                print(json.dumps(record), flush=True)
        temporary.replace(output)
        summary = {
            "video": str(output), "model": str(model_path), "algorithm": args.algorithm,
            "environment": "LunarLander-v3", "initial_reset_seed": args.start_seed,
            "subsequent_resets": "unseeded; continue the same environment RNG stream",
            "deterministic_actions": True, "fps": fps,
            "frames": total_frames, "duration_seconds": total_frames / fps,
            "counts": {kind: sum(r["outcome"] == kind for r in records)
                       for kind in ("safe_rest", "crash", "out_of_bounds", "time_limit", "other")},
            "episodes": records,
        }
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {output}")
        print(f"Duration: {summary['duration_seconds']:.2f}s; outcomes: {summary['counts']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
