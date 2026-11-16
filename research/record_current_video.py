"""Record one complete, real-time safe landing from the current best candidate."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "research" / "runs" / "dqn_zoo_lr0003_seed123" / "checkpoints" / "step_000100000.zip"
OUTPUT = ROOT / "research" / "videos" / "current_seed123_validation_249119.mp4"
SEED = 249119  # A safe 320-step validation episode with near-median reward.


def main() -> None:
    model = DQN.load(MODEL, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    fps = int(env.metadata["render_fps"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reward_total = 0.0
    steps = 0
    try:
        observation, _ = env.reset(seed=SEED)
        with imageio.get_writer(OUTPUT, fps=fps, codec="libx264", quality=8,
                                macro_block_size=16) as video:
            video.append_data(env.render())
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
                video.append_data(env.render())
                reward_total += float(reward)
                steps += 1
                done = terminated or truncated
        if not (terminated and float(reward) == 100.0):
            raise RuntimeError("Chosen episode did not end in a safe rest")
        duration = (steps + 1) / fps
        if duration >= 10:
            raise RuntimeError(f"Video is longer than 10 seconds: {duration:.2f}s")
        info = {"model": str(MODEL), "environment": "LunarLander-v3",
                "evaluation_seed": SEED, "steps": steps,
                "reward": reward_total, "safe_rest": True,
                "fps": fps, "duration_seconds": duration,
                "video": str(OUTPUT)}
        OUTPUT.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(info, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
