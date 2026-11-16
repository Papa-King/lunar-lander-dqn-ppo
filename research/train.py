"""Isolated DQN experiments with fixed validation and resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import gymnasium as gym
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from metrics import ENV_ID, evaluate_model, seeds_for, write_json
from quality_reward import QualityRewardWrapper


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True, help="Absolute target, including resumed steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--validation-count", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reward-mode", choices=["original", "quality_v1"], default="original")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_every <= 0:
        parser.error("--timesteps and --eval-every must be positive")
    if args.learning_rate <= 0 or args.batch_size <= 0 or args.buffer_size <= 0:
        parser.error("learning rate, batch size, and buffer size must be positive")
    if args.learning_starts < 0 or args.target_update_interval <= 0:
        parser.error("learning starts cannot be negative; target update interval must be positive")
    if not 0 <= args.exploration_fraction <= 1 or not 0 <= args.exploration_final_eps <= 1:
        parser.error("exploration values must be between 0 and 1")
    if args.gradient_steps == 0 or args.gradient_steps < -1 or any(n <= 0 for n in args.net_arch):
        parser.error("gradient steps must be -1 or positive; network layers must be positive")
    seeds_for("validation", args.validation_count)
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def settings(args: argparse.Namespace) -> dict:
    keys = (
        "seed", "learning_rate", "batch_size", "buffer_size", "learning_starts",
        "target_update_interval", "exploration_fraction", "exploration_final_eps",
        "gradient_steps", "net_arch", "eval_every", "validation_count",
    )
    result = {"algorithm": "dqn", "environment": ENV_ID, **{k: getattr(args, k) for k in keys}}
    if args.reward_mode != "original":
        result["reward_mode"] = args.reward_mode
    return result


def save_model(model: DQN, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)


def save_replay(model: DQN, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.pkl")
    model.save_replay_buffer(temporary)
    os.replace(temporary, path)


class ValidateAndSave(BaseCallback):
    def __init__(self, run_dir: Path, seeds: list[int], every: int, seconds_before: float, started: float):
        super().__init__()
        self.run_dir = run_dir
        self.seeds = seeds
        self.every = every
        self.seconds_before = seconds_before
        self.started = started
        self.next_eval = every

    def _on_training_start(self) -> None:
        self.next_eval = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_eval:
            self.record("periodic")
            self.next_eval += self.every
        return True

    def record(self, label: str) -> None:
        step = self.model.num_timesteps
        result = evaluate_model(self.model, self.seeds)
        result.update({"step": step, "checkpoint_kind": label, "split": "validation"})
        path = self.run_dir / "validation" / f"step_{step:09d}_{label}.json"
        write_json(path, result)
        with (self.run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: v for k, v in result.items() if k != "episodes"}) + "\n")

        save_model(self.model, self.run_dir / "checkpoints" / f"step_{step:09d}.zip")
        for criterion, new_key in (
            ("reward", (result["mean_reward"], result["two_leg_landing_rate"])),
            ("landing", (result["two_leg_landing_rate"], result["mean_reward"])),
            ("safe", (result["safe_landing_rate"], result["mean_reward"])),
            ("quality", (result["safe_centered_gentle_rate"], result["safe_centered_rate"],
                         result["mean_reward"])),
        ):
            info_path = self.run_dir / f"best_{criterion}.json"
            previous = json.loads(info_path.read_text()) if info_path.exists() else None
            old_key = tuple(previous["selection_key"]) if previous else None
            if old_key is None or new_key > old_key:
                save_model(self.model, self.run_dir / f"best_{criterion}.zip")
                write_json(info_path, {"step": step, "selection_key": new_key, "validation": result})

        save_model(self.model, self.run_dir / "resume_latest.zip")
        save_replay(self.model, self.run_dir / "resume_replay.pkl")
        write_json(self.run_dir / "resume_state.json", {
            "actual_timesteps": step,
            "training_seconds_cumulative": self.seconds_before + time.perf_counter() - self.started,
            "note": "Resume restarts the environment; it is not a bit-for-bit continuation.",
        })
        print(f"Validation at {step:,} steps: mean={result['mean_reward']:.1f}, "
              f"two-leg={result['two_leg_landing_rate']:.1%}, "
              f"truncated={result['truncation_rate']:.1%}")


def main() -> None:
    args = arguments()
    run_dir = args.run_dir
    config_path = run_dir / "config.json"
    config = settings(args)
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError("No existing run config; omit --resume for a new run")
        old = json.loads(config_path.read_text())
        if old["settings"] != config:
            raise ValueError("Resume settings differ from the saved run; use the original settings")
        state = json.loads((run_dir / "resume_state.json").read_text())
        before = float(state["training_seconds_cumulative"])
        segment_index = len(list((run_dir / "segments").glob("*.monitor.csv"))) + 1
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, {
            "settings": config,
            "versions": {
                "python": platform.python_version(),
                "gymnasium": gym.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
            },
            "device": "cpu",
            "seed_manifest": str(Path(__file__).with_name("seeds.json")),
        })
        before = 0.0
        segment_index = 1

    set_random_seed(args.seed)
    monitor_path = run_dir / "segments" / f"segment_{segment_index:03d}.monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(ENV_ID)
    if args.reward_mode == "quality_v1":
        base_env = QualityRewardWrapper(base_env)
    env = Monitor(base_env, filename=str(monitor_path))
    try:
        if args.resume:
            model = DQN.load(run_dir / "resume_latest.zip", env=env, device="cpu")
            replay_path = run_dir / "resume_replay.pkl"
            if not replay_path.exists():
                raise FileNotFoundError("Cannot resume DQN without its replay buffer")
            model.load_replay_buffer(replay_path)
        else:
            model = DQN(
                "MlpPolicy", env, verbose=1, seed=args.seed, device="cpu",
                learning_rate=args.learning_rate, batch_size=args.batch_size,
                buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                target_update_interval=args.target_update_interval,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=args.exploration_final_eps,
                gradient_steps=args.gradient_steps,
                policy_kwargs={"net_arch": args.net_arch},
            )
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            raise ValueError(f"Target {args.timesteps} must exceed saved step {model.num_timesteps}")
        started = time.perf_counter()
        callback = ValidateAndSave(run_dir, seeds_for("validation", args.validation_count),
                                   args.eval_every, before, started)
        model.learn(total_timesteps=remaining, callback=callback,
                    reset_num_timesteps=False, progress_bar=True)
        # The final DQN gradient update may happen after the last on_step callback.
        callback.record("final")
        save_model(model, run_dir / "final_model.zip")
        episodes = sum(len(pd.read_csv(p, comment="#")) for p in (run_dir / "segments").glob("*.monitor.csv"))
        write_json(run_dir / "training_summary.json", {
            "algorithm": "dqn", "environment": ENV_ID, "seed": args.seed,
            "requested_target_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "episodes_completed": episodes,
            "training_seconds_cumulative": before + time.perf_counter() - started,
            "validation_episodes_per_check": args.validation_count,
            "best_reward_step": json.loads((run_dir / "best_reward.json").read_text())["step"],
            "best_landing_step": json.loads((run_dir / "best_landing.json").read_text())["step"],
            "best_safe_step": json.loads((run_dir / "best_safe.json").read_text())["step"],
            "best_quality_step": json.loads((run_dir / "best_quality.json").read_text())["step"],
        })
        print(f"Run saved to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
