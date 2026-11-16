"""Isolated, reproducible PPO comparison using the RL Zoo LunarLander preset."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import ConstantSchedule

from metrics import ENV_ID, evaluate_model, seeds_for, write_json
from quality_reward import QualityRewardWrapper, QualityRewardWrapperV2, SlowDescentRewardWrapper
from train import save_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True, help="Absolute target")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-every", type=int, default=65_536)
    parser.add_argument("--validation-count", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-model", type=Path,
                        help="Start a new run from a saved PPO checkpoint")
    parser.add_argument("--reward-mode", choices=["original", "quality_v1", "quality_v2",
                                                  "quality_v3_slow"],
                        default="original")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_every <= 0:
        parser.error("Timesteps and evaluation interval must be positive")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    seeds_for("validation", args.validation_count)
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.init_model is not None:
        args.init_model = args.init_model.expanduser().resolve()
        if not args.init_model.is_file():
            parser.error(f"Initial model not found: {args.init_model}")
    return args


def settings(args: argparse.Namespace) -> dict:
    return {
        "algorithm": "ppo", "environment": ENV_ID, "seed": args.seed,
        "n_envs": 16, "n_steps": 1024, "batch_size": 64,
        "gae_lambda": 0.98, "gamma": 0.999, "n_epochs": 4,
        "ent_coef": 0.01, "learning_rate": args.learning_rate, "clip_range": 0.2,
        "eval_every": args.eval_every, "validation_count": args.validation_count,
        "source": "RL Baselines3 Zoo hyperparams/ppo.yml LunarLander-v3",
    }


def extra_settings(args: argparse.Namespace, base: dict) -> dict:
    result = dict(base)
    if args.init_model is not None:
        result["init_model"] = str(args.init_model)
    if args.reward_mode != "original":
        result["reward_mode"] = args.reward_mode
    return result


class ValidatePPO(BaseCallback):
    def __init__(self, run_dir: Path, seeds: list[int], every: int):
        super().__init__()
        self.run_dir = run_dir
        self.seeds = seeds
        self.every = every
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
        write_json(self.run_dir / "validation" / f"step_{step:09d}_{label}.json", result)
        with (self.run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: v for k, v in result.items() if k != "episodes"}) + "\n")
        save_model(self.model, self.run_dir / "checkpoints" / f"step_{step:09d}_{label}.zip")
        for name, key in (
            ("reward", (result["mean_reward"], result["safe_centered_rate"])),
            ("safe", (result["safe_landing_rate"], result["safe_centered_rate"],
                      result["mean_reward"])),
            ("centered", (result["safe_centered_rate"], result["safe_landing_rate"],
                          result["mean_reward"])),
            ("quality", (result["safe_centered_gentle_rate"], result["safe_centered_rate"],
                         result["mean_reward"])),
            ("slow", (result["safe_centered_rate"], result["safe_landing_rate"],
                      -(result["median_safe_centered_abs_touchdown_vy"]
                        if result["median_safe_centered_abs_touchdown_vy"] is not None else 1e9),
                      result["median_safe_centered_duration_seconds"] or 0.0,
                      result["mean_reward"])),
        ):
            info = self.run_dir / f"best_{name}.json"
            old = json.loads(info.read_text()) if info.exists() else None
            if old is None or key > tuple(old["selection_key"]):
                save_model(self.model, self.run_dir / f"best_{name}.zip")
                write_json(info, {"step": step, "checkpoint_kind": label,
                                  "selection_key": key, "validation": result})
        save_model(self.model, self.run_dir / "resume_latest.zip")
        print(f"PPO validation at {step:,}: reward={result['mean_reward']:.1f}, "
              f"safe={result['safe_landing_rate']:.0%}, "
              f"centered={result['safe_centered_rate']:.0%}, "
              f"gentle={result['safe_centered_gentle_rate']:.0%}", flush=True)


def main() -> None:
    args = arguments()
    run_dir = args.run_dir
    config = extra_settings(args, settings(args))
    if args.resume:
        saved = json.loads((run_dir / "config.json").read_text())
        if saved["settings"] != config:
            raise ValueError("Resume settings differ from saved configuration")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", {
            "settings": config,
            "versions": {"python": platform.python_version(),
                         "gymnasium": gym.__version__,
                         "stable_baselines3": stable_baselines3.__version__,
                         "torch": torch.__version__},
            "device": "cpu", "evaluation_environment": "unwrapped LunarLander-v3",
        })
    set_random_seed(args.seed)
    torch.set_num_threads(1)
    segment = len(list((run_dir / "segments").glob("segment_*"))) + 1
    monitor_dir = run_dir / "segments" / f"segment_{segment:03d}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    wrapper = {"original": None, "quality_v1": QualityRewardWrapper,
               "quality_v2": QualityRewardWrapperV2,
               "quality_v3_slow": SlowDescentRewardWrapper}[args.reward_mode]
    env = make_vec_env(ENV_ID, n_envs=16, seed=args.seed, monitor_dir=str(monitor_dir),
                       wrapper_class=wrapper)
    try:
        if args.resume:
            model = PPO.load(run_dir / "resume_latest.zip", env=env, device="cpu")
        elif args.init_model is not None:
            model = PPO.load(args.init_model, env=env, device="cpu")
            model.set_random_seed(args.seed)
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                        n_steps=1024, batch_size=64, gae_lambda=0.98,
                        gamma=0.999, n_epochs=4, ent_coef=0.01,
                        learning_rate=args.learning_rate, clip_range=0.2)
        model.learning_rate = args.learning_rate
        model.lr_schedule = ConstantSchedule(args.learning_rate)
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            raise ValueError("Target must exceed the saved step")
        started = time.perf_counter()
        callback = ValidatePPO(run_dir, seeds_for("validation", args.validation_count),
                               args.eval_every)
        model.learn(total_timesteps=remaining, callback=callback,
                    reset_num_timesteps=False, progress_bar=True)
        callback.record("final")
        save_model(model, run_dir / "final_model.zip")
        episodes = sum(len(pd.read_csv(p, comment="#")) for p in monitor_dir.glob("*.monitor.csv"))
        previous = json.loads((run_dir / "training_summary.json").read_text()) if args.resume else None
        write_json(run_dir / "training_summary.json", {
            "algorithm": "ppo", "environment": ENV_ID, "seed": args.seed,
            "requested_target_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "episodes_completed": episodes + (previous["episodes_completed"] if previous else 0),
            "training_seconds_cumulative": time.perf_counter() - started +
            (previous["training_seconds_cumulative"] if previous else 0),
        })
        print(f"Saved PPO run to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
