"""Evaluate any selected DQN/PPO checkpoint on an explicit saved seed list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN, PPO

from metrics import evaluate_model, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--seed-key", required=True)
    parser.add_argument("--output", type=Path, help="Save full results to a new JSON file")
    args = parser.parse_args()

    seed_data = json.loads(args.seed_file.read_text(encoding="utf-8"))
    seeds = seed_data[args.seed_key]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(x, int) for x in seeds):
        parser.error("The selected seed list must contain integers")
    if len(seeds) != len(set(seeds)):
        parser.error("Evaluation seed list contains duplicates")
    loader = DQN if args.algorithm == "dqn" else PPO
    model = loader.load(args.model, device="cpu")
    result = evaluate_model(model, seeds)
    result.update({"algorithm": args.algorithm, "model_path": str(args.model),
                   "seed_file": str(args.seed_file), "seed_key": args.seed_key})
    if args.output:
        if args.output.exists():
            parser.error(f"Refusing to overwrite {args.output}")
        write_json(args.output, result)
        print(f"Saved {args.output}")
    for key in ("evaluation_episodes", "mean_reward", "reward_std", "safe_landing_rate",
                "safe_centered_rate", "safe_centered_gentle_rate", "score_200_rate"):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
