"""Evaluate one saved DQN checkpoint on an explicitly chosen seed split."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from metrics import evaluate_model, seeds_for, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["best_reward", "best_landing", "best_safe",
                                            "best_quality", "final_model"],
                        default="best_reward")
    parser.add_argument("--split", choices=["validation", "test", "legacy_practice"],
                        default="validation")
    parser.add_argument("--count", type=int, help="Use the first N seeds (for short pipeline checks)")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_path = run_dir / f"{args.model}.zip"
    if not model_path.exists():
        parser.error(f"Model not found: {model_path}")
    model = DQN.load(model_path, device="cpu")
    result = evaluate_model(model, seeds_for(args.split, args.count))
    result.update({"model": args.model, "split": args.split, "model_path": str(model_path)})
    output = run_dir / f"{args.model}_{args.split}_evaluation.json"
    write_json(output, result)
    print(f"Saved {output}")
    for key in ("mean_reward", "reward_std", "safe_landing_rate", "safe_centered_rate",
                "safe_centered_gentle_rate", "two_leg_landing_rate", "score_200_rate",
                "mean_action_switch_rate", "truncation_rate"):
        print(f"{key}: {result[key]:.4f}")


if __name__ == "__main__":
    main()
