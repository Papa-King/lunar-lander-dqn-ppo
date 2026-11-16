"""Compare frozen classroom and tuned checkpoints on one explicit seed split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from metrics import SEED_FILE, evaluate_model, seeds_for, write_json


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("final_selection.json")
METRICS = (
    "mean_reward", "reward_std", "two_leg_landing_rate", "score_200_rate",
    "mean_action_switch_rate", "truncation_rate",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--count", type=int, help="First N episodes; never use this for final test")
    args = parser.parse_args()
    if args.split == "test" and args.count is not None:
        parser.error("The final test must use all held-out seeds")
    output = ROOT / "research" / "comparisons" / f"{args.split}.json"
    if args.split == "test" and output.exists():
        parser.error(f"Final test already exists; refusing to overwrite {output}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("frozen_before_test"):
        parser.error("Selection manifest must be frozen before test")
    seeds = seeds_for(args.split, args.count)
    results = []
    for entry in manifest["models"]:
        model_path = ROOT / entry["path"]
        if not model_path.is_file():
            parser.error(f"Missing checkpoint: {model_path}")
        model = DQN.load(model_path, device="cpu")
        result = evaluate_model(model, seeds)
        result.update({
            "name": entry["name"], "family": entry["family"],
            "training_seed": entry["training_seed"],
            "model_path": str(model_path), "model_sha256": digest(model_path),
        })
        results.append(result)
        print(f"{entry['name']}: reward={result['mean_reward']:.2f}, "
              f"std={result['reward_std']:.2f}, "
              f"two-leg={result['two_leg_landing_rate']:.1%}, "
              f"200+={result['score_200_rate']:.1%}", flush=True)

    summary = {}
    for family in ("classroom_dqn", "zoo_dqn_preset"):
        members = [r for r in results if r["family"] == family]
        summary[family] = {
            "training_seeds": [r["training_seed"] for r in members],
            "mean_across_training_seeds": {
                key: float(np.mean([r[key] for r in members])) for key in METRICS
            },
            "between_training_seed_std_of_mean_reward": float(np.std(
                [r["mean_reward"] for r in members], ddof=0
            )),
        }
    write_json(output, {
        "split": args.split, "evaluation_seeds": seeds,
        "manifest_sha256": digest(MANIFEST), "seeds_sha256": digest(SEED_FILE),
        "primary_model": manifest["primary_model"],
        "selection_rule": manifest["selection_rule"],
        "results": results, "summary": summary,
    })
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
