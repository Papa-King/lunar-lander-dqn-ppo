"""Frozen, one-time comparison of two DQN learning rates on unseen starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stable_baselines3 import DQN

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "phase2_selection.json"
SEEDS_FILE = RESEARCH / "phase2_seeds.json"
OUTPUT = RESEARCH / "comparisons" / "phase2_test2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Final comparison already saved: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("frozen_before_test2"):
        raise ValueError("Freeze model selection before testing")
    seeds = json.loads(SEEDS_FILE.read_text())["test2"]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("Expected 100 distinct final test seeds")
    results = []
    for entry in manifest["models"]:
        path = ROOT / entry["path"]
        model = DQN.load(path, device="cpu")
        item = evaluate_model(model, seeds)
        item.update({"name": entry["name"], "family": entry["family"],
                     "training_seed": entry["training_seed"],
                     "model_path": str(path), "model_sha256": digest(path)})
        results.append(item)
        print(f"{entry['name']}: mean={item['mean_reward']:.2f}, "
              f"safe={item['safe_landing_rate']:.1%}, "
              f"two-leg={item['two_leg_landing_rate']:.1%}, "
              f"200+={item['score_200_rate']:.1%}", flush=True)
    write_json(OUTPUT, {
        "split": "test2", "evaluation_seeds": seeds,
        "manifest_sha256": digest(MANIFEST), "seeds_sha256": digest(SEEDS_FILE),
        "selection_rule": manifest["selection_rule"],
        "primary_new_model": manifest["primary_new_model"],
        "results": results,
    })
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
