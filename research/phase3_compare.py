"""Evaluate frozen training-budget checkpoints once on fresh held-out starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stable_baselines3 import DQN

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "phase3_selection.json"
SEEDS_FILE = RESEARCH / "phase3_seeds.json"
OUTPUT = RESEARCH / "comparisons" / "phase3_test3.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Held-out test already saved: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("frozen_before_test3"):
        raise ValueError("Selection must be frozen before test3")
    seeds = json.loads(SEEDS_FILE.read_text())["test3"]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("Expected 100 distinct test3 seeds")
    results = []
    for entry in manifest["models"]:
        path = ROOT / entry["path"]
        model = DQN.load(path, device="cpu")
        result = evaluate_model(model, seeds)
        result.update({"name": entry["name"], "step": entry["step"],
                       "model_path": str(path), "model_sha256": digest(path)})
        results.append(result)
        print(f"{entry['name']}: mean={result['mean_reward']:.2f}, "
              f"safe={result['safe_landing_rate']:.1%}, "
              f"centered={result['safe_centered_rate']:.1%}, "
              f"centered+gentle={result['safe_centered_gentle_rate']:.1%}", flush=True)
    write_json(OUTPUT, {"split": "test3", "evaluation_seeds": seeds,
                        "manifest_sha256": digest(MANIFEST),
                        "seeds_sha256": digest(SEEDS_FILE),
                        "primary_model": manifest["primary_model"],
                        "results": results})
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
