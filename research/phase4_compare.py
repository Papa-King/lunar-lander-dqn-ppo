"""One-time held-out comparison of frozen DQN and PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stable_baselines3 import DQN, PPO

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "phase4_selection.json"
SEEDS_FILE = RESEARCH / "phase4_seeds.json"
OUTPUT = RESEARCH / "comparisons" / "phase4_test4.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Held-out test already saved: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("frozen_before_test4"):
        raise ValueError("Freeze model selection before test4")
    seeds = json.loads(SEEDS_FILE.read_text())["test4"]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("Expected 100 distinct test seeds")
    results = []
    for item in manifest["comparison_models"]:
        path = ROOT / item["path"]
        model_class = DQN if item["algorithm"] == "dqn" else PPO
        model = model_class.load(path, device="cpu")
        result = evaluate_model(model, seeds)
        result.update({"name": item["name"], "algorithm": item["algorithm"],
                       "model_path": str(path), "model_sha256": digest(path)})
        results.append(result)
        print(f"{item['name']}: reward={result['mean_reward']:.2f}, "
              f"safe={result['safe_landing_rate']:.1%}, "
              f"centered={result['safe_centered_rate']:.1%}, "
              f"gentle={result['safe_centered_gentle_rate']:.1%}", flush=True)
    write_json(OUTPUT, {"split": "test4", "evaluation_seeds": seeds,
                        "manifest_sha256": digest(MANIFEST),
                        "seeds_sha256": digest(SEEDS_FILE),
                        "primary_candidate": manifest["primary_candidate"],
                        "results": results})
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
