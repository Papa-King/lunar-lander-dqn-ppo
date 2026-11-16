"""One-time held-out replication of PPO quality fine-tuning seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stable_baselines3 import PPO

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "phase6_selection.json"
SEEDS_FILE = RESEARCH / "phase6_seeds.json"
OUTPUT = RESEARCH / "comparisons" / "phase6_test6.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Held-out test already saved: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("frozen_before_test6"):
        raise ValueError("Selection was not frozen")
    seeds = json.loads(SEEDS_FILE.read_text())["test6"]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("Expected 100 distinct seeds")
    results = []
    for item in manifest["models"]:
        path = ROOT / item["path"]
        model = PPO.load(path, device="cpu")
        result = evaluate_model(model, seeds)
        result.update({"name": item["name"], "model_path": str(path),
                       "model_sha256": digest(path)})
        results.append(result)
        print(f"{item['name']}: mean={result['mean_reward']:.2f}, "
              f"safe={result['safe_landing_rate']:.1%}, "
              f"centered={result['safe_centered_rate']:.1%}, "
              f"gentle={result['safe_centered_gentle_rate']:.1%}", flush=True)
    write_json(OUTPUT, {"split": "test6", "evaluation_seeds": seeds,
                        "manifest_sha256": digest(MANIFEST),
                        "seeds_sha256": digest(SEEDS_FILE),
                        "scope_note": manifest["scope_note"],
                        "results": results})
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
