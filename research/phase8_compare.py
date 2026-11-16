"""One-time held-out 200-episode replication of slower PPO fine-tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stable_baselines3 import PPO

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "phase8_selection.json"
SEEDS_FILE = RESEARCH / "phase8_seeds.json"
OUTPUT = RESEARCH / "comparisons" / "phase8_test8.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Held-out test already saved: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("frozen_before_test8"):
        raise ValueError("Freeze all model paths before testing")
    seeds = json.loads(SEEDS_FILE.read_text())["test8"]
    if len(seeds) != 200 or len(set(seeds)) != 200:
        raise ValueError("Expected 200 distinct test seeds")
    results = []
    for item in manifest["models"]:
        path = ROOT / item["path"]
        model = PPO.load(path, device="cpu")
        result = evaluate_model(model, seeds)
        result.update({"name": item["name"], "model_path": str(path),
                       "model_sha256": digest(path)})
        results.append(result)
        print(f"{item['name']}: mean={result['mean_reward']:.2f}, "
              f"safe+center={result['safe_centered_rate']:.1%}, "
              f"gentle={result['safe_centered_gentle_rate']:.1%}, "
              f"median_flight={result['median_safe_centered_duration_seconds']:.2f}s, "
              f"median_abs_vy={result['median_safe_centered_abs_touchdown_vy']:.4f}",
              flush=True)
    write_json(OUTPUT, {"split": "test8", "evaluation_seeds": seeds,
                        "manifest_sha256": digest(MANIFEST),
                        "seeds_sha256": digest(SEEDS_FILE),
                        "scope_note": manifest["scope_note"],
                        "test_acceptance": manifest["test_acceptance"],
                        "results": results})
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
