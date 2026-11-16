"""Evaluate five frozen selected policies on 50 fresh, shared LunarLander starts."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from stable_baselines3 import DQN, PPO

from metrics import evaluate_model, write_json


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "comparisons" / "phase9_five_models_50.json"
NAMES = (
    "slow_ppo_v3_seed123",
    "slow_ppo_v3_seed7",
    "slow_ppo_v3_seed42",
    "centered_ppo_v2_seed123",
    "softer_dqn_seed123",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_seeds() -> set[int]:
    used: set[int] = set()
    for path in RESEARCH.glob("*seeds.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for values in data.values():
            if isinstance(values, list):
                used.update(value for value in values if isinstance(value, int))
    return used


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite completed comparison: {OUTPUT}")
    manifest_path = RESEARCH / "selected" / "models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(NAMES) == set(manifest)
    rng = random.Random(20261104)
    used = prior_seeds()
    seeds = []
    while len(seeds) < 50:
        seed = rng.randrange(10_000_000, 2_000_000_000)
        if seed not in used:
            seeds.append(seed)
            used.add(seed)

    # Freeze paths and seed list on disk before the first policy evaluation.
    plan_path = RESEARCH / "phase9_five_models_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"Comparison plan already exists: {plan_path}")
    models = []
    for name in NAMES:
        record = manifest[name]
        path = ROOT / record["copy"]
        actual_hash = sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")
        models.append({"name": name, "path": record["copy"], "sha256": actual_hash,
                       "algorithm": "DQN" if name.startswith("softer_dqn") else "PPO"})
    write_json(plan_path, {"purpose": "Five selected models, same 50 fresh starts",
                           "selection_rule": "All five previously frozen selected checkpoints",
                           "seeds": seeds, "models": models})

    results = []
    for item in models:
        loader = DQN if item["algorithm"] == "DQN" else PPO
        model = loader.load(ROOT / item["path"], device="cpu")
        result = evaluate_model(model, seeds)
        result.update(item)
        results.append(result)
        print(f"{item['name']}: mean={result['mean_reward']:.2f}, "
              f"safe+center={result['safe_centered_rate']:.0%}, "
              f"gentle={result['safe_centered_gentle_rate']:.0%}, "
              f"median_duration={result['median_safe_centered_duration_seconds']:.2f}s",
              flush=True)
    write_json(OUTPUT, {"plan_sha256": sha256(plan_path), "seeds": seeds,
                        "results": results})
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
