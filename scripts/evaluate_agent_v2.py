"""Evaluate a saved entity-v2 checkpoint and its random baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from sts2_env.agent_v2.tensorizer import TensorizerConfig
from train_agent_v2 import evaluate, evaluate_random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_109)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Replace evaluation fields in adjacent model_metadata.json.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    metadata_path = model_path.parent / "model_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    tensorizer_config = TensorizerConfig(**metadata.get("tensorizer", {}))
    model = MaskablePPO.load(model_path, device="cpu")
    evaluation = evaluate(
        model,
        episodes=args.episodes,
        seed=args.seed,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
    )
    random_baseline = evaluate_random(
        episodes=args.episodes,
        seed=args.seed,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
    )
    result = {
        "evaluation": evaluation,
        "random_baseline": random_baseline,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.update_metadata:
        metadata.update(result)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
