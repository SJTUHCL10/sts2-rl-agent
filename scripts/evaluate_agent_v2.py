"""Evaluate a saved entity-v2 checkpoint and its random baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from sts2_env.agent_v2.tensorizer import TensorizerConfig
from sts2_env.agent_v2.categorical_vocabulary import encountered_unknown_tokens
from train_agent_v2 import evaluate, evaluate_random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100_109)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-combat-turns", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--output",
        help="Optional JSON path for the complete evaluation result.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of deterministic argmax.",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip the random baseline when comparing several checkpoints.",
    )
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
    model = MaskablePPO.load(model_path, device=args.device)
    encountered_unknown_tokens(clear=True)
    evaluation = evaluate(
        model,
        episodes=args.episodes,
        seed=args.seed,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
        max_combat_turns=args.max_combat_turns,
        deterministic=not args.stochastic,
        n_envs=args.n_envs,
    )
    result = {
        "evaluation": evaluation,
        "unknown_tokens": encountered_unknown_tokens(),
    }
    if not args.skip_random:
        result["random_baseline"] = evaluate_random(
            episodes=args.episodes,
            seed=args.seed,
            tensorizer_config=tensorizer_config,
            max_steps=args.max_steps,
            max_combat_turns=args.max_combat_turns,
            n_envs=args.n_envs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.update_metadata:
        metadata.update(result)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
