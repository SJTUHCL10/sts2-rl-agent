"""Record complete entity-v2 test runs as JSON and offline HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from sts2_env.agent_v2.tensorizer import (
    LegacyV5TensorizerConfig,
    TensorizerConfig,
)
from sts2_env.agent_v2.trajectory_trace import (
    TRACE_FORMAT,
    compact_state,
    describe_action,
    record_step,
    write_trace,
)
from sts2_env.gym_env.entity_run_env import (
    ENTITY_ACTION_SEMANTICS_VERSION,
    STS2EntityRunEnv,
)


def _metadata_path(model_path: Path) -> Path:
    for directory in (model_path.parent, model_path.parent.parent):
        candidate = directory / "model_metadata.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model_metadata.json beside {model_path}")


def load_traced_model(
    model_path: Path, device: str,
) -> tuple[MaskablePPO, TensorizerConfig]:
    """Choose exact current or legacy-v5 tensor semantics from checkpoint hash."""
    metadata = json.loads(_metadata_path(model_path).read_text(encoding="utf-8"))
    saved_config = metadata["tensorizer"]
    expected_hash = metadata["tensorizer_layout_hash"]
    candidates = (
        TensorizerConfig(**saved_config),
        LegacyV5TensorizerConfig(**saved_config),
    )
    config = next(
        (item for item in candidates if item.feature_layout_hash() == expected_hash),
        None,
    )
    if config is None:
        raise ValueError("Unsupported checkpoint tensorizer layout; refusing to guess")
    if metadata.get("action_semantics_version") != ENTITY_ACTION_SEMANTICS_VERSION:
        raise ValueError("Checkpoint action semantics do not match this environment")
    model = MaskablePPO.load(model_path, device=device)
    policy = model.policy
    if getattr(policy, "tensorizer_layout_hash", None) != expected_hash:
        raise ValueError("Checkpoint policy and metadata tensorizer hashes disagree")
    if getattr(policy, "categorical_vocabulary_hash", None) != config.categorical_vocabulary_hash:
        raise ValueError("Checkpoint categorical vocabulary does not match runtime")
    return model, config


def trace_episode(
    *, model: MaskablePPO | None, config: TensorizerConfig, seed: int,
    deterministic: bool, max_steps: int, max_combat_turns: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run one full episode while recording every decision and state delta."""
    env = STS2EntityRunEnv(
        tensorizer_config=config,
        max_steps=max_steps,
        max_combat_turns=max_combat_turns,
    )
    try:
        obs, info = env.reset(seed=seed)
        steps: list[dict[str, Any]] = []
        done = False
        while not done:
            snapshot = env.run_env.entity_observation()
            manager = env.run_env._mgr
            assert manager is not None
            before = compact_state(snapshot, manager)
            mask = env.action_masks().copy()
            if model is None:
                if rng is None:
                    rng = np.random.default_rng(seed)
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                prediction, _ = model.predict(
                    obs, action_masks=mask, deterministic=deterministic,
                )
                action = int(prediction)
            described = describe_action(
                snapshot, mask, action,
                legacy_v5=isinstance(config, LegacyV5TensorizerConfig),
            )
            if described["action_type"] == "UNRESOLVED":
                available = manager.get_available_actions()
                described["available_action_types"] = [
                    item.get("action") for item in available
                ]
                if before["phase"] == "CARD_REWARD" and action in {
                    env.run_env._layout.card_reward_start,
                    env.run_env._layout.card_reward_start + 1,
                }:
                    reward_actions = {
                        str(item.get("action")): item for item in available
                    }
                    for pick, skip in (
                        ("pick_potion", "skip_potion"),
                        ("pick_relic_reward", "skip_relic"),
                    ):
                        if pick in reward_actions:
                            selected = pick if action == env.run_env._layout.card_reward_start else skip
                            described["action_type"] = selected.upper()
                            described["payload"] = {"action": selected}
                            break
            obs, reward, terminated, truncated, info = env.step(action)
            steps.append(record_step(
                index=len(steps) + 1,
                before=before,
                action=described,
                after_snapshot=env.run_env.entity_observation(),
                after_manager=manager,
                reward=float(reward),
                info=info,
                terminated=terminated,
                truncated=truncated,
            ))
            done = terminated or truncated
        return {
            "format": TRACE_FORMAT,
            "seed": seed,
            "tensorizer_version": (
                "typed-set-tensor-v5"
                if isinstance(config, LegacyV5TensorizerConfig)
                else "typed-set-tensor-v6"
            ),
            "deterministic": deterministic,
            "summary": {
                "floor": int(info["floor"]),
                "act": int(info["act"]) + 1,
                "steps": len(steps),
                "missing_candidate_steps": sum(
                    bool(step["action"].get("candidate_missing")) for step in steps
                ),
                "legacy_model_missing_candidate_steps": sum(
                    bool(step["action"].get("legacy_model_candidate_missing"))
                    for step in steps
                ),
                "won": bool(info["player_won"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "combat_turn_limit_reached": bool(info["combat_turn_limit_reached"]),
            },
            "steps": steps,
        }
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model-path", type=Path)
    policy.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=100_109)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/trajectories"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-combat-turns", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    model: MaskablePPO | None = None
    config: TensorizerConfig = TensorizerConfig()
    if args.model_path is not None:
        model, config = load_traced_model(args.model_path, args.device)
        if args.stochastic:
            model.set_random_seed(args.seed)
    for episode in range(args.episodes):
        seed = args.seed + episode
        trace = trace_episode(
            model=model,
            config=config,
            seed=seed,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
            max_combat_turns=args.max_combat_turns,
        )
        trace["model_path"] = str(args.model_path.resolve()) if args.model_path else None
        json_path, html_path = write_trace(
            trace, args.output_dir / f"seed_{seed}",
        )
        summary = trace["summary"]
        print(
            f"seed={seed} floor={summary['floor']} won={summary['won']} "
            f"steps={summary['steps']}\n  {json_path}\n  {html_path}"
        )


if __name__ == "__main__":
    main()
