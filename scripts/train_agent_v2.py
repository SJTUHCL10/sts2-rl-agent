"""Train and evaluate the entity-v2 Typed Set Transformer agent.

The environment, encoder, policy head, and RL algorithm are selected in
separate factory functions so later architecture/algorithm experiments do not
need to fork the training loop.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sts2_env.agent_v2.schema import schema_manifest
from sts2_env.agent_v2.tensorizer import TensorizerConfig
from sts2_env.gym_env.reward_shaping import RunRewardShapingConfig
from sts2_env.gym_env.entity_run_env import ENTITY_ACTION_SEMANTICS_VERSION
from sts2_env.models.typed_set_transformer import (
    TypedSetMaskableActorCriticPolicy,
    TypedSetTransformerConfig,
)


def linear_schedule(
    initial_value: float,
    final_value: float,
) -> Callable[[float], float]:
    """Return an SB3 progress-remaining linear learning-rate schedule."""
    if initial_value <= 0 or final_value < 0:
        raise ValueError("Learning rates must be positive/non-negative")
    if final_value > initial_value:
        raise ValueError("final learning rate cannot exceed initial rate")

    def schedule(progress_remaining: float) -> float:
        return final_value + (
            initial_value - final_value
        ) * float(progress_remaining)

    return schedule


def make_env_factory(
    seed: int,
    *,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
    max_combat_turns: int,
    reward_shaping: RunRewardShapingConfig | None = None,
) -> Callable[[], Any]:
    """Return a pickle-safe masked full-run environment factory."""

    def init() -> Any:
        from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

        env = STS2EntityRunEnv(
            tensorizer_config=tensorizer_config,
            max_steps=max_steps,
            max_combat_turns=max_combat_turns,
            reward_shaping=reward_shaping,
        )
        env.reset(seed=seed)
        return env

    return init


def build_model(
    algorithm: str,
    architecture: str,
    env: Any,
    *,
    model_config: TypedSetTransformerConfig,
    tensorizer_config: TensorizerConfig,
    args: argparse.Namespace,
) -> Any:
    """Algorithm/architecture registry boundary."""
    if algorithm != "maskable_ppo":
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if architecture != "typed_set_transformer":
        raise ValueError(f"Unsupported architecture: {architecture}")

    from sb3_contrib import MaskablePPO

    learning_rate: float | Callable[[float], float] = args.lr
    if args.lr_schedule == "linear":
        learning_rate = linear_schedule(args.lr, args.final_lr)
    return MaskablePPO(
        TypedSetMaskableActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs={
            "typed_set_config": model_config,
            "categorical_vocab_size": tensorizer_config.categorical_vocab_size,
            "categorical_vocabulary_hash": (
                tensorizer_config.categorical_vocabulary_hash
            ),
            "tensorizer_layout_hash": tensorizer_config.feature_layout_hash(),
            "action_semantics_version": ENTITY_ACTION_SEMANTICS_VERSION,
            "ortho_init": False,
        },
    )


def evaluate(
    model: Any,
    *,
    episodes: int,
    seed: int,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
    max_combat_turns: int = 50,
    reward_shaping: RunRewardShapingConfig | None = None,
    deterministic: bool = True,
    n_envs: int = 1,
) -> dict[str, Any]:
    """Evaluate complete runs with the same structured observation contract."""
    if n_envs > 1:
        return _evaluate_parallel(
            model,
            episodes=episodes,
            seed=seed,
            tensorizer_config=tensorizer_config,
            max_steps=max_steps,
            max_combat_turns=max_combat_turns,
            reward_shaping=reward_shaping,
            deterministic=deterministic,
            n_envs=n_envs,
        )
    from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

    env = STS2EntityRunEnv(
        tensorizer_config=tensorizer_config,
        max_steps=max_steps,
        max_combat_turns=max_combat_turns,
        reward_shaping=reward_shaping,
    )
    rewards: list[float] = []
    base_rewards: list[float] = []
    floors: list[int] = []
    lengths: list[int] = []
    wins = 0
    terminated_episodes = 0
    truncated_episodes = 0
    episode_results: list[dict[str, Any]] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_base_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            episode_base_reward += float(info.get("base_reward", reward))
            steps += 1
            done = terminated or truncated
        rewards.append(episode_reward)
        base_rewards.append(episode_base_reward)
        floors.append(int(info.get("floor", 0)))
        lengths.append(steps)
        if bool(terminated) and episode_base_reward > 0:
            wins += 1
        terminated_episodes += int(bool(terminated))
        truncated_episodes += int(bool(truncated))
        final_snapshot = env.run_env.entity_observation()
        episode_results.append({
            "seed": seed + episode,
            "floor": floors[-1],
            "length": steps,
            "won": bool(terminated) and episode_base_reward > 0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_phase": info.get("phase"),
            "final_state_type": final_snapshot.get("type"),
            "selected_count": final_snapshot.get("selected_count"),
            "can_confirm": final_snapshot.get("can_confirm"),
            "combat_turn_limit_reached": bool(
                info.get("combat_turn_limit_reached", False)
            ),
        })
    env.close()
    return {
        "episodes": episodes,
        "wins": wins,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "win_rate": wins / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)),
        "mean_base_reward": float(np.mean(base_rewards)),
        "mean_floor": float(np.mean(floors)),
        "median_floor": float(np.median(floors)),
        "floor_p25": float(np.percentile(floors, 25)),
        "floor_p75": float(np.percentile(floors, 75)),
        "max_floor": max(floors, default=0),
        "mean_episode_length": float(np.mean(lengths)),
        "deterministic": deterministic,
        "seed": seed,
        "episode_results": episode_results,
    }


def evaluate_random(
    *,
    episodes: int,
    seed: int,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
    max_combat_turns: int = 50,
    reward_shaping: RunRewardShapingConfig | None = None,
    n_envs: int = 1,
) -> dict[str, Any]:
    """Run a mask-respecting random baseline on identical episode seeds."""
    if n_envs > 1:
        return _evaluate_parallel(
            None,
            episodes=episodes,
            seed=seed,
            tensorizer_config=tensorizer_config,
            max_steps=max_steps,
            max_combat_turns=max_combat_turns,
            reward_shaping=reward_shaping,
            deterministic=False,
            n_envs=n_envs,
        )
    from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

    env = STS2EntityRunEnv(
        tensorizer_config=tensorizer_config,
        max_steps=max_steps,
        max_combat_turns=max_combat_turns,
        reward_shaping=reward_shaping,
    )
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    base_rewards: list[float] = []
    floors: list[int] = []
    lengths: list[int] = []
    wins = 0
    terminated_episodes = 0
    truncated_episodes = 0
    episode_results: list[dict[str, Any]] = []
    for episode in range(episodes):
        _, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_base_reward = 0.0
        steps = 0
        while not done:
            valid = np.flatnonzero(env.action_masks())
            action = int(rng.choice(valid))
            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_base_reward += float(info.get("base_reward", reward))
            steps += 1
            done = terminated or truncated
        rewards.append(episode_reward)
        base_rewards.append(episode_base_reward)
        floors.append(int(info.get("floor", 0)))
        lengths.append(steps)
        if bool(terminated) and episode_base_reward > 0:
            wins += 1
        terminated_episodes += int(bool(terminated))
        truncated_episodes += int(bool(truncated))
        episode_results.append({
            "seed": seed + episode,
            "floor": floors[-1],
            "length": steps,
            "won": bool(terminated) and episode_base_reward > 0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_phase": info.get("phase"),
            "combat_turn_limit_reached": bool(
                info.get("combat_turn_limit_reached", False)
            ),
        })
    env.close()
    return {
        "episodes": episodes,
        "wins": wins,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "win_rate": wins / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)),
        "mean_base_reward": float(np.mean(base_rewards)),
        "mean_floor": float(np.mean(floors)),
        "median_floor": float(np.median(floors)),
        "floor_p25": float(np.percentile(floors, 25)),
        "floor_p75": float(np.percentile(floors, 75)),
        "max_floor": max(floors, default=0),
        "mean_episode_length": float(np.mean(lengths)),
        "seed": seed,
        "episode_results": episode_results,
    }


def _evaluate_parallel(
    model: Any | None,
    *,
    episodes: int,
    seed: int,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
    max_combat_turns: int,
    reward_shaping: RunRewardShapingConfig | None,
    deterministic: bool,
    n_envs: int,
) -> dict[str, Any]:
    """Evaluate fixed seeds in subprocess environments with batched inference."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    episode_results: list[dict[str, Any]] = []
    rewards: list[float] = []
    base_rewards: list[float] = []
    floors: list[int] = []
    lengths: list[int] = []

    for batch_start in range(0, episodes, n_envs):
        batch_size = min(n_envs, episodes - batch_start)
        batch_seed = seed + batch_start
        vec_env = SubprocVecEnv([
            make_env_factory(
                batch_seed + index,
                tensorizer_config=tensorizer_config,
                max_steps=max_steps,
                max_combat_turns=max_combat_turns,
                reward_shaping=reward_shaping,
            )
            for index in range(batch_size)
        ])
        vec_env.seed(batch_seed)
        observations = vec_env.reset()
        active = np.ones(batch_size, dtype=bool)
        batch_rewards = np.zeros(batch_size, dtype=np.float64)
        batch_base_rewards = np.zeros(batch_size, dtype=np.float64)
        batch_lengths = np.zeros(batch_size, dtype=np.int64)
        random_generators = [
            np.random.default_rng(batch_seed + index)
            for index in range(batch_size)
        ]

        while np.any(active):
            masks = np.asarray(vec_env.env_method("action_masks"))
            if model is None:
                actions = np.asarray([
                    random_generators[index].choice(
                        np.flatnonzero(masks[index])
                    )
                    for index in range(batch_size)
                ])
            else:
                actions, _ = model.predict(
                    observations,
                    action_masks=masks,
                    deterministic=deterministic,
                )
            observations, step_rewards, dones, infos = vec_env.step(actions)
            for index in np.flatnonzero(active):
                batch_rewards[index] += float(step_rewards[index])
                batch_base_rewards[index] += float(
                    infos[index].get("base_reward", step_rewards[index])
                )
                batch_lengths[index] += 1
                if not dones[index]:
                    continue
                active[index] = False
                info = infos[index]
                truncated = bool(info.get("TimeLimit.truncated", False))
                won = bool(info.get("player_won", False))
                floor = int(info.get("floor", 0))
                episode_results.append({
                    "seed": batch_seed + int(index),
                    "floor": floor,
                    "length": int(batch_lengths[index]),
                    "won": won,
                    "terminated": not truncated,
                    "truncated": truncated,
                    "final_phase": info.get("phase"),
                    "combat_turn_limit_reached": bool(
                        info.get("combat_turn_limit_reached", False)
                    ),
                })
                rewards.append(float(batch_rewards[index]))
                base_rewards.append(float(batch_base_rewards[index]))
                floors.append(floor)
                lengths.append(int(batch_lengths[index]))
        vec_env.close()

    episode_results.sort(key=lambda item: item["seed"])
    wins = sum(bool(item["won"]) for item in episode_results)
    truncated_episodes = sum(
        bool(item["truncated"]) for item in episode_results
    )
    return {
        "episodes": episodes,
        "wins": wins,
        "terminated_episodes": episodes - truncated_episodes,
        "truncated_episodes": truncated_episodes,
        "win_rate": wins / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)),
        "mean_base_reward": float(np.mean(base_rewards)),
        "mean_floor": float(np.mean(floors)),
        "median_floor": float(np.median(floors)),
        "floor_p25": float(np.percentile(floors, 25)),
        "floor_p75": float(np.percentile(floors, 75)),
        "max_floor": max(floors, default=0),
        "mean_episode_length": float(np.mean(lengths)),
        "deterministic": deterministic if model is not None else None,
        "seed": seed,
        "n_envs": n_envs,
        "episode_results": episode_results,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from sts2_env.training.capability_eval import (
        CapabilityEvalCallback,
        CapabilityScoreConfig,
    )
    from sts2_env.training.metrics import EpisodeMetricsCallback

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorizer_config = TensorizerConfig(max_entities=args.max_entities)
    reward_shaping = (
        RunRewardShapingConfig(
            floor_reward=args.floor_reward,
            combat_win_reward=args.combat_win_reward,
            hp_loss_penalty=args.hp_loss_penalty,
            step_penalty=args.step_penalty,
        )
        if args.reward_shaping
        else None
    )
    model_config = TypedSetTransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        num_memory_tokens=args.num_memory_tokens,
        num_isab_layers=args.num_isab_layers,
        feedforward_multiplier=args.feedforward_multiplier,
        dropout=args.dropout,
        num_policy_experts=args.num_policy_experts,
    )
    model_config.validate()

    factories = [
        make_env_factory(
            args.seed + index,
            tensorizer_config=tensorizer_config,
            max_steps=args.max_steps,
            max_combat_turns=args.max_combat_turns,
            reward_shaping=reward_shaping,
        )
        for index in range(args.n_envs)
    ]
    env = (
        SubprocVecEnv(factories)
        if args.n_envs > 1
        else DummyVecEnv(factories)
    )
    if args.resume_from:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(
            args.resume_from,
            env=env,
            device=args.device,
        )
        policy = model.policy
        expected_layout = getattr(policy, "tensorizer_layout_hash", None)
        if expected_layout != tensorizer_config.feature_layout_hash():
            raise ValueError(
                "Resume checkpoint tensorizer mismatch: expected "
                f"{tensorizer_config.feature_layout_hash()}, got "
                f"{expected_layout}"
            )
        expected_actions = getattr(
            policy, "action_semantics_version", None
        )
        if expected_actions != ENTITY_ACTION_SEMANTICS_VERSION:
            raise ValueError(
                "Resume checkpoint action semantics mismatch: expected "
                f"{ENTITY_ACTION_SEMANTICS_VERSION}, got {expected_actions}"
            )
        resumed_lr: float | Callable[[float], float] = args.lr
        if args.lr_schedule == "linear":
            resumed_lr = linear_schedule(args.lr, args.final_lr)
        model.learning_rate = resumed_lr
        model.lr_schedule = (
            resumed_lr if callable(resumed_lr) else lambda _: resumed_lr
        )
    else:
        model = build_model(
            args.algorithm,
            args.architecture,
            env,
            model_config=model_config,
            tensorizer_config=tensorizer_config,
            args=args,
        )
    metrics_callback = EpisodeMetricsCallback(
        output_dir / "training_curve.jsonl"
    )
    callbacks = [metrics_callback]
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(output_dir / "checkpoints"),
            name_prefix="typed_set_v4",
        ))
    capability_eval_callback = None
    if args.eval_freq > 0:
        capability_eval_callback = CapabilityEvalCallback(
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            evaluation_fn=lambda current_model: evaluate(
                current_model,
                episodes=args.periodic_eval_episodes,
                seed=args.eval_seed,
                tensorizer_config=tensorizer_config,
                max_steps=args.max_steps,
                max_combat_turns=args.max_combat_turns,
                reward_shaping=reward_shaping,
                n_envs=args.eval_envs,
            ),
            output_dir=output_dir,
            score_config=CapabilityScoreConfig(
                win_weight=args.eval_win_weight,
                truncation_penalty=args.eval_truncation_penalty,
            ),
        )
        callbacks.append(capability_eval_callback)
    callback = CallbackList(callbacks) if callbacks else None

    started = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )
    elapsed = time.perf_counter() - started
    effective_timesteps = int(model.num_timesteps)
    model_path = output_dir / "final_model"
    model.save(model_path)
    env.close()

    metrics = evaluate(
        model,
        episodes=args.eval_episodes,
        seed=args.seed + 100_000,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
        max_combat_turns=args.max_combat_turns,
        reward_shaping=reward_shaping,
        n_envs=args.eval_envs,
    )
    random_metrics = evaluate_random(
        episodes=args.eval_episodes,
        seed=args.seed + 100_000,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
        max_combat_turns=args.max_combat_turns,
        reward_shaping=reward_shaping,
        n_envs=args.eval_envs,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.policy.parameters()
    )
    metadata = {
        "algorithm": args.algorithm,
        "architecture": args.architecture,
        "requested_total_timesteps": args.total_timesteps,
        "total_timesteps": effective_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "resume_from": args.resume_from,
        "elapsed_seconds": elapsed,
        "steps_per_second": effective_timesteps / max(elapsed, 1e-9),
        "parameter_count": parameter_count,
        "tensorizer": asdict(tensorizer_config),
        "tensorizer_layout_hash": tensorizer_config.feature_layout_hash(),
        "action_semantics_version": ENTITY_ACTION_SEMANTICS_VERSION,
        "categorical_vocabulary_hash": (
            tensorizer_config.categorical_vocabulary_hash
        ),
        "model": asdict(model_config),
        "interface": schema_manifest(),
        "training": {
            key: getattr(args, key)
            for key in (
                "lr", "final_lr", "lr_schedule", "n_steps", "batch_size",
                "n_epochs", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                "max_grad_norm", "max_steps", "max_combat_turns", "device",
                "checkpoint_freq", "eval_freq", "periodic_eval_episodes",
                "eval_seed", "eval_envs", "eval_win_weight",
                "eval_truncation_penalty",
            )
        },
        "reward_shaping": (
            reward_shaping.to_dict() if reward_shaping is not None else None
        ),
        "training_progress": metrics_callback.summary(),
        "periodic_evaluation": (
            capability_eval_callback.summary()
            if capability_eval_callback is not None
            else None
        ),
        "recommended_model": (
            "best_model/best_model.zip"
            if capability_eval_callback is not None
            and capability_eval_callback.best_metrics is not None
            else "final_model.zip"
        ),
        "evaluation": metrics,
        "random_baseline": random_metrics,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    with (output_dir / "model_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the entity-v2 STS2 agent"
    )
    parser.add_argument(
        "--algorithm",
        choices=("maskable_ppo",),
        default="maskable_ppo",
    )
    parser.add_argument(
        "--architecture",
        choices=("typed_set_transformer",),
        default="typed_set_transformer",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--final-lr", type=float, default=3e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear"),
        default="linear",
    )
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-combat-turns", type=int, default=50)
    parser.add_argument("--max-entities", type=int, default=384)
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--floor-reward", type=float, default=0.02)
    parser.add_argument("--combat-win-reward", type=float, default=0.02)
    parser.add_argument("--hp-loss-penalty", type=float, default=0.20)
    parser.add_argument("--step-penalty", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-inducing-points", type=int, default=16)
    parser.add_argument("--num-memory-tokens", type=int, default=8)
    parser.add_argument("--num-isab-layers", type=int, default=2)
    parser.add_argument("--feedforward-multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-policy-experts", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=8,
        help="Subprocess environments used for fixed-seed evaluation.",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Fixed-seed capability evaluation frequency; 0 disables it.",
    )
    parser.add_argument("--periodic-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100_109)
    parser.add_argument("--eval-win-weight", type=float, default=100.0)
    parser.add_argument(
        "--eval-truncation-penalty",
        type=float,
        default=10.0,
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument(
        "--output-dir",
        default="output/typed_set_v4",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
