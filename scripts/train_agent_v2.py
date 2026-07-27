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
from sts2_env.models.typed_set_transformer import (
    TypedSetMaskableActorCriticPolicy,
    TypedSetTransformerConfig,
)


def make_env_factory(
    seed: int,
    *,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
) -> Callable[[], Any]:
    """Return a pickle-safe masked full-run environment factory."""

    def init() -> Any:
        from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

        env = STS2EntityRunEnv(
            tensorizer_config=tensorizer_config,
            max_steps=max_steps,
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

    return MaskablePPO(
        TypedSetMaskableActorCriticPolicy,
        env,
        learning_rate=args.lr,
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
            "categorical_buckets": tensorizer_config.categorical_buckets,
            "tensorizer_layout_hash": tensorizer_config.feature_layout_hash(),
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
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate complete runs with the same structured observation contract."""
    from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

    env = STS2EntityRunEnv(
        tensorizer_config=tensorizer_config,
        max_steps=max_steps,
    )
    rewards: list[float] = []
    floors: list[int] = []
    lengths: list[int] = []
    wins = 0
    terminated_episodes = 0
    truncated_episodes = 0

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(episode_reward)
        floors.append(int(info.get("floor", 0)))
        lengths.append(steps)
        if bool(terminated) and episode_reward > 0:
            wins += 1
        terminated_episodes += int(bool(terminated))
        truncated_episodes += int(bool(truncated))
    env.close()
    return {
        "episodes": episodes,
        "wins": wins,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "win_rate": wins / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)),
        "mean_floor": float(np.mean(floors)),
        "max_floor": max(floors, default=0),
        "mean_episode_length": float(np.mean(lengths)),
        "deterministic": deterministic,
        "seed": seed,
    }


def evaluate_random(
    *,
    episodes: int,
    seed: int,
    tensorizer_config: TensorizerConfig,
    max_steps: int,
) -> dict[str, Any]:
    """Run a mask-respecting random baseline on identical episode seeds."""
    from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv

    env = STS2EntityRunEnv(
        tensorizer_config=tensorizer_config,
        max_steps=max_steps,
    )
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    floors: list[int] = []
    lengths: list[int] = []
    wins = 0
    terminated_episodes = 0
    truncated_episodes = 0
    for episode in range(episodes):
        _, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            valid = np.flatnonzero(env.action_masks())
            action = int(rng.choice(valid))
            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(episode_reward)
        floors.append(int(info.get("floor", 0)))
        lengths.append(steps)
        if bool(terminated) and episode_reward > 0:
            wins += 1
        terminated_episodes += int(bool(terminated))
        truncated_episodes += int(bool(truncated))
    env.close()
    return {
        "episodes": episodes,
        "wins": wins,
        "terminated_episodes": terminated_episodes,
        "truncated_episodes": truncated_episodes,
        "win_rate": wins / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)),
        "mean_floor": float(np.mean(floors)),
        "max_floor": max(floors, default=0),
        "mean_episode_length": float(np.mean(lengths)),
        "seed": seed,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorizer_config = TensorizerConfig(max_entities=args.max_entities)
    model_config = TypedSetTransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        num_memory_tokens=args.num_memory_tokens,
        num_isab_layers=args.num_isab_layers,
        feedforward_multiplier=args.feedforward_multiplier,
        dropout=args.dropout,
    )
    model_config.validate()

    factories = [
        make_env_factory(
            args.seed + index,
            tensorizer_config=tensorizer_config,
            max_steps=args.max_steps,
        )
        for index in range(args.n_envs)
    ]
    env = (
        SubprocVecEnv(factories)
        if args.n_envs > 1
        else DummyVecEnv(factories)
    )
    model = build_model(
        args.algorithm,
        args.architecture,
        env,
        model_config=model_config,
        tensorizer_config=tensorizer_config,
        args=args,
    )
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(output_dir / "checkpoints"),
            name_prefix="typed_set_v2",
        ))
    periodic_eval_env = None
    if args.eval_freq > 0:
        periodic_eval_env = DummyVecEnv([
            make_env_factory(
                args.seed + 50_000,
                tensorizer_config=tensorizer_config,
                max_steps=args.max_steps,
            )
        ])
        callbacks.append(MaskableEvalCallback(
            periodic_eval_env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.periodic_eval_episodes,
            deterministic=True,
        ))
    callback = CallbackList(callbacks) if callbacks else None

    started = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )
    elapsed = time.perf_counter() - started
    model_path = output_dir / "final_model"
    model.save(model_path)
    env.close()
    if periodic_eval_env is not None:
        periodic_eval_env.close()

    metrics = evaluate(
        model,
        episodes=args.eval_episodes,
        seed=args.seed + 100_000,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
    )
    random_metrics = evaluate_random(
        episodes=args.eval_episodes,
        seed=args.seed + 100_000,
        tensorizer_config=tensorizer_config,
        max_steps=args.max_steps,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.policy.parameters()
    )
    metadata = {
        "algorithm": args.algorithm,
        "architecture": args.architecture,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.total_timesteps / max(elapsed, 1e-9),
        "parameter_count": parameter_count,
        "tensorizer": asdict(tensorizer_config),
        "tensorizer_layout_hash": tensorizer_config.feature_layout_hash(),
        "model": asdict(model_config),
        "interface": schema_manifest(),
        "training": {
            key: getattr(args, key)
            for key in (
                "lr", "n_steps", "batch_size", "n_epochs", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                "max_grad_norm", "max_steps", "device",
                "checkpoint_freq", "eval_freq", "periodic_eval_episodes",
            )
        },
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
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-entities", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-inducing-points", type=int, default=16)
    parser.add_argument("--num-memory-tokens", type=int, default=8)
    parser.add_argument("--num-isab-layers", type=int, default=2)
    parser.add_argument("--feedforward-multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Periodic masked evaluation frequency; 0 disables it.",
    )
    parser.add_argument("--periodic-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard-log", default=None)
    parser.add_argument(
        "--output-dir",
        default="output/typed_set_v2",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
