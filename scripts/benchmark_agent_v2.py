"""Profile entity-v2 environment, policy compute, and MaskablePPO training.

Examples:
    python scripts/benchmark_agent_v2.py --device cpu
    python scripts/benchmark_agent_v2.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from sts2_env.agent_v2.tensorizer import TensorizerConfig
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv
from sts2_env.gym_env.entity_run_env import ENTITY_ACTION_SEMANTICS_VERSION
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.models.typed_set_transformer import (
    TypedSetMaskableActorCriticPolicy,
    TypedSetTransformerConfig,
)
from train_agent_v2 import make_env_factory


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _model(
    env: Any,
    *,
    device: str,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 4,
) -> MaskablePPO:
    tensorizer = TensorizerConfig(max_entities=384)
    return MaskablePPO(
        TypedSetMaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        seed=109,
        device=device,
        verbose=0,
        policy_kwargs={
            "typed_set_config": TypedSetTransformerConfig(),
            "categorical_vocab_size": tensorizer.categorical_vocab_size,
            "categorical_vocabulary_hash": (
                tensorizer.categorical_vocabulary_hash
            ),
            "tensorizer_layout_hash": tensorizer.feature_layout_hash(),
            "action_semantics_version": ENTITY_ACTION_SEMANTICS_VERSION,
            "ortho_init": False,
        },
    )


def benchmark_environment(steps: int) -> dict[str, Any]:
    """Measure simulator work with and without v2 snapshot tensorization."""

    def run(env: Any, seed_offset: int) -> float:
        _, info = env.reset(seed=109 + seed_offset)
        rng = np.random.default_rng(109)
        started = time.perf_counter()
        for index in range(steps):
            mask = env.action_masks()
            action = int(rng.choice(np.flatnonzero(mask)))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset(seed=110 + seed_offset + index)
        elapsed = time.perf_counter() - started
        env.close()
        return elapsed

    raw_elapsed = run(STS2RunEnv(max_steps=2000), 0)
    entity_elapsed = run(
        STS2EntityRunEnv(
            tensorizer_config=TensorizerConfig(max_entities=384),
            max_steps=2000,
        ),
        0,
    )
    return {
        "steps": steps,
        "raw_seconds": raw_elapsed,
        "raw_steps_per_second": steps / raw_elapsed,
        "entity_v2_seconds": entity_elapsed,
        "entity_v2_steps_per_second": steps / entity_elapsed,
        "entity_observation_overhead_seconds_per_step": (
            entity_elapsed - raw_elapsed
        ) / steps,
    }


def _batched_observation(
    observation: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(
            np.repeat(value[None, ...], batch_size, axis=0),
            device=device,
        )
        for key, value in observation.items()
    }


def benchmark_policy(
    device: str,
    *,
    forward_iterations: int,
    backward_iterations: int,
) -> dict[str, Any]:
    """Measure policy-only compute after observations are already tensors."""
    env = STS2EntityRunEnv(
        tensorizer_config=TensorizerConfig(max_entities=384),
        max_steps=2000,
    )
    observation, _ = env.reset(seed=109)
    mask = env.action_masks().astype(bool)
    model = _model(env, device=device)
    policy = model.policy
    policy.set_training_mode(False)
    results: dict[str, Any] = {}

    for batch_size in (1, 4, 256):
        obs = _batched_observation(observation, batch_size, device)
        masks = np.repeat(mask[None, :], batch_size, axis=0)
        warmup = 5
        with torch.inference_mode():
            for _ in range(warmup):
                policy(obs, action_masks=masks)
            _synchronize(device)
            started = time.perf_counter()
            for _ in range(forward_iterations):
                policy(obs, action_masks=masks)
            _synchronize(device)
        elapsed = time.perf_counter() - started
        results[f"forward_batch_{batch_size}"] = {
            "iterations": forward_iterations,
            "seconds": elapsed,
            "batches_per_second": forward_iterations / elapsed,
            "observations_per_second": (
                forward_iterations * batch_size / elapsed
            ),
            "milliseconds_per_batch": 1000.0 * elapsed / forward_iterations,
        }

    policy.set_training_mode(True)
    batch_size = 256
    obs = _batched_observation(observation, batch_size, device)
    masks = torch.as_tensor(
        np.repeat(mask[None, :], batch_size, axis=0),
        device=device,
    )
    valid_action = int(np.flatnonzero(mask)[0])
    actions = torch.full(
        (batch_size,), valid_action, dtype=torch.long, device=device
    )
    for _ in range(3):
        policy.optimizer.zero_grad(set_to_none=True)
        values, log_prob, entropy = policy.evaluate_actions(
            obs, actions, action_masks=masks
        )
        loss = values.square().mean() - log_prob.mean()
        if entropy is not None:
            loss = loss - 0.01 * entropy.mean()
        loss.backward()
        policy.optimizer.step()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(backward_iterations):
        policy.optimizer.zero_grad(set_to_none=True)
        values, log_prob, entropy = policy.evaluate_actions(
            obs, actions, action_masks=masks
        )
        loss = values.square().mean() - log_prob.mean()
        if entropy is not None:
            loss = loss - 0.01 * entropy.mean()
        loss.backward()
        policy.optimizer.step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    results["forward_backward_batch_256"] = {
        "iterations": backward_iterations,
        "seconds": elapsed,
        "batches_per_second": backward_iterations / elapsed,
        "observations_per_second": (
            backward_iterations * batch_size / elapsed
        ),
        "milliseconds_per_batch": 1000.0 * elapsed / backward_iterations,
    }
    results["parameter_count"] = sum(
        parameter.numel() for parameter in policy.parameters()
    )
    if device.startswith("cuda"):
        results["peak_cuda_memory_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20
        )
    env.close()
    return results


class _PhaseTimingCallback(BaseCallback):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device_name = device
        self.rollout_started: float | None = None
        self.update_started: float | None = None
        self.rollout_seconds: list[float] = []
        self.update_seconds: list[float] = []

    def _now(self) -> float:
        _synchronize(self.device_name)
        return time.perf_counter()

    def _on_rollout_start(self) -> None:
        now = self._now()
        if self.update_started is not None:
            self.update_seconds.append(now - self.update_started)
            self.update_started = None
        self.rollout_started = now

    def _on_rollout_end(self) -> None:
        now = self._now()
        if self.rollout_started is not None:
            self.rollout_seconds.append(now - self.rollout_started)
        self.rollout_started = None
        self.update_started = now

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        now = self._now()
        if self.update_started is not None:
            self.update_seconds.append(now - self.update_started)
            self.update_started = None


def benchmark_ppo(
    device: str,
    *,
    total_timesteps: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
) -> dict[str, Any]:
    tensorizer = TensorizerConfig(max_entities=384)
    env = SubprocVecEnv([
        make_env_factory(
            109 + index,
            tensorizer_config=tensorizer,
            max_steps=2000,
            max_combat_turns=50,
        )
        for index in range(n_envs)
    ])
    model = _model(
        env,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    callback = _PhaseTimingCallback(device)
    _synchronize(device)
    started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    env.close()
    rollout_seconds = sum(callback.rollout_seconds)
    update_seconds = sum(callback.update_seconds)
    return {
        "requested_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "total_seconds": elapsed,
        "steps_per_second": model.num_timesteps / elapsed,
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "other_seconds": max(0.0, elapsed - rollout_seconds - update_seconds),
        "rollout_fraction": rollout_seconds / elapsed,
        "update_fraction": update_seconds / elapsed,
        "iterations": len(callback.rollout_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--environment-steps", type=int, default=2000)
    parser.add_argument("--forward-iterations", type=int, default=20)
    parser.add_argument("--backward-iterations", type=int, default=10)
    parser.add_argument("--ppo-timesteps", type=int, default=4096)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--skip-environment", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    result: dict[str, Any] = {
        "device": args.device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "torch_threads": torch.get_num_threads(),
    }
    if not args.skip_environment:
        result["environment"] = benchmark_environment(
            args.environment_steps
        )
    if not args.skip_policy:
        result["policy"] = benchmark_policy(
            args.device,
            forward_iterations=args.forward_iterations,
            backward_iterations=args.backward_iterations,
        )
    if not args.skip_ppo:
        result["ppo"] = benchmark_ppo(
            args.device,
            total_timesteps=args.ppo_timesteps,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
