"""Structured entity-v2 view of the full-run Gymnasium environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from sts2_env.agent_v2.tensorizer import (
    DEFAULT_TENSORIZER_CONFIG,
    TensorizerConfig,
    observation_space,
    tensorize_snapshot,
)
from sts2_env.gym_env.run_env import STS2RunEnv

ENTITY_ACTION_SEMANTICS_VERSION = "entity-actions-v2-monotonic-choice"


class STS2EntityRunEnv(gymnasium.Wrapper):
    """Expose typed padded sets with versioned entity action semantics."""

    def __init__(
        self,
        env: STS2RunEnv | None = None,
        *,
        tensorizer_config: TensorizerConfig = DEFAULT_TENSORIZER_CONFIG,
        **run_env_kwargs: Any,
    ) -> None:
        if env is not None and run_env_kwargs:
            raise ValueError(
                "Pass either env or STS2RunEnv keyword arguments, not both"
            )
        if env is None:
            # The entity-v4 policy uses monotonic multi-select actions. A
            # chosen item cannot be toggled off, eliminating deterministic
            # two-cycles while preserving every final subset and confirm
            # choice. Frozen v1 STS2RunEnv behavior remains unchanged.
            run_env_kwargs.setdefault("monotonic_choices", True)
        base = env or STS2RunEnv(**run_env_kwargs)
        super().__init__(base)
        self.tensorizer_config = tensorizer_config
        self.observation_space = observation_space(tensorizer_config)

    @property
    def run_env(self) -> STS2RunEnv:
        return self.env  # type: ignore[return-value]

    def _structured_observation(self) -> dict[str, np.ndarray]:
        return tensorize_snapshot(
            self.run_env.entity_observation(),
            self.run_env.action_masks(),
            self.tensorizer_config,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        return self._structured_observation(), info

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._structured_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return self.run_env.action_masks()
