"""Structured entity-v2 view of the full-run Gymnasium environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from sts2_env.agent_v2.tensorizer import (
    DEFAULT_TENSORIZER_CONFIG,
    LegacyV5TensorizerConfig,
    LegacyV6TensorizerConfig,
    TensorizerConfig,
    observation_space,
    tensorize_snapshot,
)
from sts2_env.agent_v2.unknown_diagnostics import UnknownTokenDiagnostics
from sts2_env.gym_env.run_env import ENTITY_EXTRA_CHOICE_SLOTS, STS2RunEnv

ENTITY_ACTION_SEMANTICS_VERSION = "entity-actions-v3-extended-choice"


class STS2EntityRunEnv(gymnasium.Wrapper):
    """Expose typed padded sets with versioned entity action semantics."""

    def __init__(
        self,
        env: STS2RunEnv | None = None,
        *,
        tensorizer_config: TensorizerConfig = DEFAULT_TENSORIZER_CONFIG,
        unknown_diagnostics_path: Path | None = None,
        unknown_worker_index: int = 0,
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
            run_env_kwargs.setdefault("encode_legacy_observations", False)
            run_env_kwargs.setdefault("extra_choice_slots", ENTITY_EXTRA_CHOICE_SLOTS)
            is_legacy = isinstance(
                tensorizer_config,
                (LegacyV5TensorizerConfig, LegacyV6TensorizerConfig),
            )
            run_env_kwargs.setdefault("start_with_neow", not is_legacy)
            run_env_kwargs.setdefault("act1_variant", "overgrowth" if is_legacy else "random")
        base = env or STS2RunEnv(**run_env_kwargs)
        if base.action_space.n != tensorizer_config.num_actions:
            raise ValueError(
                "Entity action count must match the tensorizer: "
                f"{base.action_space.n} != {tensorizer_config.num_actions}"
            )
        # This wrapper always replaces the base environment's v1 vector with
        # a structured observation, so constructing that vector is wasted.
        base.encode_legacy_observations = False
        super().__init__(base)
        self.tensorizer_config = tensorizer_config
        self.unknown_diagnostics = (
            UnknownTokenDiagnostics(
                unknown_diagnostics_path,
                worker_index=unknown_worker_index,
            )
            if unknown_diagnostics_path is not None else None
        )
        self.observation_space = observation_space(tensorizer_config)
        self._cached_action_mask: np.ndarray | None = None

    @property
    def run_env(self) -> STS2RunEnv:
        return self.env  # type: ignore[return-value]

    def _structured_observation(
        self,
        action_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if action_mask is None:
            action_mask = self.action_masks()
        return tensorize_snapshot(
            self.run_env.entity_observation(),
            action_mask,
            self.tensorizer_config,
            unknown_diagnostics=self.unknown_diagnostics,
        )

    def close(self) -> None:
        if self.unknown_diagnostics is not None:
            self.unknown_diagnostics.flush()
        super().close()

    def _cache_action_mask(self, info: dict[str, Any]) -> np.ndarray:
        mask = np.asarray(info["action_mask"], dtype=np.int8)
        self._cached_action_mask = mask
        return mask

    def invalidate_action_mask_cache(self) -> None:
        """Invalidate the mask after out-of-band mutation of ``run_env``.

        Normal Gym usage changes state only through reset/step and never needs
        this. It is provided for parity tools and tests that deliberately edit
        private simulator state in place.
        """
        self._cached_action_mask = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._cached_action_mask = None
        _, info = self.env.reset(seed=seed, options=options)
        action_mask = self._cache_action_mask(info)
        return self._structured_observation(action_mask), info

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
        self._cached_action_mask = None
        _, reward, terminated, truncated, info = self.env.step(action)
        action_mask = self._cache_action_mask(info)
        return (
            self._structured_observation(action_mask),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        if self._cached_action_mask is None:
            self._cached_action_mask = self.run_env.action_masks()
        return self._cached_action_mask
