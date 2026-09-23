"""Typed Set Transformer encoder and candidate-scoring MaskablePPO policy.

No positional encoding is used. Entity order therefore has no semantic
meaning; action slots are represented explicitly as a categorical feature of
candidate tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sts2_env.agent_v2.categorical_vocabulary import (
    VOCABULARY_HASH,
    VOCABULARY_SIZE,
)


@dataclass(frozen=True, slots=True)
class TypedSetTransformerConfig:
    """Architecture-only configuration, independent of STS2 tensorization."""

    d_model: int = 64
    num_heads: int = 4
    num_inducing_points: int = 16
    num_memory_tokens: int = 8
    num_isab_layers: int = 2
    feedforward_multiplier: int = 2
    dropout: float = 0.0
    num_policy_experts: int = 4

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if min(
            self.d_model,
            self.num_heads,
            self.num_inducing_points,
            self.num_memory_tokens,
            self.num_isab_layers,
            getattr(self, "num_policy_experts", 1),
        ) <= 0:
            raise ValueError("Typed Set Transformer sizes must be positive")


class MultiheadAttentionBlock(nn.Module):
    """Set Transformer MAB with pre-normalized attention and feed-forward."""

    def __init__(self, config: TypedSetTransformerConfig) -> None:
        super().__init__()
        d_model = config.d_model
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        hidden = d_model * config.feedforward_multiplier
        self.output_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        output = query + attended
        output = output + self.feedforward(self.output_norm(output))
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1)
        return output


class InducedSetAttentionBlock(nn.Module):
    """ISAB reduces entity self-attention from O(n²) to O(nm)."""

    def __init__(self, config: TypedSetTransformerConfig) -> None:
        super().__init__()
        self.inducing = nn.Parameter(
            torch.empty(config.num_inducing_points, config.d_model)
        )
        nn.init.normal_(self.inducing, std=0.02)
        self.inducing_from_entities = MultiheadAttentionBlock(config)
        self.entities_from_inducing = MultiheadAttentionBlock(config)

    def forward(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = entities.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.inducing_from_entities(
            inducing,
            entities,
            memory_padding_mask=~entity_mask,
        )
        return self.entities_from_inducing(
            entities,
            hidden,
            query_mask=entity_mask,
        )


class PoolingByMultiheadAttention(nn.Module):
    """PMA produces a small permutation-invariant memory bank."""

    def __init__(
        self,
        config: TypedSetTransformerConfig,
        num_seeds: int,
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(num_seeds, config.d_model))
        nn.init.normal_(self.seeds, std=0.02)
        self.attention = MultiheadAttentionBlock(config)

    def forward(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        seeds = self.seeds.unsqueeze(0).expand(entities.shape[0], -1, -1)
        return self.attention(
            seeds,
            entities,
            memory_padding_mask=~entity_mask,
        )


class _CategoricalNumericEmbedding(nn.Module):
    """Sum field embeddings and a numeric projection into one token."""

    def __init__(
        self,
        *,
        num_categorical_fields: int,
        num_numeric_fields: int,
        categorical_vocab_size: int = VOCABULARY_SIZE,
        d_model: int,
    ) -> None:
        super().__init__()
        self.categorical = nn.ModuleList([
            nn.Embedding(categorical_vocab_size, d_model, padding_idx=0)
            for _ in range(num_categorical_fields)
        ])
        self.numeric = nn.Sequential(
            nn.Linear(num_numeric_fields, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.categorical_scale = 1.0 / math.sqrt(num_categorical_fields)

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        categorical = categorical.long()
        embedded = sum(
            embedding(categorical[..., field])
            for field, embedding in enumerate(self.categorical)
        ) * self.categorical_scale
        return self.norm(embedded + self.numeric(numeric.float()))


class TypedSetTransformerExtractor(BaseFeaturesExtractor):
    """Encode entity sets and aligned action-candidate tokens.

    The returned flat feature has ``[global, candidate_0, ..., candidate_A]``.
    A separate candidate head can replace the policy algorithm without
    changing this encoder.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        config: TypedSetTransformerConfig | None = None,
        categorical_vocab_size: int,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("TypedSetTransformerExtractor requires Dict observations")
        self.config = config or TypedSetTransformerConfig()
        self.config.validate()
        num_actions = observation_space["candidate_categorical"].shape[0]
        features_dim = self.config.d_model * (1 + num_actions)
        super().__init__(observation_space, features_dim)
        self.num_actions = num_actions

        entity_cat = observation_space["entity_categorical"].shape[-1]
        entity_num = observation_space["entity_numeric"].shape[-1]
        candidate_cat = observation_space["candidate_categorical"].shape[-1]
        candidate_num = observation_space["candidate_numeric"].shape[-1]
        global_cat = observation_space["global_categorical"].shape[-1]
        global_num = observation_space["global_numeric"].shape[-1]
        d_model = self.config.d_model

        self.entity_embedding = _CategoricalNumericEmbedding(
            num_categorical_fields=entity_cat,
            num_numeric_fields=entity_num,
            categorical_vocab_size=categorical_vocab_size,
            d_model=d_model,
        )
        self.candidate_embedding = _CategoricalNumericEmbedding(
            num_categorical_fields=candidate_cat,
            num_numeric_fields=candidate_num,
            categorical_vocab_size=categorical_vocab_size,
            d_model=d_model,
        )
        self.global_embedding = _CategoricalNumericEmbedding(
            num_categorical_fields=global_cat,
            num_numeric_fields=global_num,
            categorical_vocab_size=categorical_vocab_size,
            d_model=d_model,
        )

        # Entity type is categorical field zero. FiLM-like modulation gives
        # each type a distinct residual route without one expensive expert
        # network per token type.
        self.type_scale = nn.Embedding(categorical_vocab_size, d_model)
        self.type_shift = nn.Embedding(categorical_vocab_size, d_model)
        nn.init.zeros_(self.type_scale.weight)
        nn.init.zeros_(self.type_shift.weight)
        self.typed_feedforward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * self.config.feedforward_multiplier),
            nn.GELU(),
            nn.Linear(d_model * self.config.feedforward_multiplier, d_model),
        )

        self.entity_blocks = nn.ModuleList([
            InducedSetAttentionBlock(self.config)
            for _ in range(self.config.num_isab_layers)
        ])
        self.memory_pool = PoolingByMultiheadAttention(
            self.config, self.config.num_memory_tokens
        )
        self.candidate_cross_attention = MultiheadAttentionBlock(self.config)
        self.source_projection = nn.Linear(d_model, d_model, bias=False)
        self.target_projection = nn.Linear(d_model, d_model, bias=False)
        self.global_norm = nn.LayerNorm(d_model)
        self.candidate_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _pointed_entities(
        entities: torch.Tensor,
        rows: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        """Gather one contextual entity per candidate; -1 means no relation."""
        valid = rows >= 0
        indices = rows.clamp(min=0).long()
        selected = torch.gather(
            entities,
            1,
            indices.unsqueeze(-1).expand(-1, -1, entities.shape[-1]),
        )
        return projection(selected) * valid.unsqueeze(-1)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        entity_mask = observations["entity_mask"].bool()
        # MHA cannot consume an all-masked row. The fallback token is zero and
        # affects only malformed/terminal synthetic observations.
        empty = ~entity_mask.any(dim=1)
        if empty.any():
            entity_mask = entity_mask.clone()
            entity_mask[empty, 0] = True

        entities = self.entity_embedding(
            observations["entity_categorical"],
            observations["entity_numeric"],
        )
        entity_types = observations["entity_categorical"][..., 0].long()
        typed = self.typed_feedforward(entities)
        scale = 1.0 + torch.tanh(self.type_scale(entity_types))
        entities = entities + typed * scale + self.type_shift(entity_types)
        entities = entities * entity_mask.unsqueeze(-1)
        for block in self.entity_blocks:
            entities = block(entities, entity_mask)

        memory = self.memory_pool(entities, entity_mask)
        global_token = self.global_embedding(
            observations["global_categorical"].unsqueeze(1),
            observations["global_numeric"].unsqueeze(1),
        ).squeeze(1)
        global_state = self.global_norm(global_token + memory.mean(dim=1))

        candidates = self.candidate_embedding(
            observations["candidate_categorical"],
            observations["candidate_numeric"],
        )
        candidates = candidates + self._pointed_entities(
            entities,
            observations["candidate_source_row"],
            self.source_projection,
        ) + self._pointed_entities(
            entities,
            observations["candidate_target_row"],
            self.target_projection,
        )
        candidates = self.candidate_cross_attention(
            candidates, entities, memory_padding_mask=~entity_mask,
        )
        candidates = self.candidate_norm(
            candidates + global_state.unsqueeze(1)
        )
        return torch.cat(
            (global_state, candidates.flatten(start_dim=1)), dim=1
        )


class _TypedSetLatentExtractor(nn.Module):
    """Adapter matching the interface expected by SB3 actor-critic policies."""

    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.latent_dim_pi = d_model * (num_actions + 1)
        self.latent_dim_vf = d_model

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features, features[:, : self.d_model]

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, : self.d_model]


class CandidateScoringHead(nn.Module):
    """Global-state-gated experts score each semantic action candidate."""

    def __init__(
        self,
        d_model: int,
        num_actions: int,
        hidden_dim: int | None = None,
        num_experts: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_experts = num_experts
        hidden = hidden_dim or d_model
        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        if num_experts == 1:
            # Preserve the v2/v3 single-head state-dict key so existing
            # checkpoints remain loadable for evaluation and Bridge use.
            self.scorer: nn.Sequential | None = make_expert()
            self.experts = nn.ModuleList()
        else:
            self.scorer = None
            self.experts = nn.ModuleList([
                make_expert() for _ in range(num_experts)
            ])
        self.gate = (
            nn.Linear(d_model, num_experts)
            if num_experts > 1
            else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_state = features[:, : self.d_model]
        candidates = features[:, self.d_model:].reshape(
            -1, self.num_actions, self.d_model
        )
        expanded_global = global_state.unsqueeze(1).expand_as(candidates)
        inputs = torch.cat((expanded_global, candidates), dim=-1)
        if self.scorer is not None:
            return self.scorer(inputs).squeeze(-1)
        expert_logits = torch.stack(
            [expert(inputs).squeeze(-1) for expert in self.experts],
            dim=-1,
        )
        if self.gate is None:
            return expert_logits[..., 0]
        weights = torch.softmax(self.gate(global_state), dim=-1)
        return torch.sum(expert_logits * weights.unsqueeze(1), dim=-1)


class TypedSetMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """MaskablePPO-compatible policy with a Typed Set Transformer backbone."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        typed_set_config: TypedSetTransformerConfig | None = None,
        categorical_vocab_size: int = VOCABULARY_SIZE,
        categorical_vocabulary_hash: str = VOCABULARY_HASH,
        categorical_buckets: int | None = None,
        tensorizer_layout_hash: str | None = None,
        action_semantics_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Typed set policy currently requires Discrete actions")
        if categorical_buckets is not None:
            # Constructor-only compatibility lets SB3 deserialize an old v2
            # checkpoint far enough for the runner to issue its explicit
            # tensor-layout incompatibility error. It does not make hash
            # observations compatible with the v3 vocabulary.
            categorical_vocab_size = categorical_buckets
        self.typed_set_config = (
            typed_set_config or TypedSetTransformerConfig()
        )
        self.categorical_vocab_size = categorical_vocab_size
        self.categorical_vocabulary_hash = categorical_vocabulary_hash
        self.tensorizer_layout_hash = tensorizer_layout_hash
        self.action_semantics_version = action_semantics_version
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            features_extractor_class=TypedSetTransformerExtractor,
            features_extractor_kwargs={
                "config": self.typed_set_config,
                "categorical_vocab_size": categorical_vocab_size,
            },
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _TypedSetLatentExtractor(
            self.typed_set_config.d_model,
            int(self.action_space.n),  # type: ignore[union-attr]
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        d_model = self.typed_set_config.d_model
        num_actions = int(self.action_space.n)  # type: ignore[union-attr]
        self.action_net = CandidateScoringHead(
            d_model,
            num_actions,
            num_experts=getattr(
                self.typed_set_config,
                "num_policy_experts",
                1,
            ),
        )
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        if self.ortho_init:
            for module, gain in (
                (self.features_extractor, np.sqrt(2)),
                (self.action_net, 0.01),
                (self.value_net, 1.0),
            ):
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({
            "typed_set_config": self.typed_set_config,
            "categorical_vocab_size": self.categorical_vocab_size,
            "categorical_vocabulary_hash": self.categorical_vocabulary_hash,
            "tensorizer_layout_hash": self.tensorizer_layout_hash,
            "action_semantics_version": self.action_semantics_version,
        })
        return data
