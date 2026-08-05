"""Regression tests for the first capability-oriented agent iteration."""

from __future__ import annotations

import numpy as np

from sts2_env.agent_v2.categorical_vocabulary import (
    DEFAULT_CATEGORICAL_VOCABULARY,
    UNKNOWN_ID,
    categorical_id,
)
from sts2_env.agent_v2.tensorizer import TensorizerConfig, tensorize_snapshot
from sts2_env.core.enums import CardId, PowerId
from sts2_env.gym_env.reward_shaping import (
    RunRewardShapingConfig,
    shaped_run_reward,
)
from sts2_env.relics.base import RelicId
from scripts.analyze_training_curve import summarize_curve
from scripts.train_agent_v2 import linear_schedule
from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.core.selection import CardChoiceOption, PendingCardChoice
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run.run_manager import RunManager
from sts2_env.training.capability_eval import CapabilityScoreConfig


def test_known_content_ids_have_collision_free_vocabulary_entries() -> None:
    for enum_type in (CardId, PowerId, RelicId):
        encoded = [categorical_id(member.name, strict=True) for member in enum_type]
        assert len(encoded) == len(set(encoded))
        assert UNKNOWN_ID not in encoded


def test_vocabulary_is_sorted_unique_and_versioned() -> None:
    vocabulary = DEFAULT_CATEGORICAL_VOCABULARY
    assert vocabulary.tokens == tuple(sorted(set(vocabulary.tokens)))
    assert len(vocabulary.vocabulary_hash) == 64
    assert vocabulary.size == len(vocabulary.tokens) + 2


def test_afflictions_and_enchantments_use_individual_explicit_ids() -> None:
    config = TensorizerConfig(max_entities=8)
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "cards": [{
            "entity_id": "card:player:0:1",
            "card_id": "STRIKE_IRONCLAD",
            "owner_id": "player:0",
            "zone": "hand",
            "afflictions": {"BOUND": 1, "ENTANGLED": 1},
            "enchantments": {"ADROIT": 1, "NIMBLE": 1},
        }],
        "candidates": [],
    }
    mask = np.zeros(157, dtype=np.int8)
    mask[0] = 1

    observation = tensorize_snapshot(snapshot, mask, config)
    row = observation["entity_categorical"][0]

    assert set(row[7:9]) == {
        categorical_id("BOUND", strict=True),
        categorical_id("ENTANGLED", strict=True),
    }
    assert set(row[9:11]) == {
        categorical_id("ADROIT", strict=True),
        categorical_id("NIMBLE", strict=True),
    }
    assert UNKNOWN_ID not in row


def test_reward_shaping_is_decomposed_and_penalizes_hp_loss() -> None:
    config = RunRewardShapingConfig()
    reward, components = shaped_run_reward(
        0.0,
        previous_floor=3,
        current_floor=4,
        previous_hp=80,
        current_hp=72,
        previous_max_hp=80,
        previous_phase="COMBAT",
        current_phase="CARD_REWARD",
        run_over=False,
        player_won=False,
        config=config,
    )

    assert components == {
        "base": 0.0,
        "floor": 0.02,
        "combat_win": 0.02,
        "hp_loss": -0.02,
        "step": -0.001,
    }
    assert reward == sum(components.values())


def test_no_progress_stalling_has_no_discount_advantage() -> None:
    gamma = 0.999
    step_penalty = 1.0 - gamma
    horizon = 2000
    delayed_loss = -step_penalty * sum(
        gamma**step for step in range(horizon)
    ) - gamma**horizon

    assert np.isclose(delayed_loss, -1.0)


def test_training_curve_summary_uses_timestep_buckets() -> None:
    records = [
        {
            "timesteps": 999,
            "floor": 3,
            "return": -1.1,
            "length": 40,
            "won": False,
            "truncated": False,
        },
        {
            "timesteps": 1000,
            "floor": 7,
            "return": 1.0,
            "length": 100,
            "won": True,
            "truncated": False,
        },
    ]

    summary = summarize_curve(records, bucket_steps=1000)

    assert summary["episodes"] == 2
    assert [bucket["mean_floor"] for bucket in summary["buckets"]] == [3, 7]
    assert summary["buckets"][1]["wins"] == 1


def _install_run_choice(env: STS2RunEnv) -> None:
    assert env._mgr is not None
    env._mgr._phase = RunManager.PHASE_MAP_CHOICE
    env._mgr.run_state.pending_choice = PendingCardChoice(
        prompt="choose up to two",
        options=[
            CardChoiceOption(make_strike_ironclad(), "deck"),
            CardChoiceOption(make_defend_ironclad(), "deck"),
        ],
        resolver=lambda _selected: None,
        min_choices=1,
        max_choices=2,
    )


def test_entity_v4_multi_select_mask_is_monotonic() -> None:
    env = STS2EntityRunEnv(max_steps=20)
    env.reset(seed=1)
    _install_run_choice(env.run_env)

    assert env.action_masks()[1:3].tolist() == [1, 1]
    env.step(1)
    mask = env.action_masks()

    assert mask[0] == 1  # confirm
    assert mask[1] == 0  # selected item cannot be toggled off
    assert mask[2] == 1  # another item can still be added
    env.step(2)
    assert env.action_masks()[:3].tolist() == [1, 0, 0]
    env.close()


def test_frozen_v1_multi_select_still_allows_deselection() -> None:
    env = STS2RunEnv(max_steps=20)
    env.reset(seed=1)
    _install_run_choice(env)

    env.step(1)

    assert env.action_masks()[1] == 1
    env.close()


def test_capability_score_penalizes_truncation_and_prioritizes_wins() -> None:
    config = CapabilityScoreConfig()

    clean = config.score({
        "episodes": 10,
        "win_rate": 0.0,
        "mean_floor": 7.0,
        "truncated_episodes": 0,
    })
    looping = config.score({
        "episodes": 10,
        "win_rate": 0.0,
        "mean_floor": 8.0,
        "truncated_episodes": 2,
    })
    winner = config.score({
        "episodes": 10,
        "win_rate": 0.1,
        "mean_floor": 5.0,
        "truncated_episodes": 0,
    })

    assert clean > looping
    assert winner > clean


def test_candidate_policy_experts_are_global_state_gated() -> None:
    torch = __import__("torch")
    from sts2_env.models.typed_set_transformer import CandidateScoringHead

    head = CandidateScoringHead(d_model=8, num_actions=3, num_experts=4)
    features = torch.randn(2, 8 * 4)

    logits = head(features)

    assert logits.shape == (2, 3)
    assert head.gate is not None
    assert len(head.experts) == 4


def test_linear_learning_rate_schedule_reaches_configured_endpoints() -> None:
    schedule = linear_schedule(3e-4, 3e-5)

    assert schedule(1.0) == 3e-4
    assert schedule(0.0) == 3e-5
    assert np.isclose(schedule(0.5), 1.65e-4)
