"""Focused tests for bridge-side potion action masking and decoding."""

from __future__ import annotations

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.core.constants import (
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    POTION_ACTION_START,
    POTION_TARGET_OPTIONS,
)


def _base_state() -> dict:
    return {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [],
        "enemies": [
            {
                "id": "NIBBIT",
                "hp": 30,
                "max_hp": 30,
                "block": 0,
                "is_alive": True,
                "intent": "ATTACK",
                "intent_damage": 6,
                "intent_hits": 1,
                "powers": [],
            }
        ],
        "draw_pile_count": 5,
        "discard_pile_count": 2,
        "exhaust_pile_count": 1,
    }


def test_encode_observation_keeps_reserved_pile_summary_slots_zero() -> None:
    adapter = StateAdapter()
    obs = adapter.encode_observation(_base_state())

    pile_start = 4 + 6 + 50
    assert obs[pile_start] == 5 / 20.0
    assert obs[pile_start + 1] == 2 / 20.0
    assert obs[pile_start + 2] == 1 / 20.0
    assert obs[pile_start + 3] == 0.0
    assert obs[pile_start + 4] == 0.0
    assert obs[pile_start + 5] == 0.0


def test_compute_action_mask_includes_targeted_and_untargeted_potions() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["potions"] = [
        {"slot": 0, "id": "BlockPotion", "usage": "CombatOnly", "target": "Self", "can_use": True},
        {
            "slot": 1,
            "id": "FirePotion",
            "usage": "AnyTime",
            "target": "AnyEnemy",
            "requires_target": True,
            "can_use": True,
        },
        {"slot": 2, "id": "FairyInABottle", "usage": "Automatic", "target": "Self", "can_use": True},
    ]

    mask = adapter.compute_action_mask(state)

    assert mask[POTION_ACTION_START] == 1
    fire_base = POTION_ACTION_START + POTION_TARGET_OPTIONS
    assert mask[fire_base] == 0
    assert mask[fire_base + 1] == 1
    fairy_base = POTION_ACTION_START + 2 * POTION_TARGET_OPTIONS
    assert mask[fairy_base] == 0


def test_compute_action_mask_honors_authoritative_card_playability() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["player"]["energy"] = 0
    state["hand"] = [
        {
            "id": "INFECTION",
            "cost": -1,
            "playable": False,
            "target": "None",
            "type": "Status",
        },
        {
            "id": "GRAND_FINALE",
            "cost": 0,
            "playable": False,
            "target": "AnyEnemy",
            "type": "Attack",
        },
        {
            "id": "WHIRLWIND",
            "cost": -1,
            "playable": True,
            "target": "AllEnemies",
            "type": "Attack",
        },
    ]

    mask = adapter.compute_action_mask(state)

    infection_action = 1
    grand_finale_target_0 = 1 + MAX_HAND_SIZE + MAX_ENEMIES
    whirlwind_action = 3
    assert mask[infection_action] == 0
    assert mask[grand_finale_target_0] == 0
    assert mask[whirlwind_action] == 1


def test_compute_action_mask_keeps_legacy_cost_fallback() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["player"]["energy"] = 1
    state["hand"] = [
        {"id": "DEFEND_IRONCLAD", "cost": 2, "target": "Self", "type": "Skill"},
        {"id": "STRIKE_IRONCLAD", "cost": 1, "target": "AnyEnemy", "type": "Attack"},
    ]

    mask = adapter.compute_action_mask(state)

    strike_target_0 = 1 + MAX_HAND_SIZE + MAX_ENEMIES
    assert mask[1] == 0
    assert mask[strike_target_0] == 1


def test_decode_action_returns_out_of_hand_play_payload_for_potion() -> None:
    adapter = StateAdapter()
    decoded = adapter.decode_action(POTION_ACTION_START + POTION_TARGET_OPTIONS + 1, _base_state())

    assert decoded == {
        "type": "PLAY",
        "out_of_hand": True,
        "potion_slot": 1,
        "target_index": 0,
    }
