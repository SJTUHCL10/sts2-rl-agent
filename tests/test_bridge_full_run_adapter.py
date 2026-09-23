"""Tests for the live Bridge to full-run policy adapter."""

from __future__ import annotations

from sts2_env.bridge.full_run_adapter import FullRunStateAdapter
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, TOTAL_ACTIONS


def test_full_run_combat_observation_and_mask_embed_combat_interface():
    adapter = FullRunStateAdapter()
    state = {
        "type": "combat_action",
        "act": 1,
        "floor": 4,
        "player": {"hp": 50, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [
            {"id": "ASCENDERS_BANE", "cost": -1, "playable": False, "target": "None", "type": "Curse"},
            {"id": "STRIKE_IRONCLAD", "cost": 1, "playable": True, "target": "AnyEnemy", "type": "Attack"},
        ],
        "enemies": [{"id": "NIBBIT", "hp": 30, "max_hp": 40, "block": 0, "is_alive": True, "intent": "Attack", "powers": []}],
    }

    obs = adapter.encode_observation(state)
    mask = adapter.compute_action_mask(state)

    assert obs.shape == (RUN_OBS_SIZE,)
    assert obs[134] == 50 / 80
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask[1] == 0
    assert mask[16] == 1  # hand index 1 targeting enemy index 0
    assert mask[115:].sum() == 0


def test_full_run_map_mask_and_decode_use_bridge_node_index():
    adapter = FullRunStateAdapter()
    state = {
        "type": "map_select",
        "act": 1,
        "floor": 3,
        "nodes": [
            {"index": 7, "type": "Monster"},
            {"index": 9, "type": "Elite"},
        ],
    }

    mask = adapter.compute_action_mask(state)
    command = adapter.decode_action(116, state)

    assert mask[115:120].tolist() == [1, 1, 0, 0, 0]
    assert command == {"action": "choose", "index": 9}
    assert adapter.room_type == "Elite"


def test_full_run_card_reward_keeps_skip_distinct_from_extra_cards():
    adapter = FullRunStateAdapter()
    state = {
        "type": "card_reward",
        "can_skip": True,
        "cards": [{"index": i, "id": f"CARD_{i}"} for i in range(6)],
    }

    mask = adapter.compute_action_mask(state)

    assert mask[120:127].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert adapter.decode_action(123, state) == {"action": "skip"}
    assert adapter.decode_action(126, state) == {"action": "choose", "index": 5}


def test_full_run_reward_screen_maps_proceed_to_policy_skip_slot():
    adapter = FullRunStateAdapter()
    state = {
        "type": "reward_screen",
        "options": [
            {"action": "pick_reward", "index": 4, "enabled": True},
            {"action": "proceed", "index": 8, "enabled": True},
        ],
    }

    mask = adapter.compute_action_mask(state)

    assert mask[120:124].tolist() == [1, 0, 0, 1]
    assert adapter.decode_action(120, state) == {"action": "choose", "index": 4}
    assert adapter.decode_action(123, state) == {"action": "choose", "index": 8}


def test_finished_crystal_sphere_prioritizes_proceed_over_stale_cells():
    adapter = FullRunStateAdapter()
    state = {
        "type": "crystal_sphere",
        "minigame": {
            "finished": True,
            "divinations_remaining": 0,
        },
        "options": [
            {
                "action": "divine_cell",
                "index": index,
                "enabled": True,
            }
            for index in range(58)
        ]
        + [{"action": "proceed", "index": 58, "enabled": True}],
    }

    mask = adapter.compute_action_mask(state)

    assert mask.sum() == 1
    assert adapter.decode_action(mask.argmax(), state) == {
        "action": "choose",
        "index": 58,
    }


def test_extended_choices_cover_all_crystal_cells_and_rest_options():
    adapter = FullRunStateAdapter(extra_choice_slots=128)
    crystal = {
        "type": "crystal_sphere",
        "minigame": {"finished": False, "divinations_remaining": 6},
        "options": [
            {"action": "divine_cell", "index": index, "enabled": True}
            for index in range(121)
        ],
    }
    mask = adapter.compute_action_mask(crystal)
    assert mask.shape == (285,)
    assert mask.sum() == 121
    assert adapter.decode_action(157, crystal) == {"action": "choose", "index": 4}
    assert adapter.decode_action(273, crystal) == {"action": "choose", "index": 120}

    rest = {
        "type": "rest_site",
        "options": [{"index": index, "enabled": True} for index in range(7)],
    }
    rest_mask = adapter.compute_action_mask(rest)
    assert rest_mask.sum() == 7
    assert adapter.decode_action(158, rest) == {"action": "choose", "index": 6}
