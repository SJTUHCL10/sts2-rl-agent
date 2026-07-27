"""Tests for bridge agent non-combat choices."""

from sts2_env.bridge.agent_runner import (
    TERMINAL_PHASES,
    _phase_for_state,
    _pick_boss_relic_option,
    _pick_card_bundle_index,
    _pick_card_reward_index,
    _pick_card_select_indexes,
    _pick_crystal_sphere_option,
    _pick_map_node,
    _pick_reward_screen_option,
    _pick_rest_option,
    _pick_shop_option,
    _pick_treasure_option,
    _replay_mode_for_policy,
)
from sts2_env.bridge.protocol import BridgeStateType


def test_phase_mapping_treats_run_complete_as_terminal() -> None:
    phase = _phase_for_state({"type": BridgeStateType.RUN_COMPLETE, "result": "victory"})

    assert phase == BridgeStateType.RUN_COMPLETE
    assert phase in TERMINAL_PHASES


def test_replay_mode_matches_detected_policy_interface() -> None:
    assert _replay_mode_for_policy(True) == "full_run"
    assert _replay_mode_for_policy(False) == "combat"


def test_map_policy_prefers_rest_when_hp_is_low() -> None:
    state = {
        "player": {"hp": 20, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Elite"},
            {"index": 1, "type": "RestSite"},
            {"index": 2, "type": "Monster"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_prefers_elite_when_hp_is_healthy() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "RestSite"},
            {"index": 1, "type": "Monster"},
            {"index": 2, "type": "Elite"},
        ],
    }

    assert _pick_map_node(state) == 2


def test_card_reward_policy_prefers_power_and_skips_large_decks() -> None:
    state = {
        "cards": [
            {"index": 0, "id": "STRIKE_IRONCLAD", "type": "Attack"},
            {"index": 1, "id": "INFLAME", "type": "Power"},
        ],
        "can_skip": True,
        "run_state": {"deck": ["card"] * 10},
    }

    assert _pick_card_reward_index(state) == 1

    state["run_state"] = {"deck": ["card"] * 31}
    assert _pick_card_reward_index(state) is None


def test_card_select_policy_uses_required_card_count() -> None:
    state = {
        "cards": [{"index": 3}, {"index": 5}, {"index": 8}],
        "min_select": 2,
        "max_select": 3,
    }

    assert _pick_card_select_indexes(state) == [3, 5]


def test_reward_screen_policy_picks_rewards_before_proceeding() -> None:
    state = {
        "options": [
            {"index": 0, "action": "proceed", "enabled": True},
            {"index": 1, "action": "pick_reward", "enabled": True},
        ],
    }

    assert _pick_reward_screen_option(state) == 1


def test_card_bundle_policy_picks_enabled_bundle_by_action() -> None:
    state = {
        "bundles": [
            {"index": 0, "action": "inspect", "enabled": True},
            {"index": 3, "action": "pick_card_bundle", "enabled": True},
        ],
    }

    assert _pick_card_bundle_index(state) == 3


def test_crystal_sphere_policy_clicks_cells_before_proceeding() -> None:
    state = {
        "options": [
            {"index": 0, "action": "proceed", "enabled": True},
            {"index": 7, "action": "divine_cell", "x": 5, "y": 6, "enabled": True},
        ],
    }

    assert _pick_crystal_sphere_option(state) == 7


def test_rest_policy_uses_option_ids_not_order() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "options": [
            {"index": 0, "id": "HEAL", "enabled": True},
            {"index": 1, "id": "SMITH", "enabled": True},
        ],
    }

    assert _pick_rest_option(state) == 1

    state["player"] = {"hp": 20, "max_hp": 80}
    assert _pick_rest_option(state) == 0


def test_shop_policy_buys_before_leaving() -> None:
    state = {
        "options": [
            {"index": 0, "action": "leave_shop", "enabled": True},
            {"index": 1, "action": "buy_card", "enabled": True},
            {"index": 2, "action": "buy_relic", "enabled": True},
        ],
    }

    assert _pick_shop_option(state) == 2


def test_treasure_and_boss_relic_policy_use_action_labels() -> None:
    treasure = {
        "options": [
            {"index": 4, "action": "collect", "enabled": True},
        ],
    }
    boss_relic = {
        "options": [
            {"index": 2, "action": "pick_relic", "enabled": True},
        ],
    }

    assert _pick_treasure_option(treasure) == 4
    assert _pick_boss_relic_option(boss_relic) == 2
