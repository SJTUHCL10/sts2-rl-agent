"""Contract tests for the versioned entity/candidate agent interface."""

from __future__ import annotations

import copy
from pathlib import Path

import sts2_env.potions  # noqa: F401

from sts2_env.agent_v2.candidates import (
    build_action_candidates,
    resolve_candidate,
)
from sts2_env.agent_v2.schema import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    schema_manifest,
    validate_v2_envelope,
)
from sts2_env.agent_v2.snapshot import (
    attach_v2_envelope,
    build_combat_snapshot,
    build_run_snapshot,
    build_run_decision_snapshot,
)
from sts2_env.agent_v2.state_fields import (
    serialize_power_fields,
    serialize_relic_fields,
)
from sts2_env.bridge.entity_adapter import EntityStateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.potions.base import create_potion
from sts2_env.powers.remaining_c import SlowPower
from sts2_env.relics.registry import create_relic_by_name
from sts2_env.run.run_manager import RunManager


def _combat() -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=109,
        character_id="Ironclad",
    )
    enemy, ai = create_shrinker_beetle(Rng(109))
    combat.add_enemy(enemy, ai)
    combat.start_combat()
    combat.hand = [make_strike_ironclad(), make_defend_ironclad()]
    combat.energy = 2
    combat.potions = [create_potion("FirePotion", slot=0), None, None]
    return combat


def test_schema_manifest_is_deterministic_and_versioned() -> None:
    first = schema_manifest()
    second = schema_manifest()

    assert first == second
    assert first["protocol_version"] == PROTOCOL_VERSION == 2
    assert first["observation_schema"] == OBSERVATION_SCHEMA_VERSION
    assert first["action_schema"] == ACTION_SCHEMA_VERSION
    assert len(first["feature_layout_hash"]) == 64
    assert all(len(value) == 64 for value in first["vocabulary_hashes"].values())


def test_csharp_and_python_protocol_constants_are_locked_together() -> None:
    manifest = schema_manifest()
    source = (
        Path(__file__).parents[1] / "shared_mod" / "ProtocolV2.cs"
    ).read_text(encoding="utf-8")

    assert f"ProtocolVersion = {manifest['protocol_version']};" in source
    assert f'ObservationSchema = "{manifest["observation_schema"]}";' in source
    assert f'ActionSchema = "{manifest["action_schema"]}";' in source
    assert manifest["feature_layout_hash"] in source
    for field in (
        "power.Type",
        "power.StackType",
        "power.AmountOnTurnStart",
        "power.SkipNextDurationTick",
        "power.DisplayAmount",
        "relic.StackCount",
        "relic.Status",
        "relic.IsUsedUp",
        "relic.IsMelted",
        "relic.ShowCounter",
        "relic.DisplayAmount",
        "relic.FloorAddedToDeck",
    ):
        assert field in source


def test_envelope_validation_fails_fast_on_layout_drift() -> None:
    state = attach_v2_envelope({"type": "map_select"})
    validate_v2_envelope(state)

    bad = dict(state)
    bad["feature_layout_hash"] = "0" * 64
    try:
        validate_v2_envelope(bad)
    except ValueError as exc:
        assert "feature_layout_hash" in str(exc)
    else:
        raise AssertionError("layout drift must fail validation")


def test_combat_snapshot_contains_entities_all_piles_and_semantic_candidates() -> None:
    snapshot = build_combat_snapshot(_combat())

    validate_v2_envelope(snapshot)
    assert {card["zone"] for card in snapshot["cards"]} >= {"hand", "draw"}
    assert snapshot["discard_pile_count"] == 0
    assert snapshot["exhaust_pile_count"] == 0
    assert all(card["entity_id"].startswith("card:") for card in snapshot["cards"])
    assert snapshot["creatures"][0]["intents"]
    assert snapshot["potions"][0]["potion_id"] == "FirePotion"

    candidates = {item["candidate_id"]: item for item in snapshot["candidates"]}
    assert "combat:end_turn" in candidates
    strike_id = snapshot["hand"][0]["entity_id"]
    enemy_id = snapshot["enemies"][0]["entity_id"]
    play_id = f"combat:play:{strike_id}:{enemy_id}"
    assert candidates[play_id]["payload"] == {
        "action": "play",
        "card_index": 0,
        "target_index": 0,
    }


def test_combat_snapshot_can_defer_candidate_construction() -> None:
    snapshot = build_combat_snapshot(_combat(), include_candidates=False)

    validate_v2_envelope(snapshot)
    assert "candidates" not in snapshot
    assert snapshot["cards"]


def test_candidate_ids_are_semantic_not_policy_slot_numbers() -> None:
    state = {
        "type": "card_reward",
        "cards": [
            {"index": 8, "id": "BASH", "entity_id": "card:reward:bash"},
            {"index": 3, "id": "ANGER", "entity_id": "card:reward:anger"},
        ],
        "can_skip": True,
    }
    candidates = build_action_candidates(state)
    bash = next(item for item in candidates if item.source_id == "card:reward:bash")

    assert bash.candidate_id == "card_reward:pick_card:card:reward:bash"
    assert resolve_candidate(state, bash.candidate_id) == {
        "action": "choose",
        "index": 8,
    }
    reordered = copy.deepcopy(state)
    reordered["cards"].reverse()
    assert resolve_candidate(reordered, bash.candidate_id)["index"] == 8


def test_optional_card_selection_exposes_semantic_skip() -> None:
    state = {
        "type": "card_select",
        "cards": [{"index": 0, "id": "BASH"}],
        "min_select": 0,
        "max_select": 1,
    }

    candidates = build_action_candidates(state)

    skip = next(item for item in candidates if item.action_type == "SKIP")
    assert skip.candidate_id == "card_select:skip"
    assert resolve_candidate(state, skip.candidate_id) == {"action": "skip"}


def test_run_snapshot_contains_full_inventory_and_map_graph() -> None:
    manager = RunManager(seed=109, character_id="Ironclad")
    run = manager.run_state
    snapshot = build_run_snapshot(run)

    assert snapshot["character_id"] == "Ironclad"
    assert len(snapshot["cards"]) == len(run.player.deck)
    assert snapshot["relics"][0]["relic_id"] == run.player.relics[0]
    assert snapshot["map_nodes"]
    assert snapshot["map_edges"]
    assert any(node["reachable"] for node in snapshot["map_nodes"])


def test_run_snapshot_reuses_immutable_map_topology(monkeypatch) -> None:
    manager = RunManager(seed=110, character_id="Ironclad")
    run = manager.run_state
    first = build_run_snapshot(run)
    assert run.map is not None

    def fail_if_rebuilt():
        raise AssertionError("map topology should be cached")

    monkeypatch.setattr(run.map, "all_points", fail_if_rebuilt)
    second = build_run_snapshot(run)

    assert second["map_nodes"] == first["map_nodes"]
    assert second["map_edges"] == first["map_edges"]


def test_power_projection_keeps_amount_and_visible_counter_state() -> None:
    power = SlowPower(2)
    power._slow_amount = 3  # noqa: SLF001

    state = serialize_power_fields(power)

    assert state["power_type"] == "DEBUFF"
    assert state["stack_type"] == "COUNTER"
    assert state["amount"] == 2
    assert state["display_amount"] == 30
    assert state["counters"] == {
        "slow_amount": 3,
        "display_amount": 30,
    }


def test_relic_projection_keeps_persistent_and_display_counter_state() -> None:
    girya = create_relic_by_name("GIRYA")
    girya._times_lifted = 2  # noqa: SLF001
    nunchaku = create_relic_by_name("NUNCHAKU")
    nunchaku._attacks_played = 19  # noqa: SLF001

    girya_state = serialize_relic_fields(girya)
    nunchaku_state = serialize_relic_fields(nunchaku)

    assert girya_state["show_counter"] is True
    assert girya_state["display_amount"] == 2
    assert girya_state["counters"]["times_lifted"] == 2
    assert nunchaku_state["display_amount"] == 9
    assert nunchaku_state["counters"]["attacks_played"] == 19


def test_entity_adapter_merges_run_inventory_with_combat_entities() -> None:
    state = build_combat_snapshot(_combat())
    state["decision_id"] = "17"
    state["run_state"] = {
        "character_id": "Ironclad",
        "act": 1,
        "floor": 2,
        "players": [{
            "entity_id": "player:0",
            "character_id": "Ironclad",
            "gold": 120,
        }],
        "cards": [{
            "entity_id": "card:player:0:deck:99",
            "card_id": "BASH",
            "zone": "deck",
        }],
        "relics": [{"entity_id": "relic:0", "relic_id": "BURNING_BLOOD"}],
        "potions": [],
        "map_nodes": [{"entity_id": "map:0:0"}],
        "map_edges": [],
    }
    adapter = EntityStateAdapter()
    decision = adapter.decode(state)

    assert decision.decision_id == "17"
    assert any(card.get("card_id") == "BASH" for card in decision.cards)
    assert decision.relics[0]["relic_id"] == "BURNING_BLOOD"
    assert decision.candidates
    command = adapter.resolve(state, decision.candidates[0].candidate_id)
    assert command["candidate_id"] == decision.candidates[0].candidate_id
    assert command["decision_id"] == "17"


def test_run_decision_snapshot_keeps_v1_phase_and_adds_v2_candidates() -> None:
    manager = RunManager(seed=109, character_id="Ironclad")
    state = build_run_decision_snapshot(manager)

    validate_v2_envelope(state)
    assert state["type"] == "map_select"
    assert state["run_state"]["cards"]
    assert state["run_state"]["relics"]
    assert len(state["candidates"]) == len(state["nodes"])
    assert all(item["candidate_id"].startswith("map_select:") for item in state["candidates"])
