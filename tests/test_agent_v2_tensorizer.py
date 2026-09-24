from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

import sts2_env.agent_v2.tensorizer as tensorizer_module
from scripts.summarize_unknown_diagnostics import summarize
from sts2_env.agent_v2.candidates import build_action_candidates
from sts2_env.agent_v2.categorical_vocabulary import categorical_id
from sts2_env.agent_v2.snapshot import build_run_decision_snapshot
from sts2_env.agent_v2.tensorizer import (
    ENTITY_TYPE_TO_ID,
    LegacyV5TensorizerConfig,
    TensorizerConfig,
    observation_space,
    tensorize_snapshot,
)
from sts2_env.agent_v2.snapshot import serialize_card
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.agent_v2.unknown_diagnostics import UnknownTokenDiagnostics
from sts2_env.bridge.full_run_adapter import FullRunStateAdapter
from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE, POTION_ACTION_START
from sts2_env.events.act2 import CrystalSphere
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv
from sts2_env.run.run_manager import RunManager
from sts2_env.potions.base import create_potion


def test_card_affliction_and_enchantment_amount_reach_v6_tensor() -> None:
    card = create_ironclad_starter_deck()[0]
    card.afflict("bound")
    card.add_enchantment("Sharp", 3)
    serialized = serialize_card(
        card, zone="hand", zone_index=0, owner_id="player:0", playable=True,
    )
    assert serialized["afflictions"] == {"bound": 1}
    assert serialized["enchantments"] == {"Sharp": 3}

    config = TensorizerConfig(max_entities=8)
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0] = 1
    state = {"type": "combat_action", "phase": "COMBAT", "cards": [serialized]}
    observation = tensorize_snapshot(state, mask, config, validate=True)
    categorical = observation["entity_categorical"][0]
    numeric = observation["entity_numeric"][0]
    assert categorical[7] == categorical_id("BOUND", strict=True)
    assert categorical[9] == categorical_id("SHARP", strict=True)
    assert numeric[43] == pytest.approx(math.log1p(1) / 5)
    assert numeric[44] == pytest.approx(math.log1p(3) / 5)


def test_legacy_v5_trace_tensor_preserves_old_modifier_projection() -> None:
    config = LegacyV5TensorizerConfig(max_entities=8)
    assert LegacyV5TensorizerConfig().feature_layout_hash() == (
        "ec9db7fe8628c702f690df2449ba21d9553a7953f16350dce19f75f13d3b533e"
    )
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0] = 1
    state = {"type": "combat_action", "phase": "COMBAT", "cards": [{
        "entity_id": "card:0", "card_id": "STRIKE", "zone": "hand",
        "afflictions": {"BOUND": 1}, "enchantments": {"Sharp": 3},
    }]}
    observation = tensorize_snapshot(state, mask, config, validate=True)
    assert observation["entity_numeric"].shape == (8, 43)
    assert observation["entity_categorical"][0, 7] == 0
    assert observation["entity_categorical"][0, 9] == categorical_id("SHARP")


def test_current_intents_and_instance_pointers_are_distinct() -> None:
    config = TensorizerConfig(max_entities=16)
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "cards": [
            {"entity_id": f"card:hand:{i}", "card_id": "STRIKE", "zone": "hand", "zone_index": i}
            for i in range(2)
        ],
        "potions": [{"entity_id": "potion:0", "id": "FIRE_POTION", "slot": 0}],
        "creatures": [
            {"entity_id": f"enemy:{i}", "id": "CULTIST", "intents": [
                {"intent_type": "ATTACK", "damage": 7 + i, "hits": 1},
                {"intent_type": "BUFF", "damage": 0, "hits": 1},
            ]}
            for i in range(2)
        ],
        "candidates": [
            {
                "candidate_id": f"play:{hand}:{target}",
                "action_type": "PLAY_CARD",
                "source_id": f"card:hand:{hand}",
                "target_id": f"enemy:{target}",
                "payload": {"action": "play", "card_index": hand, "target_index": target},
            }
            for hand, target in ((0, 0), (0, 1), (1, 1))
        ] + [
            {
                "candidate_id": "play:0:none", "action_type": "PLAY_CARD",
                "source_id": "card:hand:0",
                "payload": {"action": "play", "card_index": 0, "target_index": -1},
            },
            {
                "candidate_id": "potion:0:none", "action_type": "USE_POTION",
                "source_id": "potion:0",
                "payload": {"action": "potion", "slot": 0, "target_index": -1},
            },
        ],
    }
    slots = [
        1 + MAX_HAND_SIZE + hand * MAX_ENEMIES + target
        for hand, target in ((0, 0), (0, 1), (1, 1))
    ]
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[slots] = 1
    mask[[1, POTION_ACTION_START]] = 1
    observation = tensorize_snapshot(snapshot, mask, config, validate=True)

    sources = observation["candidate_source_row"][slots]
    targets = observation["candidate_target_row"][slots]
    assert sources[0] == sources[1] != sources[2]
    assert targets[0] != targets[1] == targets[2]
    assert observation["entity_categorical"][targets[0], 13] == categorical_id("ATTACK")
    assert observation["entity_categorical"][targets[0], 14] == categorical_id("BUFF")
    assert observation["entity_numeric"][targets[0], 34] == pytest.approx(0.2)
    assert observation["entity_numeric"][targets[1], 35] == pytest.approx(0.08)
    assert observation["candidate_source_row"][1] == sources[0]
    assert observation["candidate_target_row"][1] == -1
    assert observation["candidate_source_row"][POTION_ACTION_START] >= 0
    assert observation["candidate_target_row"][POTION_ACTION_START] == -1


def test_live_bridge_hand_and_enemy_intent_become_pointed_entities() -> None:
    config = TensorizerConfig(max_entities=16)
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "hand": [{"id": "STRIKE", "playable": True, "target": "AnyEnemy"}],
        "enemies": [{
            "id": "CULTIST", "is_alive": True,
            "intent": "Attack", "intent_damage": 9, "intent_hits": 2,
        }],
    }
    snapshot["candidates"] = [
        candidate.to_dict() for candidate in build_action_candidates(snapshot)
    ]
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[1 + MAX_HAND_SIZE] = 1
    observation = tensorize_snapshot(snapshot, mask, config, validate=True)
    source = observation["candidate_source_row"][1 + MAX_HAND_SIZE]
    target = observation["candidate_target_row"][1 + MAX_HAND_SIZE]
    assert source >= 0 and target >= 0
    assert observation["entity_categorical"][source, 0] == ENTITY_TYPE_TO_ID["CARD"]
    assert observation["entity_categorical"][target, 0] == ENTITY_TYPE_TO_ID["CREATURE"]
    assert observation["entity_numeric"][target, 35] == pytest.approx(0.09)
    assert observation["entity_numeric"][target, 36] == pytest.approx(0.2)


def test_fourth_reward_card_points_to_its_own_entity() -> None:
    config = TensorizerConfig(max_entities=16)
    snapshot = {
        "type": "card_reward",
        "cards": [{"index": index, "id": "STRIKE"} for index in range(4)],
        "can_skip": True,
    }
    snapshot["candidates"] = [
        candidate.to_dict() for candidate in build_action_candidates(snapshot)
    ]
    adapter = FullRunStateAdapter(extra_choice_slots=128)
    mask = adapter.compute_action_mask(snapshot)
    observation = tensorize_snapshot(snapshot, mask, config, validate=True)
    assert mask[124]
    rows = observation["candidate_source_row"][[120, 121, 122, 124]]
    assert len(set(rows.tolist())) == 4
    assert min(rows) >= 0
    assert adapter.decode_action(124, snapshot) == {"action": "choose", "index": 3}


@pytest.mark.parametrize("potion_id", ["FairyInABottle", "ShipInABottle"])
def test_bottle_potion_entity_and_action_use_known_content(
    potion_id: str,
    tmp_path,
) -> None:
    config = TensorizerConfig(max_entities=8)
    diagnostics = UnknownTokenDiagnostics(tmp_path / "worker_0.jsonl", worker_index=0)
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "potions": [{"entity_id": "potion:0", "potion_id": potion_id, "slot": 0}],
        "candidates": [{
            "candidate_id": "potion:0:none",
            "action_type": "USE_POTION",
            "source_id": "potion:0",
            "payload": {"action": "potion", "slot": 0, "target_index": -1},
        }],
    }
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[POTION_ACTION_START] = 1
    observation = tensorize_snapshot(
        snapshot, mask, config, validate=True, unknown_diagnostics=diagnostics,
    )
    diagnostics.flush()
    expected = categorical_id(potion_id, strict=True)
    assert observation["entity_categorical"][0, 1] == expected
    assert observation["candidate_categorical"][POTION_ACTION_START, 2] == expected
    assert not diagnostics.output_path.exists()


def test_unknown_log_names_field_token_and_first_context(tmp_path) -> None:
    config = TensorizerConfig(max_entities=8)
    log_path = tmp_path / "worker_0.jsonl"
    diagnostics = UnknownTokenDiagnostics(log_path, worker_index=0)
    snapshot = {
        "type": "combat_action", "phase": "COMBAT",
        "creatures": [{
            "entity_id": "enemy:42", "id": "NEW_MONSTER_NOT_IN_VOCAB",
            "intents": [{"intent_type": "NEW_INTENT_NOT_IN_VOCAB"}],
        }],
    }
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0] = 1
    tensorize_snapshot(snapshot, mask, config, unknown_diagnostics=diagnostics)
    diagnostics.flush()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    by_field = {record["field"]: record for record in records}
    assert by_field["entity.content"]["token"] == "NEW_MONSTER_NOT_IN_VOCAB"
    assert by_field["entity.content"]["sample"]["entity_id"] == "enemy:42"
    assert by_field["entity.intent_0"]["token"] == "NEW_INTENT_NOT_IN_VOCAB"
    assert all(record["worker_index"] == 0 for record in records)
    assert "candidate.action_slot" not in by_field
    summary = summarize(tmp_path)
    assert {item["field"] for item in summary} == {
        "entity.content", "entity.intent_0",
    }


def test_crystal_sphere_extended_choice_points_to_cell() -> None:
    env = STS2EntityRunEnv(max_steps=20)
    env.reset(seed=843)
    manager = env.run_env._mgr
    assert manager is not None
    manager._phase = RunManager.PHASE_EVENT
    crystal = CrystalSphere()
    manager._event_model = crystal
    manager._event_options = crystal.generate_initial_options(manager.run_state)
    manager._do_event_choice({"option_id": "debt"})
    env.invalidate_action_mask_cache()
    mask = env.action_masks()
    assert len(manager.get_available_actions()) > 4
    assert mask[157]
    snapshot = env.run_env.entity_observation()
    chosen_id = snapshot["options"][4]["entity_id"]
    fifth_cell = next(cell for cell in snapshot["crystal_cells"] if cell["entity_id"] == chosen_id)
    observation = env._structured_observation(mask)
    row = observation["candidate_source_row"][157]
    assert row >= 0
    assert observation["entity_categorical"][row, 0] == ENTITY_TYPE_TO_ID["CRYSTAL_CELL"]
    assert chosen_id == fifth_cell["entity_id"]
    assert observation["entity_numeric"][row, 15] == pytest.approx(fifth_cell["y"] / 20)
    assert observation["entity_numeric"][row, 16] == pytest.approx(fifth_cell["x"] / 10)
    remaining = crystal.minigame.divination_count
    env.step(157)
    assert crystal.minigame.divination_count < remaining


@pytest.mark.parametrize(
    ("reward_type", "item_id", "pick_action", "skip_action"),
    [
        ("potion", "FruitJuice", "pick_potion", "skip_potion"),
        ("relic", "FESTIVE_POPPER", "pick_relic_reward", "skip_relic"),
    ],
)
def test_post_combat_item_reward_has_semantic_candidate_and_source_pointer(
    reward_type: str, item_id: str, pick_action: str, skip_action: str,
) -> None:
    env = STS2EntityRunEnv(max_steps=20)
    env.reset(seed=109)
    manager = env.run_env._mgr
    assert manager is not None
    manager._phase = RunManager.PHASE_CARD_REWARD
    if reward_type == "potion":
        manager._offered_potion = create_potion(item_id)
    else:
        manager._offered_relic = item_id
    env.invalidate_action_mask_cache()
    mask = env.action_masks()
    snapshot = env.run_env.entity_observation()
    aligned = tensorizer_module._candidate_slots(snapshot, mask)
    assert aligned[120]["payload"]["action"] == pick_action
    assert aligned[123]["payload"]["action"] == skip_action
    assert snapshot["options"][0]["id"] == item_id
    observation = env._structured_observation(mask)
    source_row = observation["candidate_source_row"][120]
    assert source_row >= 0
    assert observation["entity_categorical"][source_row, 0] == ENTITY_TYPE_TO_ID["CHOICE"]
    legacy = LegacyV5TensorizerConfig()
    old = tensorize_snapshot(snapshot, mask, legacy)
    assert old["candidate_source_row"][120] == -1
    assert old["candidate_source_row"][123] == -1
    original = {key: value for key, value in snapshot.items() if key not in {
        "reward_item_type", "options", "candidates",
    }}
    original["candidates"] = []
    prior_projection = tensorize_snapshot(original, mask, legacy)
    for key in old:
        np.testing.assert_array_equal(old[key], prior_projection[key])
    env.close()


def test_event_identity_and_option_ids_reach_current_tensor_only() -> None:
    env = STS2EntityRunEnv(max_steps=20)
    env.reset(seed=100109)
    manager = env.run_env._mgr
    assert manager is not None
    manager._enter_event()
    assert manager._event_model is not None
    env.invalidate_action_mask_cache()
    mask = env.action_masks()
    snapshot = env.run_env.entity_observation()
    assert snapshot["event_id"] == manager._event_model.event_id
    assert snapshot["options"][0]["id"] == manager.get_available_actions()[0]["option_id"]
    assert categorical_id(snapshot["options"][0]["model_content"]) != 1
    observation = env._structured_observation(mask)
    assert observation["candidate_source_row"][145] >= 0
    assert observation["candidate_categorical"][145, 2] != 1
    assert any(
        observation["entity_categorical"][row, 1] == categorical_id(snapshot["event_id"])
        for row in range(int(observation["entity_mask"].sum()))
    )
    legacy = LegacyV5TensorizerConfig()
    old = tensorize_snapshot(snapshot, mask, legacy)
    original = {
        **snapshot,
        "options": snapshot["legacy_event_options"],
        "candidates": snapshot["legacy_event_candidates"],
        "event_context": [],
    }
    prior_projection = tensorize_snapshot(original, mask, legacy)
    for key in old:
        np.testing.assert_array_equal(old[key], prior_projection[key])
    env.close()


def test_live_event_option_event_id_becomes_context_entity() -> None:
    config = TensorizerConfig(max_entities=8)
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[145] = 1
    snapshot = {
        "type": "event",
        "options": [{
            "index": 0, "id": "event_choice", "action": "event_choice",
            "event_id": "AbyssalBaths", "label": "Linger", "enabled": True,
        }],
    }
    current = tensorize_snapshot(snapshot, mask, config)
    assert any(
        current["entity_categorical"][row, 1] == categorical_id("AbyssalBaths")
        for row in range(int(current["entity_mask"].sum()))
    )


@pytest.mark.parametrize("state, slot, expected_index", [
    ({"type": "reward_screen", "options": [
        {"index": 9, "action": "proceed", "id": "proceed"},
        {"index": 2, "action": "pick_reward", "id": "reward"},
    ]}, 123, 9),
    ({"type": "shop", "options": [
        {"index": 4, "action": "buy_card", "id": "STRIKE"},
        {"index": 8, "action": "leave_shop", "id": "leave"},
    ]}, 130, 8),
    ({"type": "crystal_sphere", "minigame": {"finished": True}, "options": [
        {"index": 0, "action": "divine_cell", "entity_id": "crystal-cell:0:0"},
        {"index": 1, "action": "proceed", "entity_id": "option:proceed:1"},
    ]}, 145, 1),
])
def test_special_screen_candidate_pointer_matches_decoded_choice(state, slot, expected_index):
    config = TensorizerConfig(max_entities=16)
    state = copy.deepcopy(state)
    state["candidates"] = [candidate.to_dict() for candidate in build_action_candidates(state)]
    adapter = FullRunStateAdapter(extra_choice_slots=128)
    mask = adapter.compute_action_mask(state)
    observation = tensorize_snapshot(state, mask, config, validate=True)
    command = adapter.decode_action(slot, state)
    assert command == {"action": "choose", "index": expected_index}
    row = observation["candidate_source_row"][slot]
    assert row >= 0
    assert observation["candidate_numeric"][slot, 5] == pytest.approx(expected_index / 256)


def test_entity_run_env_observation_matches_declared_space() -> None:
    config = TensorizerConfig(max_entities=128)
    env = STS2EntityRunEnv(
        tensorizer_config=config,
        max_steps=20,
    )
    observation, info = env.reset(seed=109)

    assert env.observation_space.contains(observation)
    assert observation["candidate_categorical"].shape == (config.num_actions, 7)
    assert observation["candidate_numeric"].shape == (config.num_actions, 15)
    assert observation["entity_mask"].sum() > 0
    valid = np.flatnonzero(info["action_mask"])
    next_observation, _, _, _, _ = env.step(int(valid[0]))
    assert env.observation_space.contains(next_observation)


def test_entity_run_env_reuses_info_action_mask(monkeypatch) -> None:
    env = STS2EntityRunEnv(max_steps=20)
    original = env.run_env.action_masks
    calls = 0

    def counted_action_masks() -> np.ndarray:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(env.run_env, "action_masks", counted_action_masks)
    _, info = env.reset(seed=109)

    assert calls == 1
    assert env.action_masks() is info["action_mask"]
    env._structured_observation()
    assert calls == 1

    env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert calls == 2
    env.action_masks()
    assert calls == 2

    env.invalidate_action_mask_cache()
    env.action_masks()
    assert calls == 3


def test_entity_run_env_skips_discarded_legacy_observation(monkeypatch) -> None:
    env = STS2EntityRunEnv(max_steps=20)

    def fail_if_called() -> np.ndarray:
        raise AssertionError("entity wrapper should not encode the v1 vector")

    monkeypatch.setattr(env.run_env, "_encode_obs", fail_if_called)
    observation, info = env.reset(seed=109)
    env.step(int(np.flatnonzero(info["action_mask"])[0]))

    assert env.observation_space.contains(observation)


def test_entity_run_env_returns_structured_terminal_observation() -> None:
    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    env.reset(seed=109)
    assert env.run_env._mgr is not None
    env.run_env._mgr.run_state.lose_run()
    env.invalidate_action_mask_cache()

    observation = env._structured_observation()

    assert env.observation_space.contains(observation)
    assert env.run_env.entity_observation()["phase"] == "RUN_OVER"


def test_identical_deck_cards_are_count_compressed() -> None:
    config = TensorizerConfig(max_entities=32)
    card = {
        "entity_id": "card:player:0:template",
        "card_id": "CLONE",
        "owner_id": "player:0",
        "zone": "deck",
        "zone_index": 0,
        "card_type": "SKILL",
        "upgrade_level": 0,
        "afflictions": {},
        "enchantments": {},
    }
    cards = []
    for index in range(10_000):
        duplicate = dict(card)
        duplicate["entity_id"] = f"card:player:0:{index}"
        duplicate["zone_index"] = index
        cards.append(duplicate)
    snapshot = {
        "type": "map_select",
        "phase": "MAP_CHOICE",
        "run_state": {
            "character_id": "Ironclad",
            "players": [{"entity_id": "player:0", "hp": 80, "max_hp": 80}],
            "cards": cards,
        },
        "candidates": [],
    }
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[115] = 1

    observation = tensorize_snapshot(snapshot, mask, config, validate=True)
    card_rows = (
        observation["entity_categorical"][:, 0]
        == ENTITY_TYPE_TO_ID["CARD"]
    )

    assert card_rows.sum() == 1
    count_feature = observation["entity_numeric"][card_rows, 12]
    assert count_feature.item() > 1.0
    assert observation_space(config).contains(observation)


def test_full_observation_validation_is_opt_in(monkeypatch) -> None:
    config = TensorizerConfig(max_entities=8)
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0] = 1
    snapshot = {
        "type": "run_complete",
        "phase": "RUN_OVER",
        "candidates": [],
    }

    class RejectingSpace:
        def contains(self, _value) -> bool:
            return False

    monkeypatch.setattr(
        tensorizer_module,
        "observation_space",
        lambda _config: RejectingSpace(),
    )

    tensorize_snapshot(snapshot, mask, config)
    with pytest.raises(ValueError, match="violates its Gymnasium space"):
        tensorize_snapshot(snapshot, mask, config, validate=True)


def test_combat_piles_compress_but_hand_instances_do_not() -> None:
    config = TensorizerConfig(max_entities=32)
    cards = [
        {
            "entity_id": f"card:player:0:draw:{index}",
            "card_id": "CLONE",
            "owner_id": "player:0",
            "zone": "draw",
            "zone_index": index,
            "combat_vars": {},
        }
        for index in range(1000)
    ]
    cards.extend([
        {
            "entity_id": f"card:player:0:hand:{index}",
            "card_id": "CLONE",
            "owner_id": "player:0",
            "zone": "hand",
            "zone_index": index,
            "combat_vars": {},
        }
        for index in range(2)
    ])
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "cards": cards,
        "candidates": [],
    }
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0] = 1

    observation = tensorize_snapshot(snapshot, mask, config)
    card_rows = (
        observation["entity_categorical"][:, 0]
        == ENTITY_TYPE_TO_ID["CARD"]
    )

    assert card_rows.sum() == 3
    counts = observation["entity_numeric"][card_rows, 12]
    assert sum(count > 1.0 for count in counts) == 1


def test_candidate_uses_stable_content_not_runtime_instance_id() -> None:
    config = TensorizerConfig(max_entities=16)
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[1] = 1

    def make_snapshot(instance: str) -> dict:
        entity_id = f"card:player:0:{instance}"
        return {
            "type": "combat_action",
            "phase": "COMBAT",
            "cards": [{
                "entity_id": entity_id,
                "card_id": "STRIKE",
                "owner_id": "player:0",
                "zone": "hand",
                "zone_index": 0,
            }],
            "candidates": [{
                "candidate_id": f"combat:play:{entity_id}:none",
                "action_type": "PLAY_CARD",
                "source_id": entity_id,
                "enabled": True,
                "payload": {
                    "action": "play",
                    "card_index": 0,
                    "target_index": -1,
                },
            }],
        }

    first = tensorize_snapshot(make_snapshot("100"), mask, config)
    second = tensorize_snapshot(make_snapshot("999999"), mask, config)

    np.testing.assert_array_equal(
        first["candidate_categorical"][1],
        second["candidate_categorical"][1],
    )


def test_card_selection_state_changes_candidate_features() -> None:
    config = TensorizerConfig(max_entities=16)
    mask = np.zeros(config.num_actions, dtype=np.int8)
    mask[0:3] = 1
    base = {
        "type": "card_select",
        "phase": "CARD_REWARD",
        "min_select": 1,
        "max_select": 2,
        "selected_count": 0,
        "can_confirm": False,
        "cards": [
            {"index": 0, "id": "STRIKE", "type": "Attack", "selected": False},
            {"index": 1, "id": "DEFEND", "type": "Skill", "selected": False},
        ],
    }
    from sts2_env.agent_v2.candidates import build_action_candidates

    base["candidates"] = [
        item.to_dict() for item in build_action_candidates(base)
    ]
    unselected = tensorize_snapshot(base, mask, config)

    selected_state = copy.deepcopy(base)
    selected_state["cards"][0]["selected"] = True
    selected_state["selected_count"] = 1
    selected_state["can_confirm"] = True
    selected_state.pop("candidates")
    selected_state["candidates"] = [
        item.to_dict() for item in build_action_candidates(selected_state)
    ]
    selected = tensorize_snapshot(selected_state, mask, config)

    assert unselected["candidate_numeric"][1, 10] == 0.0
    assert selected["candidate_numeric"][1, 10] == 1.0
    assert selected["candidate_numeric"][0, 14] == 1.0

    from sts2_env.agent_v2.categorical_vocabulary import categorical_id

    # The candidate itself is linked to STRIKE, rather than relying on the
    # arbitrary action-slot embedding to identify which card it selects.
    assert selected["candidate_categorical"][1, 2] == categorical_id(
        "STRIKE", strict=True
    )
    assert selected["candidate_categorical"][1, 5] == categorical_id(
        "STRIKE", strict=True
    )


def test_run_snapshot_preserves_incremental_deck_selection() -> None:
    manager = RunManager(seed=803, character_id="Ironclad")
    assert manager.run_state.player.obtain_relic("KIFUDA")

    before = build_run_decision_snapshot(manager)
    assert before["type"] == "card_select"
    assert before["selected_count"] == 0
    assert before["can_confirm"] is True
    assert not any(card["selected"] for card in before["cards"])

    manager.take_action({"action": "choose", "index": 0})
    after = build_run_decision_snapshot(manager)

    assert after["selected_count"] == 1
    assert after["cards"][0]["selected"] is True
    selected_candidate = next(
        candidate for candidate in after["candidates"]
        if candidate["payload"] == {"action": "choose", "index": 0}
    )
    assert selected_candidate["features"]["selected"] is True


def test_typed_set_encoder_is_entity_permutation_invariant() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetTransformerConfig,
        TypedSetTransformerExtractor,
    )

    config = TensorizerConfig(max_entities=96)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    observation, _ = env.reset(seed=9)
    permuted = copy.deepcopy(observation)
    permutation = np.random.default_rng(1).permutation(config.max_entities)
    for key in ("entity_categorical", "entity_numeric", "entity_mask"):
        permuted[key] = permuted[key][permutation]
    old_to_new = np.argsort(permutation)
    for key in ("candidate_source_row", "candidate_target_row"):
        rows = permuted[key]
        permuted[key] = np.where(rows >= 0, old_to_new[rows.clip(min=0)], -1)

    extractor = TypedSetTransformerExtractor(
        env.observation_space,
        config=TypedSetTransformerConfig(
            d_model=32,
            num_heads=4,
            num_inducing_points=8,
            num_memory_tokens=4,
            num_isab_layers=1,
        ),
        categorical_vocab_size=config.categorical_vocab_size,
    )
    extractor.eval()

    def tensors(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in obs.items()
        }

    with torch.no_grad():
        original_features = extractor(tensors(observation))
        permuted_features = extractor(tensors(permuted))
    torch.testing.assert_close(
        original_features,
        permuted_features,
        atol=1e-5,
        rtol=1e-5,
    )


def test_maskable_policy_forward_shapes() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetMaskableActorCriticPolicy,
        TypedSetTransformerConfig,
    )

    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    observation, _ = env.reset(seed=5)
    policy = TypedSetMaskableActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        typed_set_config=TypedSetTransformerConfig(
            d_model=32,
            num_heads=4,
            num_inducing_points=8,
            num_memory_tokens=4,
            num_isab_layers=1,
        ),
        categorical_vocab_size=config.categorical_vocab_size,
        categorical_vocabulary_hash=config.categorical_vocabulary_hash,
        ortho_init=False,
    )
    batched = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }
    actions, values, log_prob = policy(
        batched,
        action_masks=env.action_masks()[None, :],
    )

    assert actions.shape == (1,)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)
    assert env.action_masks()[int(actions.item())] == 1


def test_maskable_ppo_learns_saves_and_loads(tmp_path) -> None:
    pytest.importorskip("torch")
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetMaskableActorCriticPolicy,
        TypedSetTransformerConfig,
    )

    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    model = sb3_contrib.MaskablePPO(
        TypedSetMaskableActorCriticPolicy,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "typed_set_config": TypedSetTransformerConfig(
                d_model=16,
                num_heads=4,
                num_inducing_points=4,
                num_memory_tokens=2,
                num_isab_layers=1,
            ),
            "categorical_vocab_size": config.categorical_vocab_size,
            "categorical_vocabulary_hash": config.categorical_vocabulary_hash,
            "tensorizer_layout_hash": config.feature_layout_hash(),
            "ortho_init": False,
        },
        seed=1,
        verbose=0,
    )

    model.learn(total_timesteps=8)
    model_path = tmp_path / "typed_set_model"
    model.save(model_path)
    loaded = sb3_contrib.MaskablePPO.load(model_path, env=env)
    observation, _ = env.reset(seed=2)
    action, _ = loaded.predict(
        observation,
        action_masks=env.action_masks(),
        deterministic=True,
    )

    assert env.action_masks()[int(action)] == 1
