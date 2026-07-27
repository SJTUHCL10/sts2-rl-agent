"""Tests for the bridge replay golden-comparison harness."""

from __future__ import annotations

from pathlib import Path

import sts2_env.powers  # noqa: F401
import sts2_env.potions  # noqa: F401

from sts2_env.bridge.protocol import BridgeAction, BridgeStateType
from sts2_env.cards.ironclad_basic import (
    create_ironclad_starter_deck,
    make_bash,
    make_defend_ironclad,
    make_strike_ironclad,
)
from sts2_env.cards.silent import create_silent_starter_deck, make_strike_silent, make_survivor
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.events.act1 import TheLegendsWereTrue
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.parity.bridge_replay import (
    BridgeReplayRecorder,
    BridgeReplayStep,
    BridgeReplayTrace,
    combat_state_to_bridge_state,
    compare_combat_replay,
    compare_run_replay,
    load_replay_trace,
    normalize_bridge_state,
    run_manager_to_bridge_state,
    save_replay_trace,
)
from sts2_env.parity.bridge_replay_cli import build_parser
from sts2_env.potions.base import create_potion
from sts2_env.run.reward_objects import RelicReward
from sts2_env.run.reward_objects import CardBundlesReward
from sts2_env.run.run_manager import RunManager

BRIDGE_REPLAY_SEED = 42
BRIDGE_REPLAY_LOW_HP = 20
FIRST_BRIDGE_CHOICE_INDEX = 0


def make_basic_replay_combat() -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=BRIDGE_REPLAY_SEED,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(BRIDGE_REPLAY_SEED))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    combat.hand = [make_strike_ironclad()]
    combat.energy = 1
    return combat


def make_choice_replay_combat() -> CombatState:
    combat = CombatState(
        player_hp=70,
        player_max_hp=70,
        deck=create_silent_starter_deck(),
        rng_seed=BRIDGE_REPLAY_SEED,
        character_id="Silent",
    )
    creature, ai = create_shrinker_beetle(Rng(BRIDGE_REPLAY_SEED))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    combat.hand = [make_survivor(), make_strike_silent(), make_strike_silent()]
    combat.energy = 1
    return combat


def make_basic_replay_combat_with_block_potion() -> CombatState:
    combat = make_basic_replay_combat()
    combat.potions = [create_potion("BlockPotion"), None, None]
    return combat


def make_basic_replay_run() -> RunManager:
    return RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")


def make_card_reward_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run._enter_card_reward(context="regular")
    return run


def make_card_bundle_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run._phase = RunManager.PHASE_CARD_REWARD
    run._current_reward = CardBundlesReward(
        run.run_state.player.player_id,
        [
            [make_strike_ironclad(), make_defend_ironclad(), make_bash()],
            [make_defend_ironclad(), make_strike_ironclad(), make_bash()],
        ],
    )
    run._prime_next_card_bundles_reward()
    return run


def make_rest_site_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run.run_state.player.current_hp = BRIDGE_REPLAY_LOW_HP
    run._enter_rest_site()
    return run


def make_shop_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run._enter_shop()
    return run


def make_event_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    event = TheLegendsWereTrue()
    run._phase = RunManager.PHASE_EVENT
    run._event_model = event
    run._event_started = True
    run._event_options = event.generate_initial_options(run.run_state)
    return run


def make_treasure_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run._phase = RunManager.PHASE_TREASURE
    run._current_reward = RelicReward(run.run_state.player.player_id, relic_id="LANTERN")
    return run


def make_boss_relic_replay_run() -> RunManager:
    run = RunManager(seed=BRIDGE_REPLAY_SEED, character_id="Ironclad")
    run._phase = RunManager.PHASE_BOSS_RELIC
    run._boss_relics = ["BLACK_STAR", "SOZU", "BEAUTIFUL_BRACELET"]
    return run


class FakeBridgeClient:
    def __init__(self, states: list[dict]):
        self._states = list(states)
        self.sent_actions: list[dict] = []
        self.connected = True

    def receive_state(self) -> dict:
        if not self._states:
            raise RuntimeError("No more states queued")
        return self._states.pop(0)

    def send_action(self, action: dict) -> None:
        self.sent_actions.append(dict(action))

    def ping(self) -> bool:
        return True


def test_bridge_replay_trace_round_trip(tmp_path: Path):
    combat = make_basic_replay_combat()
    initial_state = combat_state_to_bridge_state(combat)
    assert combat.play_card(0, 0)
    next_state = combat_state_to_bridge_state(combat)
    trace = BridgeReplayTrace(
        metadata={"scenario_factory": "tests.test_bridge_replay_harness:make_basic_replay_combat"},
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.PLAY, "card_index": 0, "target_index": 0},
                resulting_state=next_state,
            )
        ],
    )

    path = save_replay_trace(trace, tmp_path / "basic_trace.json")
    loaded = load_replay_trace(path)

    assert loaded.metadata["scenario_factory"] == "tests.test_bridge_replay_harness:make_basic_replay_combat"
    assert loaded.initial_state == trace.initial_state
    assert loaded.steps[0].action == trace.steps[0].action
    assert loaded.steps[0].resulting_state == trace.steps[0].resulting_state


def test_bridge_replay_recorder_records_state_action_state_sequence():
    initial = {"type": BridgeStateType.COMBAT_ACTION, "player": {"hp": 80, "max_hp": 80, "block": 0, "energy": 1, "max_energy": 3, "powers": []}, "hand": [], "enemies": [], "draw_pile_count": 0, "discard_pile_count": 0, "exhaust_pile_count": 0, "round": 1}
    next_state = {"type": BridgeStateType.COMBAT_ACTION, "player": {"hp": 80, "max_hp": 80, "block": 0, "energy": 0, "max_energy": 3, "powers": []}, "hand": [], "enemies": [], "draw_pile_count": 0, "discard_pile_count": 1, "exhaust_pile_count": 0, "round": 1}
    client = FakeBridgeClient([initial, next_state])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.play_card(0, 0)
    assert recorder.receive_state() == next_state

    assert client.sent_actions == [{"action": BridgeAction.PLAY, "card_index": 0, "target_index": 0}]
    assert recorder.trace.initial_state["type"] == BridgeStateType.COMBAT_ACTION
    assert recorder.trace.steps[0].action == client.sent_actions[0]
    assert recorder.trace.steps[0].resulting_state["player"]["energy"] == 0


def test_bridge_replay_recorder_preserves_explicit_full_run_mode():
    recorder = BridgeReplayRecorder(FakeBridgeClient([]), mode="full_run")

    assert recorder.trace.mode == "full_run"


def test_bridge_replay_recorder_preserves_potion_slots_and_mask_metadata():
    base = {
        "type": BridgeStateType.COMBAT_ACTION,
        "player": {"hp": 80, "max_hp": 80, "block": 0, "energy": 1, "max_energy": 3, "powers": []},
        "hand": [],
        "enemies": [],
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "round": 1,
        "max_potion_slots": 3,
    }
    initial = {
        **base,
        "potions": [{
            "slot": 2,
            "id": "BufferPotion",
            "usage": "CombatOnly",
            "can_use": True,
            "target": "Self",
            "requires_target": False,
            "target_type": "Self",
        }],
    }
    after = {**base, "potions": []}
    client = FakeBridgeClient([initial, after])
    recorder = BridgeReplayRecorder(client)

    recorder.receive_state()
    recorder.use_potion(2)
    recorder.receive_state()

    assert recorder.trace.initial_state["potions"] == initial["potions"]
    assert recorder.trace.initial_state["max_potion_slots"] == 3
    assert recorder.trace.steps[0].resulting_state["potions"] == []


def test_normalize_bridge_state_canonicalizes_python_potion_enum_names():
    state = normalize_bridge_state({
        "type": BridgeStateType.COMBAT_ACTION,
        "player": {},
        "potions": [{
            "slot": 0,
            "id": "FirePotion",
            "usage": "COMBAT_ONLY",
            "target": "ANY_ENEMY",
            "target_type": "ANY_ENEMY",
            "requires_target": True,
        }],
        "max_potion_slots": 3,
    })

    assert state["potions"][0]["usage"] == "CombatOnly"
    assert state["potions"][0]["target"] == "AnyEnemy"
    assert state["potions"][0]["target_type"] == "AnyEnemy"


def test_bridge_replay_recorder_records_skip_and_multi_select_actions():
    initial = {
        "type": BridgeStateType.CARD_SELECT,
        "cards": [{"index": 0, "id": "STRIKE_IRONCLAD"}, {"index": 1, "id": "DEFEND_IRONCLAD"}],
        "min_select": 0,
        "max_select": 2,
    }
    after_multi = {
        "type": BridgeStateType.CARD_SELECT,
        "cards": [{"index": 0, "id": "STRIKE_IRONCLAD"}],
        "min_select": 0,
        "max_select": 1,
    }
    after_skip = {"type": BridgeStateType.MAP_SELECT, "nodes": [], "floor": 1, "act": 1}
    client = FakeBridgeClient([initial, after_multi, after_skip])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.choose_many([0, 1])
    assert recorder.receive_state() == after_multi
    recorder.skip()
    assert recorder.receive_state() == after_skip

    assert recorder.trace.steps[0].action == {"action": BridgeAction.CHOOSE, "indexes": [0, 1]}
    assert recorder.trace.steps[1].action == {"action": BridgeAction.SKIP}


def test_bridge_replay_recorder_records_terminal_run_state():
    initial = {"type": BridgeStateType.MAP_SELECT, "nodes": [{"index": 0, "type": "Monster"}], "floor": 1, "act": 1}
    terminal = {"type": BridgeStateType.RUN_COMPLETE, "result": "victory"}
    client = FakeBridgeClient([initial, terminal])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.choose(0)
    assert recorder.receive_state() == terminal

    assert recorder.trace.steps[0].action == {"action": BridgeAction.CHOOSE, "index": 0}
    assert recorder.trace.steps[0].resulting_state == terminal


def test_bridge_replay_recorder_normalizes_reward_screen_options():
    initial = {
        "type": BridgeStateType.REWARD_SCREEN,
        "options": [
            {
                "index": 0,
                "id": "gold:25",
                "action": "pick_reward",
                "label": "gold:25",
                "description": "25 Gold",
                "enabled": True,
            },
            {
                "index": 1,
                "id": "proceed",
                "action": "proceed",
                "label": "proceed",
                "enabled": True,
            },
        ],
        "floor": 1,
        "act": 1,
    }
    after_pick = {
        "type": BridgeStateType.REWARD_SCREEN,
        "options": [
            {
                "index": 0,
                "id": "proceed",
                "action": "proceed",
                "label": "proceed",
                "enabled": True,
            }
        ],
        "floor": 1,
        "act": 1,
    }
    client = FakeBridgeClient([initial, after_pick])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.choose(0)
    assert recorder.receive_state() == after_pick

    assert recorder.trace.initial_state == {
        "type": BridgeStateType.REWARD_SCREEN,
        "options": [
            {"index": 0, "action": "pick_reward", "enabled": True},
            {"index": 1, "action": "proceed", "enabled": True},
        ],
        "floor": 1,
        "act": 1,
    }
    assert recorder.trace.steps[0].action == {"action": BridgeAction.CHOOSE, "index": 0}
    assert recorder.trace.steps[0].resulting_state["options"] == [
        {"index": 0, "action": "proceed", "enabled": True}
    ]


def test_normalize_bridge_state_supports_reward_screen():
    state = normalize_bridge_state({
        "type": BridgeStateType.REWARD_SCREEN,
        "options": [
            {"action": "pick_reward", "enabled": True},
            {"action": "proceed", "enabled": False},
        ],
        "floor": 2,
        "act": 1,
    })

    assert state == {
        "type": BridgeStateType.REWARD_SCREEN,
        "options": [
            {"index": 0, "action": "pick_reward", "enabled": True},
            {"index": 1, "action": "proceed", "enabled": False},
        ],
        "floor": 2,
        "act": 1,
    }


def test_bridge_replay_recorder_normalizes_card_bundle_state():
    initial = {
        "type": BridgeStateType.CARD_BUNDLE,
        "bundles": [
            {
                "index": 0,
                "action": "pick_card_bundle",
                "cards": [
                    {"id": "STRIKE_IRONCLAD", "type": "Attack", "cost": 1},
                    {"id": "DEFEND_IRONCLAD", "type": "Skill", "cost": 1},
                ],
                "enabled": True,
            }
        ],
        "floor": 1,
        "act": 1,
    }
    after_pick = {"type": BridgeStateType.MAP_SELECT, "nodes": [], "floor": 2, "act": 1}
    client = FakeBridgeClient([initial, after_pick])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.choose(0)
    assert recorder.receive_state() == after_pick

    assert recorder.trace.initial_state == initial
    assert recorder.trace.steps[0].action == {"action": BridgeAction.CHOOSE, "index": 0}
    assert recorder.trace.steps[0].resulting_state == after_pick


def test_bridge_replay_recorder_normalizes_crystal_sphere_options():
    initial = {
        "type": BridgeStateType.CRYSTAL_SPHERE,
        "options": [
            {
                "index": 0,
                "action": "divine_cell",
                "x": 4,
                "y": 5,
                "label": "4,5",
                "enabled": True,
            },
            {
                "index": 1,
                "action": "proceed",
                "enabled": False,
            },
        ],
        "floor": 20,
        "act": 2,
    }
    after_pick = {
        "type": BridgeStateType.CRYSTAL_SPHERE,
        "options": [
            {
                "index": 0,
                "action": "proceed",
                "enabled": True,
            }
        ],
        "floor": 20,
        "act": 2,
    }
    client = FakeBridgeClient([initial, after_pick])
    recorder = BridgeReplayRecorder(client)

    assert recorder.receive_state() == initial
    recorder.choose(0)
    assert recorder.receive_state() == after_pick

    assert recorder.trace.initial_state == {
        "type": BridgeStateType.CRYSTAL_SPHERE,
        "options": [
            {
                "index": 0,
                "action": "divine_cell",
                "enabled": True,
                "x": 4,
                "y": 5,
                "entity_id": "crystal-cell:4:5",
                "affected_cell_ids": [],
            },
            {"index": 1, "action": "proceed", "enabled": False},
        ],
        "floor": 20,
        "act": 2,
    }
    assert recorder.trace.steps[0].resulting_state["options"] == [
        {"index": 0, "action": "proceed", "enabled": True}
    ]


def test_bridge_replay_recorder_delegates_unknown_attributes():
    client = FakeBridgeClient([])
    recorder = BridgeReplayRecorder(client)

    assert recorder.connected is True
    assert recorder.ping() is True


def test_compare_combat_replay_passes_for_simple_combat_trace():
    combat = make_basic_replay_combat()
    initial_state = combat_state_to_bridge_state(combat)
    assert combat.play_card(0, 0)
    resulting_state = combat_state_to_bridge_state(combat)

    trace = BridgeReplayTrace(
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.PLAY, "card_index": 0, "target_index": 0},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_combat_replay(trace, factory=make_basic_replay_combat)

    assert result.success is True
    assert result.mismatches == []


def test_compare_combat_replay_handles_card_select_round_trip():
    combat = make_choice_replay_combat()
    initial_state = combat_state_to_bridge_state(combat)

    assert combat.play_card(0)
    choice_state = combat_state_to_bridge_state(combat)
    assert combat.resolve_pending_choice(0)
    resulting_state = combat_state_to_bridge_state(combat)

    trace = BridgeReplayTrace(
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.PLAY, "card_index": 0, "target_index": -1},
                resulting_state=choice_state,
            ),
            BridgeReplayStep(action={"action": BridgeAction.CHOOSE, "index": 0}, resulting_state=resulting_state),
        ],
    )
    result = compare_combat_replay(trace, factory=make_choice_replay_combat)

    assert result.success is True
    assert result.mismatches == []


def test_compare_combat_replay_reports_state_mismatch():
    combat = make_basic_replay_combat()
    initial_state = combat_state_to_bridge_state(combat)
    assert combat.play_card(0, 0)
    resulting_state = combat_state_to_bridge_state(combat)
    resulting_state["player"]["hp"] = 999

    trace = BridgeReplayTrace(
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.PLAY, "card_index": 0, "target_index": 0},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_combat_replay(trace, factory=make_basic_replay_combat)

    assert result.success is False
    assert any("player.hp" in mismatch for mismatch in result.mismatches)


def test_compare_combat_replay_handles_potion_action():
    combat = make_basic_replay_combat()
    combat.potions = [create_potion("BlockPotion"), None, None]
    initial_state = combat_state_to_bridge_state(combat)
    assert combat.use_potion(0)
    resulting_state = combat_state_to_bridge_state(combat)

    assert initial_state["potions"] == [{
        "slot": 0,
        "id": "BlockPotion",
        "can_use": True,
        "usage": "CombatOnly",
        "target": "AnyPlayer",
        "requires_target": False,
        "target_type": "AnyPlayer",
    }]
    assert initial_state["max_potion_slots"] == 3
    assert resulting_state["potions"] == []

    trace = BridgeReplayTrace(
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.POTION, "slot": 0, "target_index": -1},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_combat_replay(trace, factory=lambda: make_basic_replay_combat_with_block_potion())

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_map_select():
    run = make_basic_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.MAP_SELECT
    assert state["nodes"]
    assert all("row" in node and "col" in node and "type" in node for node in state["nodes"])


def test_compare_run_replay_handles_map_select_to_combat_transition():
    run = make_basic_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    actions = [action for action in run.get_available_actions() if action.get("action") == "move"]
    run.take_action(actions[0])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_basic_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_compare_run_replay_handles_card_reward_pick_to_map_transition():
    run = make_card_reward_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action({"action": "pick_card", "index": 0})
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_card_reward_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_card_bundle_options():
    run = make_card_bundle_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.CARD_BUNDLE
    assert [bundle["action"] for bundle in state["bundles"]] == ["pick_card_bundle", "pick_card_bundle"]
    assert [bundle["index"] for bundle in state["bundles"]] == [0, 1]
    assert [card["id"] for card in state["bundles"][0]["cards"]] == [
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "BASH",
    ]


def test_compare_run_replay_handles_card_bundle_pick_to_map_transition():
    run = make_card_bundle_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action({"action": "pick_card_bundle", "index": 0})
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_card_bundle_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_rest_site_options():
    run = make_rest_site_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.REST_SITE
    assert state["options"]
    assert state["options"][FIRST_BRIDGE_CHOICE_INDEX]["id"] == "HEAL"
    assert all("label" in option and "enabled" in option for option in state["options"])


def test_compare_run_replay_handles_rest_site_choice_to_map_transition():
    run = make_rest_site_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action(run.get_available_actions()[FIRST_BRIDGE_CHOICE_INDEX])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_rest_site_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_shop_options():
    run = make_shop_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.SHOP
    assert state["options"]
    assert state["options"][FIRST_BRIDGE_CHOICE_INDEX]["action"] == "leave_shop"


def test_compare_run_replay_handles_shop_choice_to_map_transition():
    run = make_shop_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action(run.get_available_actions()[FIRST_BRIDGE_CHOICE_INDEX])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_shop_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_event_options():
    run = make_event_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.EVENT
    assert [option["action"] for option in state["options"]] == ["event_choice", "event_choice"]
    assert [option["index"] for option in state["options"]] == [0, 1]


def test_compare_run_replay_handles_event_choice_to_map_transition():
    run = make_event_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action(run.get_available_actions()[FIRST_BRIDGE_CHOICE_INDEX])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_event_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_treasure_collect_option():
    run = make_treasure_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.TREASURE
    assert state["options"] == [
        {"index": 0, "action": "collect", "enabled": True}
    ]


def test_compare_run_replay_handles_treasure_collect_to_map_transition():
    run = make_treasure_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action(run.get_available_actions()[FIRST_BRIDGE_CHOICE_INDEX])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_treasure_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_run_manager_to_bridge_state_serializes_boss_relic_options():
    run = make_boss_relic_replay_run()
    state = run_manager_to_bridge_state(run)

    assert state["type"] == BridgeStateType.BOSS_RELIC
    assert [option["action"] for option in state["options"]] == ["pick_relic", "pick_relic", "pick_relic"]
    assert [option["index"] for option in state["options"]] == [0, 1, 2]


def test_compare_run_replay_handles_boss_relic_pick_to_next_act_map():
    run = make_boss_relic_replay_run()
    initial_state = run_manager_to_bridge_state(run)
    run.take_action(run.get_available_actions()[FIRST_BRIDGE_CHOICE_INDEX])
    resulting_state = run_manager_to_bridge_state(run)

    trace = BridgeReplayTrace(
        mode="run",
        initial_state=initial_state,
        steps=[
            BridgeReplayStep(
                action={"action": BridgeAction.CHOOSE, "index": FIRST_BRIDGE_CHOICE_INDEX},
                resulting_state=resulting_state,
            )
        ],
    )
    result = compare_run_replay(trace, factory=make_boss_relic_replay_run)

    assert result.success is True
    assert result.mismatches == []


def test_bridge_replay_cli_show_and_compare(tmp_path: Path, capsys):
    trace = BridgeReplayTrace(
        mode="combat",
        metadata={"scenario_factory": "tests.test_bridge_replay_harness:make_basic_replay_combat"},
        initial_state=combat_state_to_bridge_state(make_basic_replay_combat()),
        steps=[],
    )
    trace_path = save_replay_trace(trace, tmp_path / "trace.json")

    parser = build_parser()

    show_args = parser.parse_args(["show", str(trace_path)])
    assert show_args.func(show_args) == 0
    show_output = capsys.readouterr().out
    assert '"step_count": 0' in show_output

    compare_args = parser.parse_args([
        "compare",
        str(trace_path),
        "--factory",
        "tests.test_bridge_replay_harness:make_basic_replay_combat",
    ])
    assert compare_args.func(compare_args) == 0
    compare_output = capsys.readouterr().out
    assert "comparison: ok" in compare_output
