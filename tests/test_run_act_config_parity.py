"""Run act configuration parity tests."""

import numpy as np

from sts2_env.agent_v2.categorical_vocabulary import UNKNOWN_ID, categorical_id
from sts2_env.events.act2 import LuminousChoir, RanwidTheElder
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv
from sts2_env.map.acts import ACT_0, ACT_0_UNDERDOCKS, ACT_1, ACT_2, ALL_ACTS
from sts2_env.potions.base import create_potion
from sts2_env.relics.base import RelicId
from sts2_env.run.events import pick_event
from sts2_env.run.run_state import RunState

RUN_SEED = 42
ACT_TWO_INDEX = 1
ENTRY_BLOCKED_GOLD = LuminousChoir.ENTRY_GOLD_COST - 1
FIRE_POTION_ID = "FirePotion"


def test_initialize_run_generates_shuffled_act_event_rooms_like_csharp_runmanager():
    run_state = RunState(seed=RUN_SEED)
    static_event_ids = [event_id for act in ALL_ACTS for event_id in act.event_ids]

    run_state.initialize_run()

    generated_event_ids = [event_id for act in run_state.acts for event_id in act.event_ids]
    assert sorted(generated_event_ids) == sorted(static_event_ids)
    assert generated_event_ids != static_event_ids
    assert run_state.rng.up_front.counter == len(static_event_ids) - len(run_state.acts)


def test_initialize_run_does_not_regenerate_event_rooms_after_first_initialization():
    run_state = RunState(seed=RUN_SEED)
    run_state.initialize_run()
    event_ids_by_act = [list(act.event_ids) for act in run_state.acts]
    up_front_counter = run_state.rng.up_front.counter

    run_state.initialize_run()

    assert [act.event_ids for act in run_state.acts] == event_ids_by_act
    assert run_state.rng.up_front.counter == up_front_counter


def test_run_state_uses_mutable_act_copies_like_csharp_runstate():
    first_run = RunState(seed=RUN_SEED)
    second_run = RunState(seed=RUN_SEED)

    first_run.acts[ACT_TWO_INDEX].event_ids = [RanwidTheElder.event_id]

    assert second_run.acts[ACT_TWO_INDEX].event_ids != [RanwidTheElder.event_id]
    assert "Amalgamator" in second_run.acts[ACT_TWO_INDEX].event_ids


def test_act_event_pools_match_decompiled_act_ownership():
    assert "HungryForMushrooms" not in ACT_0.event_ids
    assert "HungryForMushrooms" not in ACT_0_UNDERDOCKS.event_ids
    assert "HungryForMushrooms" not in ACT_1.event_ids
    assert "HungryForMushrooms" in ACT_2.event_ids
    assert "AromaOfChaos" in ACT_0.event_ids
    assert "AbyssalBaths" in ACT_0_UNDERDOCKS.event_ids
    assert set(ACT_0.event_ids) & set(ACT_0_UNDERDOCKS.event_ids) == {"SunkenStatue"}


def test_act_one_variant_is_seeded_and_uses_matching_encounters():
    from sts2_env.encounters import act1, act4
    from sts2_env.run.run_manager import _get_encounter_pools

    variants = {
        RunState(seed=seed, act1_variant="random").current_act.act_id
        for seed in range(20)
    }
    assert variants == {"Overgrowth", "Underdocks"}
    assert (
        RunState(seed=100109, act1_variant="random").current_act.act_id
        == RunState(seed=100109, act1_variant="random").current_act.act_id
    )
    assert _get_encounter_pools(0, "Overgrowth")["weak"] == act1.WEAK_ENCOUNTERS
    assert _get_encounter_pools(0, "Underdocks")["weak"] == act4.WEAK_ENCOUNTERS


def test_entity_run_starts_with_neow_and_exposes_act_variant():
    env = STS2EntityRunEnv(max_steps=20)
    try:
        observation, info = env.reset(seed=100109)
        manager = env.run_env._mgr
        assert manager is not None
        assert manager.phase == "EVENT"
        assert manager._event_model.event_id == "Neow"
        assert manager.run_state.total_floor == 0
        assert len(manager.get_available_actions()) == 3
        assert int(np.sum(info["action_mask"])) == 3
        assert observation["global_categorical"][4] == categorical_id(
            manager.run_state.current_act.act_id
        )
        assert UNKNOWN_ID not in observation["global_categorical"]
        assert UNKNOWN_ID not in observation["candidate_categorical"][info["action_mask"].astype(bool)]

        chosen = int(np.flatnonzero(info["action_mask"])[0])
        relics_before = set(manager.run_state.player.relics)
        _, _, _, _, reward_info = env.step(chosen)
        assert manager.phase == "CARD_REWARD"
        assert manager.get_available_actions()[0]["relic_id"] == "POMANDER"
        env.step(int(np.flatnonzero(reward_info["action_mask"])[0]))
        assert manager.phase == "CARD_REWARD"  # Pomander requests a deck card.
        assert len(set(manager.run_state.player.relics) - relics_before) == 1
        assert manager.run_state.total_floor == 0
    finally:
        env.close()


def test_pick_event_advances_through_current_act_event_order_like_csharp_roomset():
    run_state = RunState(seed=RUN_SEED)
    run_state.current_act_index = ACT_TWO_INDEX
    run_state.current_act.event_ids = [
        LuminousChoir.event_id,
        RanwidTheElder.event_id,
    ]
    run_state.player.gold = ENTRY_BLOCKED_GOLD
    run_state.player.add_potion(create_potion(FIRE_POTION_ID))
    run_state.player.obtain_relic(RelicId.ANCHOR.name)

    event = pick_event(run_state)

    assert isinstance(event, RanwidTheElder)
    assert run_state.current_act.events_visited == 2
    assert run_state.rng.up_front.counter == 0


def test_pick_event_with_explicit_pool_does_not_mutate_act_event_cursor():
    run_state = RunState(seed=RUN_SEED)
    run_state.current_act_index = ACT_TWO_INDEX
    run_state.current_act.events_visited = 1
    run_state.player.gold = RanwidTheElder.ENTRY_GOLD_COST
    run_state.player.add_potion(create_potion(FIRE_POTION_ID))
    run_state.player.obtain_relic(RelicId.ANCHOR.name)

    event = pick_event(run_state, pool=[RanwidTheElder.event_id])

    assert isinstance(event, RanwidTheElder)
    assert run_state.current_act.events_visited == 1


def test_pick_event_repeats_current_act_event_after_unique_events_are_exhausted_like_csharp_roomset():
    run_state = RunState(seed=RUN_SEED)
    run_state.current_act_index = ACT_TWO_INDEX
    run_state.current_act.event_ids = [RanwidTheElder.event_id]
    run_state.current_act.events_visited = 1
    run_state.visited_event_ids.add(RanwidTheElder.event_id)
    run_state.player.gold = RanwidTheElder.ENTRY_GOLD_COST
    run_state.player.add_potion(create_potion(FIRE_POTION_ID))
    run_state.player.obtain_relic(RelicId.ANCHOR.name)

    event = pick_event(run_state)

    assert isinstance(event, RanwidTheElder)
    assert run_state.current_act.events_visited == 3
