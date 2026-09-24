"""Parity tests for quest, potion, and Scriptorium events."""

import sts2_env.events.act1  # noqa: F401
import sts2_env.events.act2  # noqa: F401

from sts2_env.cards.factory import create_card
from sts2_env.cards.factory import eligible_registered_cards
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.status import make_spore_mind
from sts2_env.core.card_pools import CardPoolId
from sts2_env.core.enums import CardId, CardRarity, CardType, MapPointType
from sts2_env.events.act2 import (
    SpiralingWhirlpool,
    StoneOfAllTime,
    TheFutureOfPotions,
    WaterloggedScriptorium,
)
from sts2_env.potions.base import create_potion
from sts2_env.relics.base import RelicId
from sts2_env.run.reward_objects import CardReward, RelicReward
from sts2_env.run.rewards import CARD_CREATION_SOURCE_OTHER
from sts2_env.run.run_manager import RunManager
from sts2_env.run.run_state import PlayerState, RunState
from sts2_env.map.generator import generate_act_map


def _make_run_state(seed: int = 401) -> RunState:
    run_state = RunState(seed=seed, character_id="Ironclad")
    run_state.initialize_run()
    run_state.player.deck = create_ironclad_starter_deck()
    return run_state


class _LastChoiceCountingRng:
    def __init__(self) -> None:
        self.choice_calls = 0
        self.next_int_calls = 0
        self.next_int_args: list[tuple[int, int]] = []

    def choice(self, seq):
        self.choice_calls += 1
        return seq[-1]

    def next_int(self, low: int, high: int) -> int:
        self.next_int_calls += 1
        self.next_int_args.append((low, high))
        return low


def test_spoils_map_generates_act2_center_treasure_quest():
    run_state = _make_run_state(441)
    spoils = create_card(CardId.SPOILS_MAP)
    run_state.player.deck.append(spoils)
    run_state.current_act_index = 1

    run_state.generate_map()

    assert run_state.map is not None
    treasure_points = [point for point in run_state.map.room_points() if point.point_type == MapPointType.TREASURE]
    assert len(treasure_points) == 1
    treasure = treasure_points[0]
    assert treasure.col == 3
    assert treasure.row == run_state.map.map_length - 7
    assert run_state.map.get_row(treasure.row) == [treasure]
    assert spoils in treasure.quests
    assert spoils.effect_vars["spoils_col"] == treasure.col
    assert spoils.effect_vars["spoils_row"] == treasure.row


def test_spoils_map_card_hook_replaces_only_configured_act_map():
    run_state = _make_run_state(443)
    spoils = create_card(CardId.SPOILS_MAP)
    standard_map = generate_act_map(
        num_rooms=run_state.current_act.num_rooms,
        rng=run_state.rng.get_map_rng(run_state.current_act_index),
        ascension_level=run_state.ascension_level,
        act_index=run_state.current_act_index,
    )

    assert spoils.modify_generated_map(run_state, standard_map, 0) is standard_map

    act2_map = spoils.modify_generated_map(run_state, standard_map, 1)
    treasure_points = [point for point in act2_map.room_points() if point.point_type == MapPointType.TREASURE]

    assert act2_map is not standard_map
    assert len(treasure_points) == 1
    assert treasure_points[0].col == 3


def test_spoils_map_treasure_completion_grants_gold_and_removes_card():
    mgr = RunManager(seed=442, character_id="Ironclad")
    spoils = create_card(CardId.SPOILS_MAP)
    mgr.run_state.player.deck.append(spoils)
    mgr.run_state.current_act_index = 1
    mgr.run_state.generate_map()
    assert mgr.run_state.map is not None
    treasure = next(point for point in mgr.run_state.map.room_points() if point.point_type == MapPointType.TREASURE)
    mgr.run_state.visited_map_coords = [treasure.coord]
    mgr._phase = RunManager.PHASE_TREASURE
    mgr._current_reward = RelicReward(mgr.run_state.player.player_id, relic_id="LANTERN")
    starting_gold = mgr.run_state.player.gold

    result = mgr._do_treasure_collect()

    assert result["phase"] == RunManager.PHASE_MAP_CHOICE
    assert result["spoils_gold"] == 600
    assert mgr.run_state.player.gold == starting_gold + result["treasure_gold"] + 600
    assert spoils not in mgr.run_state.player.deck
    assert spoils not in treasure.quests


def test_spoils_map_card_hook_completes_quest_and_removes_itself():
    run_state = _make_run_state(444)
    spoils = create_card(CardId.SPOILS_MAP)
    run_state.player.deck.append(spoils)
    run_state.current_act_index = 1
    run_state.generate_map()
    treasure = next(point for point in run_state.map.room_points() if point.point_type == MapPointType.TREASURE)
    starting_gold = run_state.player.gold

    gold = spoils.on_quest_complete(run_state)

    assert gold == spoils.effect_vars["gold"]
    assert run_state.player.gold == starting_gold + gold
    assert spoils not in run_state.player.deck
    assert spoils not in treasure.quests


def test_spiraling_whirlpool_requires_spiral_targets_and_applies_observe_drink():
    blocked_state = _make_run_state(402)
    blocked_state.player.deck = [create_card(CardId.BASH)]
    blocked_event = SpiralingWhirlpool()
    assert blocked_event.is_allowed(blocked_state) is False

    run_state = _make_run_state(403)
    event = SpiralingWhirlpool()
    assert event.is_allowed(run_state)

    observe = event.choose(run_state, "observe")
    assert not observe.finished
    assert event.pending_choice is not None
    target = event.pending_choice.options[0].card

    resolved = event.resolve_pending_choice(0)
    assert resolved.finished
    assert target.enchantments.get("Spiral") == 1

    run_state.player.current_hp = run_state.player.max_hp - 20
    heal_before = run_state.player.current_hp
    heal_amount = event.heal_amount(run_state)
    drink = event.choose(run_state, "drink")
    assert drink.finished
    assert run_state.player.current_hp == min(run_state.player.max_hp, heal_before + heal_amount)


def test_spiraling_whirlpool_requires_all_players_to_have_spiral_targets():
    run_state = _make_run_state(4031)
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent"))
    ally.deck = [create_card(CardId.BASH)]
    event = SpiralingWhirlpool()

    assert event.is_allowed(run_state) is False

    ally.deck = [create_card(CardId.STRIKE_IRONCLAD)]
    assert event.is_allowed(run_state) is True


def test_stone_of_all_time_lift_discards_potion_and_push_enchants_attack():
    run_state = _make_run_state(404)
    run_state.current_act_index = 1
    run_state.player.add_potion(create_potion("FirePotion"))
    run_state.player.add_potion(create_potion("FlexPotion"))
    event = StoneOfAllTime()

    assert event.is_allowed(run_state)
    options = event.generate_initial_options(run_state)
    assert [option.enabled for option in options] == [True, True]
    potions_before = len(run_state.player.held_potions())
    max_hp_before = run_state.player.max_hp
    lift = event.choose(run_state, "lift")
    assert lift.finished
    assert run_state.player.max_hp == max_hp_before + 10
    assert len(run_state.player.held_potions()) == potions_before - 1

    hp_before = run_state.player.current_hp
    push = event.choose(run_state, "push")
    assert not push.finished
    assert event.pending_choice is not None
    target = event.pending_choice.options[0].card

    resolved = event.resolve_pending_choice(0)
    assert resolved.finished
    assert run_state.player.current_hp == hp_before - 6
    assert target.enchantments.get("Vigorous") == 8

    no_push_state = _make_run_state(4041)
    no_push_state.current_act_index = 1
    no_push_state.player.add_potion(create_potion("FirePotion"))
    no_push_state.player.deck = [make_spore_mind(), make_spore_mind()]
    no_push_event = StoneOfAllTime()
    no_push_options = no_push_event.generate_initial_options(no_push_state)
    assert [option.enabled for option in no_push_options] == [True, False]


def test_stone_of_all_time_requires_all_players_to_have_potions():
    run_state = _make_run_state(4042)
    run_state.current_act_index = 1
    run_state.player.add_potion(create_potion("FirePotion"))
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent"))
    event = StoneOfAllTime()

    assert event.is_allowed(run_state) is False

    ally.add_potion(create_potion("FirePotion"))
    assert event.is_allowed(run_state) is True


def test_stone_of_all_time_lift_uses_preselected_event_rng_potion_and_consumes_rng():
    run_state = _make_run_state(4042)
    run_state.current_act_index = 1
    run_state.player.add_potion(create_potion("FirePotion"))
    run_state.player.add_potion(create_potion("FlexPotion"))
    event = StoneOfAllTime()
    event.rng = _LastChoiceCountingRng()

    options = event.generate_initial_options(run_state)
    assert [option.enabled for option in options] == [True, True]
    assert event.rng.choice_calls == 1
    assert event._lift_potion_slot == 1  # noqa: SLF001

    result = event.choose(run_state, "lift")

    assert result.finished
    assert event.rng.choice_calls == 1
    assert event.rng.next_int_calls == 1
    assert event.rng.next_int_args == [(0, 99)]
    assert [potion.potion_id for potion in run_state.player.held_potions()] == ["FirePotion"]


def test_stone_of_all_time_push_consumes_event_rng_after_enchanting():
    run_state = _make_run_state(4043)
    run_state.current_act_index = 1
    run_state.player.add_potion(create_potion("FirePotion"))
    event = StoneOfAllTime()
    event.rng = _LastChoiceCountingRng()
    event.generate_initial_options(run_state)

    result = event.choose(run_state, "push")
    target = event.pending_choice.options[0].card
    resolved = event.resolve_pending_choice(0)

    assert result.finished is False
    assert resolved.finished
    assert target.enchantments.get("Vigorous") == 8
    assert event.rng.next_int_calls == 1
    assert event.rng.next_int_args == [(0, 99)]


def test_stone_of_all_time_deferred_push_consumes_event_rng_after_reward_choice():
    mgr = RunManager(seed=4044, character_id="Ironclad")
    mgr.run_state.current_act_index = 1
    mgr.run_state.player.add_potion(create_potion("FirePotion"))
    mgr._phase = RunManager.PHASE_EVENT
    event = StoneOfAllTime()
    event.rng = _LastChoiceCountingRng()
    mgr._event_model = event
    mgr._event_options = event.generate_initial_options(mgr.run_state)

    result = mgr._do_event_choice({"option_id": "push"})

    assert result["phase"] == RunManager.PHASE_CARD_REWARD
    assert event.rng.next_int_calls == 0
    assert mgr.run_state.pending_choice is not None

    final = mgr.take_action({"action": "choose", "index": 0})

    assert final["phase"] == RunManager.PHASE_MAP_CHOICE
    assert any(card.enchantments.get("Vigorous") == 8 for card in mgr.run_state.player.deck)
    assert event.rng.next_int_calls == 1
    assert event.rng.next_int_args == [(0, 99)]


def test_future_of_potions_trade_discards_potion_and_builds_upgraded_rewards():
    run_state = _make_run_state(405)
    run_state.player.add_potion(create_potion("FirePotion"))
    run_state.player.add_potion(create_potion("Clarity"))
    run_state.player.add_potion(create_potion("FairyInABottle"))
    event = TheFutureOfPotions()
    up_front_counter = run_state.rng.up_front.counter

    assert event.is_allowed(run_state)
    options = event.generate_initial_options(run_state)
    assert [opt.option_id for opt in options] == ["trade_0", "trade_1", "trade_2"]
    assert run_state.rng.up_front.counter == up_front_counter

    potions_before = len(run_state.player.held_potions())
    rewards_counter_before = run_state.rng.rewards.counter
    result = event.choose(run_state, "trade_0")
    assert result.finished
    assert len(run_state.player.held_potions()) == potions_before - 1
    assert run_state.rng.rewards.counter == rewards_counter_before

    reward_objects = result.rewards["reward_objects"]
    assert len(reward_objects) == 1
    assert isinstance(reward_objects[0], CardReward)
    reward = reward_objects[0]
    assert reward.cards == []
    assert reward.option_count == 3
    assert reward.forced_rarities == ()
    assert reward.generation_context is None
    assert reward.roll_upgrade is False
    assert reward.card_creation_source == CARD_CREATION_SOURCE_OTHER
    assert reward.use_default_character_pool is False
    assert reward.card_type in {CardType.ATTACK, CardType.SKILL}
    assert reward.allow_rarity_modifications is False
    assert reward.card_pool_rarity_filter is CardRarity.COMMON
    assert reward.use_uniform_noncombat_odds is True
    assert reward.upgrade_after_generation is True

    reward.populate(run_state, None)
    assert len(reward.cards) == 3
    assert all(card.upgraded for card in reward.cards)
    assert all(card.rarity == CardRarity.COMMON for card in reward.cards)
    assert len({card.card_type for card in reward.cards}) == 1
    assert reward.cards[0].card_type in {CardType.ATTACK, CardType.SKILL}


def test_future_of_potions_rolls_card_types_for_all_held_potions_before_showing_first_three():
    run_state = _make_run_state(40502)
    run_state.player.max_potion_slots = 5
    for potion_id in ("FirePotion", "Clarity", "FairyInABottle", "AttackPotion", "LiquidMemories"):
        assert run_state.player.add_potion(create_potion(potion_id))
    event = TheFutureOfPotions()
    event.rng = _LastChoiceCountingRng()

    options = event.generate_initial_options(run_state)

    assert [option.option_id for option in options] == ["trade_0", "trade_1", "trade_2"]
    assert event.rng.choice_calls == 5


def test_future_of_potions_no_rarity_modification_limits_dingy_rug_pool():
    run_state = _make_run_state(40501)
    assert run_state.player.obtain_relic(RelicId.DINGY_RUG.name)
    run_state.player.add_potion(create_potion("Clarity"))
    run_state.player.add_potion(create_potion("FairyInABottle"))
    event = TheFutureOfPotions()
    event.rng = _LastChoiceCountingRng()
    event.generate_initial_options(run_state)

    result = event.choose(run_state, "trade_0")
    reward = result.rewards["reward_objects"][0]
    reward.populate(run_state, None)

    colorless_uncommon_pool = set(
        eligible_registered_cards(
            card_pool=CardPoolId.COLORLESS,
            rarity=CardRarity.UNCOMMON,
            card_type=reward.card_type,
            generation_context=None,
        )
    )
    colorless_rare_pool = set(
        eligible_registered_cards(
            card_pool=CardPoolId.COLORLESS,
            rarity=CardRarity.RARE,
            card_type=reward.card_type,
            generation_context=None,
        )
    )
    assert any(card_id in reward.custom_card_ids for card_id in colorless_uncommon_pool)
    assert not any(card_id in reward.custom_card_ids for card_id in colorless_rare_pool)
    assert all(card.rarity is CardRarity.UNCOMMON for card in reward.cards)
    assert all(card.upgraded for card in reward.cards)


def test_future_of_potions_requires_all_players_to_have_two_potions():
    run_state = _make_run_state(4051)
    run_state.player.add_potion(create_potion("FirePotion"))
    run_state.player.add_potion(create_potion("Clarity"))
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent"))
    ally.add_potion(create_potion("FirePotion"))
    event = TheFutureOfPotions()

    assert event.is_allowed(run_state) is False

    ally.add_potion(create_potion("Clarity"))
    assert event.is_allowed(run_state) is True


def test_waterlogged_scriptorium_tentacle_and_prickly_apply_steady_enchantments():
    run_state = _make_run_state(406)
    run_state.player.gold = 320
    event = WaterloggedScriptorium()

    locked_state = _make_run_state(4051)
    locked_state.player.gold = 64
    locked_options = event.generate_initial_options(locked_state)
    assert [option.option_id for option in locked_options] == ["bloody_ink", "tentacle_quill", "prickly_sponge"]
    assert [option.enabled for option in locked_options] == [True, False, False]

    max_hp_before = run_state.player.max_hp
    bloody = event.choose(run_state, "bloody_ink")
    assert bloody.finished
    assert run_state.player.max_hp == max_hp_before + 6

    gold_before = run_state.player.gold
    tentacle = event.choose(run_state, "tentacle_quill")
    assert not tentacle.finished
    assert event.pending_choice is not None
    assert run_state.player.gold == gold_before - 65
    first = event.pending_choice.options[0].card
    tentacle_done = event.resolve_pending_choice(0)
    assert tentacle_done.finished
    assert first.enchantments.get("Steady") == 1

    gold_before = run_state.player.gold
    prickly = event.choose(run_state, "prickly_sponge")
    assert not prickly.finished
    assert event.pending_choice is not None
    assert run_state.player.gold == gold_before - 155
    selected_cards = [event.pending_choice.options[0].card, event.pending_choice.options[1].card]
    event.resolve_pending_choice(0)
    event.resolve_pending_choice(1)
    prickly_done = event.resolve_pending_choice(None)
    assert prickly_done.finished
    assert all(card.enchantments.get("Steady") == 1 for card in selected_cards)


def test_waterlogged_scriptorium_requires_all_players_to_have_spawn_gold():
    run_state = _make_run_state(4061)
    run_state.player.gold = 65
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent", gold=64))
    event = WaterloggedScriptorium()

    assert event.is_allowed(run_state) is False

    ally.gold = 65
    assert event.is_allowed(run_state) is True


def test_waterlogged_scriptorium_prickly_handles_sparse_or_empty_candidates():
    one_candidate_state = _make_run_state(407)
    one_candidate_state.player.gold = 200
    one_candidate_state.player.deck = [
        create_card(CardId.STRIKE_IRONCLAD),
        make_spore_mind(),
        make_spore_mind(),
    ]
    event = WaterloggedScriptorium()
    result = event.choose(one_candidate_state, "prickly_sponge")
    assert not result.finished
    assert event.pending_choice is not None
    assert len(event.pending_choice.options) == 1
    done = event.resolve_pending_choice(0)
    assert done.finished
    assert one_candidate_state.player.deck[0].enchantments.get("Steady") == 1

    no_candidate_state = _make_run_state(408)
    no_candidate_state.player.gold = 200
    no_candidate_state.player.deck = [make_spore_mind(), make_spore_mind()]
    no_card_event = WaterloggedScriptorium()
    gold_before = no_candidate_state.player.gold
    finished = no_card_event.choose(no_candidate_state, "prickly_sponge")
    assert finished.finished
    assert no_card_event.pending_choice is None
    assert no_candidate_state.player.gold == gold_before - 155
