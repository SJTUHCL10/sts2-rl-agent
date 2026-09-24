"""Tests for post-combat reward sequencing in RunManager."""

import subprocess
import sys

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.status import make_guilty
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, RoomType
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.potions.base import create_potion
from sts2_env.run.run_manager import RunManager
from sts2_env.run.reward_objects import RelicReward


def _won_combat(extra_card_rewards: int = 0) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=7,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(7))
    combat.add_enemy(creature, ai)
    combat.player_won = True
    combat.extra_card_rewards = extra_card_rewards
    return combat


def test_the_hunt_extra_reward_creates_second_card_reward_screen():
    mgr = RunManager(seed=1, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=1)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0

    result = mgr._resolve_combat_end()

    assert result["player_won"] is True
    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert len(mgr._offered_cards) == 3

    skip_one = mgr._do_card_reward_skip()
    assert skip_one["phase"] == RunManager.PHASE_CARD_REWARD
    assert len(mgr._offered_cards) == 3

    skip_two = mgr._do_card_reward_skip()
    assert skip_two["phase"] == RunManager.PHASE_MAP_CHOICE


def test_normal_victory_still_uses_single_card_reward_screen():
    mgr = RunManager(seed=2, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0

    mgr._resolve_combat_end()

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    skip = mgr._do_card_reward_skip()
    assert skip["phase"] == RunManager.PHASE_MAP_CHOICE


def test_burning_blood_heals_once_when_combat_returns_to_run() -> None:
    mgr = RunManager(seed=92, character_id="Ironclad")
    mgr.run_state.player.current_hp = 50
    mgr._enter_combat(RoomType.MONSTER)
    combat = mgr.get_combat_state()
    assert combat is not None
    combat._end_combat(player_won=True)

    assert combat.player.current_hp == 56
    assert combat.victory_healed == 6
    result = mgr._resolve_combat_end()

    assert mgr.run_state.player.current_hp == 56
    assert result["healed"] == 6


def test_plain_run_manager_import_registers_playable_events() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from sts2_env.run.run_manager import RunManager; "
            "from sts2_env.run.events import all_events; "
            "m=RunManager(seed=100109, character_id='Ironclad'); "
            "m._enter_event(); "
            "assert len(all_events()) >= 60; "
            "assert m._event_model is not None"
        )],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_opening_chest_rolls_gold_before_relic_reward() -> None:
    mgr = RunManager(seed=93, character_id="Ironclad")
    starting_gold = mgr.run_state.player.gold
    rewards_rng = mgr.run_state.rng.rewards
    expected_roll = Rng(rewards_rng.seed, counter=rewards_rng.counter).next_int_exclusive(42, 53)
    mgr._enter_treasure()
    assert isinstance(mgr._current_reward, RelicReward)
    assert [type(item) for item in mgr._current_rewards.rewards] == [RelicReward]
    assert mgr.run_state.player.gold == starting_gold + expected_roll
    mgr._current_reward = RelicReward(mgr.run_state.player.player_id, relic_id="LANTERN")
    counter_after_open = rewards_rng.counter

    result = mgr._do_treasure_collect()

    assert result["treasure_gold"] == expected_roll
    assert mgr.run_state.player.gold == starting_gold + expected_roll
    assert rewards_rng.counter == counter_after_open
    assert "LANTERN" in mgr.run_state.player.relics


def test_poverty_reduces_chest_gold_after_roll() -> None:
    mgr = RunManager(seed=94, character_id="Ironclad", ascension_level=3)
    starting_gold = mgr.run_state.player.gold
    rewards_rng = mgr.run_state.rng.rewards
    expected_roll = Rng(rewards_rng.seed, counter=rewards_rng.counter).next_int_exclusive(42, 53)
    mgr._enter_treasure()
    assert mgr.run_state.player.gold == starting_gold + int(expected_roll * 0.75)
    mgr._current_reward = RelicReward(mgr.run_state.player.player_id, relic_id="LANTERN")

    result = mgr._do_treasure_collect()

    assert result["treasure_gold"] == int(expected_roll * 0.75)
    assert mgr.run_state.player.gold == starting_gold + result["treasure_gold"]


def test_skipped_treasure_does_not_open_chest_or_grant_gold() -> None:
    mgr = RunManager(seed=95, character_id="Ironclad")
    assert mgr.run_state.player.obtain_relic("SILVER_CRUCIBLE")
    starting_gold = mgr.run_state.player.gold
    rewards_counter = mgr.run_state.rng.rewards.counter

    mgr._enter_room(RoomType.TREASURE)

    assert mgr.phase == RunManager.PHASE_MAP_CHOICE
    assert mgr.run_state.player.gold == starting_gold
    assert mgr.run_state.rng.rewards.counter == rewards_counter


def test_elite_victory_offers_relic_reward_object():
    mgr = RunManager(seed=3, character_id="Ironclad")
    mgr.run_state.potion_reward_odds.current_value = 0.0
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.ELITE

    mgr._resolve_combat_end()

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert mgr._offered_relic is None

    mgr._do_card_reward_skip()

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert mgr._offered_relic is not None


def test_combat_end_syncs_player_max_hp_back_to_run_state():
    mgr = RunManager(seed=4, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0
    mgr._combat.player.max_hp = 86
    mgr._combat.player.current_hp = 70

    mgr._resolve_combat_end()

    assert mgr.run_state.player.max_hp == 86
    assert mgr.run_state.player.current_hp == 70


def test_guilty_counts_combat_victories_and_removes_itself_after_five():
    mgr = RunManager(seed=5, character_id="Ironclad")
    guilty = make_guilty()
    mgr.run_state.player.deck = [guilty]
    mgr.run_state.potion_reward_odds.current_value = -1.0

    for expected_seen in range(1, 5):
        mgr._combat = _won_combat(extra_card_rewards=0)
        mgr._current_room_type = RoomType.MONSTER
        mgr._current_room = None

        mgr._resolve_combat_end()

        assert mgr.run_state.player.deck == [guilty]
        assert guilty.effect_vars["combats_seen"] == expected_seen
        assert guilty.effect_vars["combats"] == 5 - expected_seen

    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._current_room = None

    mgr._resolve_combat_end()

    assert all(card.card_id != CardId.GUILTY for card in mgr.run_state.player.deck)


def test_guilty_deck_combat_end_behavior_lives_on_card_hook():
    mgr = RunManager(seed=55, character_id="Ironclad")
    guilty = make_guilty()

    for expected_seen in range(1, 5):
        assert guilty.after_combat_end_in_deck(mgr.run_state.player)
        assert guilty.effect_vars["combats_seen"] == expected_seen
        assert guilty.effect_vars["combats"] == 5 - expected_seen

    assert not guilty.after_combat_end_in_deck(mgr.run_state.player)
    assert guilty.effect_vars["combats_seen"] == 5
    assert guilty.effect_vars["combats"] == 0


def test_fruit_juice_updates_persistent_player_state_inside_combat():
    combat = _won_combat()
    combat.player_won = False
    starting_max_hp = combat.current_player_state.player_state.max_hp

    potion = create_potion("FruitJuice")
    potion.use(combat, combat.player)

    assert combat.player.max_hp == starting_max_hp + 5
    assert combat.current_player_state.player_state.max_hp == starting_max_hp + 5
