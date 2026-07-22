"""Focused regression tests for the v0.108-v0.109 reference refresh."""

import sts2_env.potions  # noqa: F401
import sts2_env.powers  # noqa: F401

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad_basic import create_ironclad_starter_deck, make_defend_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, PowerId
from sts2_env.core.hooks import fire_after_taking_extra_turn, should_take_extra_turn
from sts2_env.core.rng import Rng
from sts2_env.encounters.act3 import setup_aeonglass_boss
from sts2_env.encounters.act4 import setup_seapunk_normal
from sts2_env.encounters.events import (
    setup_battleworn_dummy_event_v1,
    setup_battleworn_dummy_event_v2,
    setup_battleworn_dummy_event_v3,
)
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.monsters.act3 import (
    AEONGLASS_BASE_EBB_DAMAGE,
    AEONGLASS_BASE_HP,
    AEONGLASS_EBB_MOVE,
    AEONGLASS_EYE_LASERS_MOVE,
    AEONGLASS_INCREASING_INTENSITY_MOVE,
)
from sts2_env.monsters.shared import create_deprecated_monster
from sts2_env.potions.all import AMBERGRIS_ID
from sts2_env.potions.base import create_potion
from sts2_env.relics.base import RelicId
from sts2_env.relics.registry import create_relic
from sts2_env.run.run_state import RunState


NEW_CARD_IDS = (
    CardId.ABUNDANCE,
    CardId.BLADE_SYMPHONY,
    CardId.CACOPHONY,
    CardId.CONCOCT,
    CardId.CONSTELLATION,
    CardId.HIBERNATE,
    CardId.IMITATION_LEARNING,
    CardId.NOT_YET,
    CardId.ONE_FOR_ALL,
    CardId.PLOT,
    CardId.SCARE,
    CardId.SOULBOUND,
    CardId.TUTOR,
    CardId.UNDERWORLD,
    CardId.WITHER,
)

NEW_ANCIENT_RELIC_IDS = (
    RelicId.FISHING_ROD,
    RelicId.HEFTY_TABLET,
    RelicId.KALEIDOSCOPE,
    RelicId.NEOWS_BONES,
    RelicId.NEOWS_TALISMAN,
    RelicId.PHIAL_HOLSTER,
    RelicId.SILKEN_TRESS,
    RelicId.WINGED_BOOTS,
)


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
    return combat


def test_aeonglass_boss_setup_and_rotation_match_v0109() -> None:
    combat = _combat()
    combat.enemies.clear()
    combat.enemy_ais.clear()

    setup_aeonglass_boss(combat, Rng(109))

    aeonglass = combat.enemies[0]
    ai = combat.enemy_ais[aeonglass.combat_id]
    assert aeonglass.max_hp == AEONGLASS_BASE_HP
    assert aeonglass.get_power_amount(PowerId.ARTIFACT) == 3
    assert combat.player.get_power_amount(PowerId.WITHERING_PRESENCE) == 6
    assert ai.states[AEONGLASS_EBB_MOVE].intents[0].damage == AEONGLASS_BASE_EBB_DAMAGE
    assert ai.states[AEONGLASS_EBB_MOVE].follow_up_id == AEONGLASS_EYE_LASERS_MOVE
    assert ai.states[AEONGLASS_EYE_LASERS_MOVE].follow_up_id == AEONGLASS_INCREASING_INTENSITY_MOVE


def test_ambergris_heals_half_max_hp_and_grants_one_extra_turn() -> None:
    combat = _combat()
    combat.player.current_hp = 20

    create_potion(AMBERGRIS_ID).use(combat, combat.player, combat.player)

    assert combat.player.current_hp == 60
    assert combat.player.get_power_amount(PowerId.AMBERGRIS) == 1
    assert should_take_extra_turn(combat)
    fire_after_taking_extra_turn(combat)
    assert combat.player.get_power_amount(PowerId.AMBERGRIS) == 0


def test_midnight_cost_drops_after_each_exhaust() -> None:
    combat = _combat()
    midnight = create_card(CardId.MIDNIGHT)
    fodder = make_defend_ironclad()
    combat.hand = [midnight, fodder]

    combat.exhaust_card(fodder)

    assert midnight.cost == 11


def test_new_ancient_relic_pickups_add_their_rewards() -> None:
    dowsing_run = RunState(seed=109, character_id="Ironclad")
    assert dowsing_run.player.obtain_relic("DOWSING_ROD")
    assert any(card.card_id is CardId.DOWSING for card in dowsing_run.player.deck)

    sacrifice_run = RunState(seed=110, character_id="Ironclad")
    assert sacrifice_run.player.obtain_relic("NEOWS_SACRIFICE")
    assert any(card.card_id is CardId.GUILTY for card in sacrifice_run.player.deck)
    assert any(potion is not None and potion.potion_id == AMBERGRIS_ID for potion in sacrifice_run.player.potions)


def test_new_card_and_ancient_relic_factories_are_registered() -> None:
    assert all(create_card(card_id).card_id is card_id for card_id in NEW_CARD_IDS)
    assert all(create_relic(relic_id).relic_id is relic_id for relic_id in NEW_ANCIENT_RELIC_IDS)


def test_seapunk_normal_replaces_toadpoles_normal() -> None:
    combat = _combat()
    combat.enemies.clear()
    combat.enemy_ais.clear()

    setup_seapunk_normal(combat, Rng(109))

    assert [enemy.monster_id for enemy in combat.enemies] == ["CALCIFIED_CULTIST", "SEAPUNK"]


def test_split_battleworn_dummy_event_encounters() -> None:
    for setup, expected in (
        (setup_battleworn_dummy_event_v1, "BATTLE_FRIEND_V1"),
        (setup_battleworn_dummy_event_v2, "BATTLE_FRIEND_V2"),
        (setup_battleworn_dummy_event_v3, "BATTLE_FRIEND_V3"),
    ):
        combat = _combat()
        combat.enemies.clear()
        combat.enemy_ais.clear()
        setup(combat, Rng(109))
        assert combat.enemies[0].monster_id == expected


def test_no_energy_gain_and_tainted_powers_match_new_hooks() -> None:
    combat = _combat()
    combat.energy = 0
    combat.apply_power_to(combat.player, PowerId.NO_ENERGY_GAIN, 1)
    combat.gain_energy(combat.player, 3)
    assert combat.energy == 0

    enemy = combat.enemies[0]
    combat.apply_power_to(combat.player, PowerId.TAINTED, 3, applier=enemy)
    start_hp = combat.player.current_hp
    combat.deal_damage(enemy, combat.player, 5)
    assert combat.player.current_hp == start_hp - 8


def test_deprecated_monster_run_history_placeholder() -> None:
    monster, ai = create_deprecated_monster(Rng(109))
    assert monster.monster_id == "DEPRECATED_MONSTER"
    assert monster.max_hp == 0
    assert ai.current_move.state_id == "STUB"
