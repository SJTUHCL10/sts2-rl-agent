"""Focused behavioral parity checks for the STS2 v0.111.0 update."""

import sts2_env.powers  # noqa: F401

from sts2_env.cards.defect import create_defect_starter_deck, make_hyperbeam
from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, CombatSide, PowerId
from sts2_env.core.hooks import fire_after_turn_end
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle


def _combat(deck, *, character_id: str) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=deck,
        rng_seed=111,
        character_id=character_id,
    )
    enemy, ai = create_shrinker_beetle(Rng(111))
    combat.add_enemy(enemy, ai)
    combat.start_combat()
    return combat


def test_hyperbeam_focus_loss_is_restored_at_turn_end() -> None:
    combat = _combat(create_defect_starter_deck(), character_id="Defect")
    enemy = combat.enemies[0]
    combat.player.apply_power(PowerId.FOCUS, 5)
    combat.hand = [make_hyperbeam()]
    combat.energy = 2

    assert combat.play_card(0, 0)
    assert enemy.current_hp == enemy.max_hp - 24
    assert combat.player.get_power_amount(PowerId.FOCUS) == 2
    assert combat.player.get_power_amount(PowerId.HYPERBEAM_FOCUS_DOWN) == 3

    fire_after_turn_end(CombatSide.PLAYER, combat)

    assert combat.player.get_power_amount(PowerId.FOCUS) == 5
    assert combat.player.get_power_amount(PowerId.HYPERBEAM_FOCUS_DOWN) == 0


def test_expect_a_fight_scales_block_with_nonnegative_strength() -> None:
    combat = _combat(create_ironclad_starter_deck(), character_id="Ironclad")
    combat.player.apply_power(PowerId.STRENGTH, 2)
    combat.hand = [create_card(CardId.EXPECT_A_FIGHT)]
    combat.energy = 3

    assert combat.play_card(0)
    assert combat.player.block == 25


def test_inky_applies_weak_without_adding_damage() -> None:
    combat = _combat(create_ironclad_starter_deck(), character_id="Ironclad")
    enemy = combat.enemies[0]
    strike = make_strike_ironclad()
    strike.add_enchantment("Inky", 1)
    combat.hand = [strike]
    combat.energy = 1

    assert combat.play_card(0, 0)
    assert enemy.current_hp == enemy.max_hp - 6
    assert enemy.get_power_amount(PowerId.WEAK) == 1
