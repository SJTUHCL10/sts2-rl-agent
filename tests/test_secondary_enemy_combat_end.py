"""Combat-end parity for primary enemies and Minion secondary enemies."""

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1 import create_eye_with_teeth, create_flyconid, create_fogmog


def _fogmog_combat(
    *, another_primary: bool = False,
) -> tuple[CombatState, Creature, Creature, Creature | None]:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=109,
        character_id="Ironclad",
    )
    fogmog, fogmog_ai = create_fogmog(Rng(109))
    eye, eye_ai = create_eye_with_teeth(Rng(110))
    combat.add_enemy(fogmog, fogmog_ai)
    combat.add_enemy(eye, eye_ai)
    other = None
    if another_primary:
        other, other_ai = create_flyconid(Rng(111))
        combat.add_enemy(other, other_ai)
    combat.start_combat()
    return combat, fogmog, eye, other


def test_last_primary_death_kills_eye_minion_and_wins() -> None:
    """Game CreatureCmd.Kill cascades to Minions after the final primary dies."""
    combat, fogmog, eye, _ = _fogmog_combat()

    assert eye.has_power(PowerId.MINION)
    assert combat.kill_creature(fogmog)

    assert fogmog.is_dead
    assert eye.is_dead
    assert combat.is_over
    assert combat.player_won


def test_minion_survives_until_all_primary_enemies_die() -> None:
    combat, fogmog, eye, other = _fogmog_combat(another_primary=True)
    assert other is not None

    assert combat.kill_creature(fogmog)
    assert other.is_alive
    assert eye.is_alive
    assert not combat.is_over

    assert combat.kill_creature(other)
    assert eye.is_dead
    assert combat.is_over
    assert combat.player_won


def test_combat_end_checks_primary_enemies_not_all_living_enemies() -> None:
    combat, fogmog, eye, _ = _fogmog_combat()
    # Isolate CombatManager.IsCombatEnding from CreatureCmd.Kill's Minion
    # death cascade: the last primary is gone, but a secondary is still alive.
    fogmog.current_hp = 0
    fogmog.escaped = True

    combat._check_combat_end()  # noqa: SLF001

    assert eye.is_alive
    assert combat.is_over
    assert combat.player_won
