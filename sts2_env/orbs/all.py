"""All 5 Defect orb types: Lightning, Frost, Dark, Plasma, Glass.

Based on MegaCrit.Sts2.Core.Models.Orbs and GAME_SYSTEMS_REFERENCE.md Section 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.core.enums import OrbType, PowerId, ValueProp
from sts2_env.orbs.base import OrbInstance

if TYPE_CHECKING:
    from sts2_env.core.creature import Creature
    from sts2_env.core.combat import CombatState


def _gain_unpowered_block(owner: Creature, amount: int, combat: CombatState) -> int:
    before = owner.block
    owner.gain_block(amount, unpowered=True)
    gained = owner.block - before
    if gained > 0:
        from sts2_env.core.hooks import fire_after_block_gained

        fire_after_block_gained(owner, gained, combat)
    return gained


# ---------------------------------------------------------------------------
#  Lightning Orb
# ---------------------------------------------------------------------------

class LightningOrb(OrbInstance):
    """Lightning orb: passive deals 3 damage to random enemy, evoke deals 8.

    Focus modifies both values.
    Triggers: BeforeTurnEnd.
    """

    orb_type = OrbType.LIGHTNING

    @property
    def base_passive(self) -> int:
        return 3

    @property
    def base_evoke(self) -> int:
        return 8

    def on_passive(self, combat: CombatState) -> None:
        from sts2_env.core.damage import apply_damage
        value = self.get_passive_value(combat)
        if value <= 0:
            return
        alive = combat.hittable_enemies
        if alive:
            target = combat.combat_targets_rng.choice(alive)
            apply_damage(target, value, ValueProp.UNPOWERED, combat, combat.player)

    def on_evoke(self, combat: CombatState) -> list[Creature]:
        from sts2_env.core.damage import apply_damage
        value = self.get_evoke_value(combat)
        if value <= 0:
            return []
        alive = combat.hittable_enemies
        if alive:
            target = combat.combat_targets_rng.choice(alive)
            apply_damage(target, value, ValueProp.UNPOWERED, combat, combat.player)
            return [target]
        return []


# ---------------------------------------------------------------------------
#  Frost Orb
# ---------------------------------------------------------------------------

class FrostOrb(OrbInstance):
    """Frost orb: passive gives 2 Block to self, evoke gives 5 Block.

    Focus modifies both values.
    Triggers: BeforeTurnEnd.
    """

    orb_type = OrbType.FROST

    @property
    def base_passive(self) -> int:
        return 2

    @property
    def base_evoke(self) -> int:
        return 5

    def on_passive(self, combat: CombatState) -> None:
        value = self.get_passive_value(combat)
        if value > 0:
            targets = (
                [state.creature for state in combat.combat_player_states if state.creature.is_alive]
                if combat.player.get_power_amount(PowerId.HIBERNATE) > 0
                else [combat.player]
            )
            for target in targets:
                _gain_unpowered_block(target, value, combat)

    def on_evoke(self, combat: CombatState) -> list[Creature]:
        value = self.get_evoke_value(combat)
        targets = (
            [state.creature for state in combat.combat_player_states if state.creature.is_alive]
            if combat.player.get_power_amount(PowerId.HIBERNATE) > 0
            else [combat.player]
        )
        if value > 0:
            for target in targets:
                _gain_unpowered_block(target, value, combat)
        return targets


# ---------------------------------------------------------------------------
#  Dark Orb
# ---------------------------------------------------------------------------

class DarkOrb(OrbInstance):
    """Dark orb: passive adds +6 to its evoke value each turn (stacking).

    Evoke deals accumulated value to lowest-HP enemy.
    Starts at 6 evoke; gains PassiveVal each turn.
    Focus modifies the passive gain amount.
    Triggers: BeforeTurnEnd.
    """

    orb_type = OrbType.DARK

    def __init__(self) -> None:
        super().__init__()
        self._accumulated_evoke: int = 6

    @property
    def base_passive(self) -> int:
        return 6

    @property
    def base_evoke(self) -> int:
        return 6

    def get_evoke_value(self, combat: CombatState) -> int:
        # Dark orb uses accumulated value, not base + focus
        return max(0, self._accumulated_evoke)

    def on_passive(self, combat: CombatState) -> None:
        # Add (base_passive + focus) to accumulated evoke
        gain = self.base_passive + combat.player.get_power_amount(PowerId.FOCUS)
        gain = max(0, gain)
        self._accumulated_evoke += gain

    def on_evoke(self, combat: CombatState) -> list[Creature]:
        from sts2_env.core.damage import apply_damage
        value = self.get_evoke_value(combat)
        if value <= 0:
            return []
        hittable = combat.hittable_enemies
        if not hittable:
            return []
        target = min(hittable, key=lambda e: e.current_hp)
        apply_damage(target, value, ValueProp.UNPOWERED, combat, combat.player)
        return [target]


# ---------------------------------------------------------------------------
#  Plasma Orb
# ---------------------------------------------------------------------------

class PlasmaOrb(OrbInstance):
    """Plasma orb: passive gives +1 Energy, evoke gives +2 Energy.

    Values NOT modified by Focus.
    Triggers: AfterTurnStart.
    """

    orb_type = OrbType.PLASMA

    @property
    def base_passive(self) -> int:
        return 1

    @property
    def base_evoke(self) -> int:
        return 2

    def get_passive_value(self, combat: CombatState) -> int:
        # Plasma does NOT use Focus
        return self.base_passive

    def get_evoke_value(self, combat: CombatState) -> int:
        # Plasma does NOT use Focus
        return self.base_evoke

    def on_passive(self, combat: CombatState) -> None:
        combat.energy += self.get_passive_value(combat)

    def on_evoke(self, combat: CombatState) -> list[Creature]:
        combat.energy += self.get_evoke_value(combat)
        return [combat.player]


# ---------------------------------------------------------------------------
#  Glass Orb
# ---------------------------------------------------------------------------

class GlassOrb(OrbInstance):
    """Glass orb: passive deals 4 damage to all enemies (decreases by 1 each trigger).

    Evoke deals passive_val * 2 to all enemies.
    Focus modifies values. Decaying passive.
    Triggers: BeforeTurnEnd.
    """

    orb_type = OrbType.GLASS

    def __init__(self) -> None:
        super().__init__()
        self._current_passive: int = 4

    @property
    def base_passive(self) -> int:
        return self._current_passive

    @property
    def base_evoke(self) -> int:
        return self._current_passive * 2

    def on_passive(self, combat: CombatState) -> None:
        from sts2_env.core.damage import apply_damage
        value = self.get_passive_value(combat)
        if value > 0:
            for enemy in combat.hittable_enemies:
                apply_damage(enemy, value, ValueProp.UNPOWERED, combat, combat.player)
        # Decay: reduce by 1 each trigger
        self._current_passive = max(0, self._current_passive - 1)

    def on_evoke(self, combat: CombatState) -> list[Creature]:
        from sts2_env.core.damage import apply_damage
        value = self.get_evoke_value(combat)
        targets = []
        if value > 0:
            for enemy in combat.hittable_enemies:
                apply_damage(enemy, value, ValueProp.UNPOWERED, combat, combat.player)
                targets.append(enemy)
        return targets


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

_ORB_CLASSES: dict[OrbType, type[OrbInstance]] = {
    OrbType.LIGHTNING: LightningOrb,
    OrbType.FROST: FrostOrb,
    OrbType.DARK: DarkOrb,
    OrbType.PLASMA: PlasmaOrb,
    OrbType.GLASS: GlassOrb,
}


def create_orb(orb_type: OrbType) -> OrbInstance:
    """Create a new orb instance of the given type."""
    cls = _ORB_CLASSES[orb_type]
    return cls()
