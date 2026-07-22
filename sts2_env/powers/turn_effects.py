"""Turn-start and turn-end effect powers.

Implements powers that trigger effects at the start or end of a turn,
verified against the decompiled C# source from MegaCrit.Sts2.Core.Models.Powers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.core.enums import (
    CardType,
    CombatSide,
    OrbType,
    PowerId,
    PowerType,
    PowerStackType,
    ValueProp,
)
from sts2_env.powers.base import PowerInstance

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


# =====================================================================
#  Turn-start powers
# =====================================================================


class DemonFormPower(PowerInstance):
    """Gain Amount Strength at the start of each turn.

    C# hook: AfterSideTurnStart (side == Owner.Side)
    Applies StrengthPower(Amount) to owner.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DEMON_FORM, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)


class RitualPower(PowerInstance):
    """Gain Amount Strength at end of owner's turn.

    C# hook: AfterTurnEnd (side == Owner.Side).
    Enemy-applied Ritual skips its first tick (WasJustAppliedByEnemy flag).
    In our simulator this is modeled with skip_next_tick.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.RITUAL, amount)
        self._initial_skip_set: bool = False

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if (
            owner is target
            and power_id == PowerId.RITUAL
            and amount > 0
            and owner.side == CombatSide.ENEMY
            and not self._initial_skip_set
        ):
            self.skip_next_tick = True
            self._initial_skip_set = True

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            if self.skip_next_tick:
                self.skip_next_tick = False
                return
            owner.apply_power(PowerId.STRENGTH, self.amount, applier=owner)


class RegenPower(PowerInstance):
    """Heal Amount HP at end of turn, then decrement.

    C# hook: AfterTurnEnd (side == Owner.Side).
    Heals owner, then decrements by 1. Removes at 0.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    should_scale_in_multiplayer = True

    def __init__(self, amount: int):
        super().__init__(PowerId.REGEN, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side and owner.is_alive:
            owner.heal(self.amount)
            self.amount -= 1


class NoxiousFumesPower(PowerInstance):
    """Apply Amount Poison to all enemies at the start of owner's turn.

    C# hook: AfterSideTurnStart (side == Owner.Side).
    Applies PoisonPower(Amount) to all hittable enemies.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.NOXIOUS_FUMES, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        for enemy in combat.hittable_enemies:
            combat.apply_power_to(enemy, PowerId.POISON, self.amount, applier=owner)


class CreativeAiPower(PowerInstance):
    """Generate Amount random Power card(s) into hand at start of turn.

    C# hook: BeforeHandDraw (player == Owner.Player).
    In the simulator we add a random power card to hand.
    The combat state handles the actual card generation.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CREATIVE_AI, amount)

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            for _ in range(self.amount):
                combat.generate_card_to_hand(owner, card_type=CardType.POWER)


class StormPower(PowerInstance):
    """Channel Amount Lightning orb(s) when a Power card is played.

    C# hook: AfterCardPlayed — checks card.Type == Power.
    Channels Lightning orb for each stack.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.STORM, amount)
        self._amounts_for_played_cards: dict[int, int] = {}

    def before_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        if getattr(card, "owner", None) is not owner:
            return
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        if card_type == CardType.POWER:
            self._amounts_for_played_cards[id(card)] = self.amount

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        amount = self._amounts_for_played_cards.pop(id(card), 0)
        for _ in range(max(0, amount)):
            combat.channel_orb(owner, OrbType.LIGHTNING)


class DrawCardsNextTurnPower(PowerInstance):
    """Draw Amount extra cards next turn, then remove self.

    C# hooks: ModifyHandDraw (adds Amount) + AfterSideTurnStart (removes self).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DRAW_CARDS_NEXT_TURN, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        if self.amount != 0:
            return draw + self.amount
        return draw

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side and self.amount != 0:
            # Remove self after the extra draw has been applied
            self.amount = 0


class EnergyNextTurnPower(PowerInstance):
    """Gain Amount extra energy next turn, then remove self.

    C# hook: AfterEnergyReset (player == Owner.Player).
    Mapped to after_side_turn_start for the simulator.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ENERGY_NEXT_TURN, amount)

    def after_energy_reset(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.gain_energy(owner, self.amount)
            self.amount = 0

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if (
            side == owner.side
            and owner.is_player
            and not combat.has_energy_reset_this_turn(owner)
        ):
            self.after_energy_reset(owner, combat)


class StarNextTurnPower(PowerInstance):
    """Gain Amount stars next turn, then remove self.

    C# hook: AfterEnergyReset (player == Owner.Player).
    Stars are Regent-specific resource; mapped to combat.gain_stars.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.STAR_NEXT_TURN, amount)

    def after_energy_reset(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.gain_stars(owner, self.amount)
            self.amount = 0

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if (
            side == owner.side
            and owner.is_player
            and not combat.has_energy_reset_this_turn(owner)
        ):
            self.after_energy_reset(owner, combat)


class SummonNextTurnPower(PowerInstance):
    """Summon Amount Osty(s) next turn, then remove self.

    C# hook: AfterPlayerTurnStart.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.SUMMON_NEXT_TURN, amount)

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player and self.amount != 0:
            combat.summon_osty(owner, self.amount)
            self.amount = 0


class ToolsOfTheTradePower(PowerInstance):
    """Draw Amount extra cards at turn start, then discard Amount cards.

    C# hooks: ModifyHandDraw (adds Amount) + AfterPlayerTurnStart (discard Amount).
    The discard selection is handled by the combat/agent layer.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.TOOLS_OF_THE_TRADE, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        return draw + self.amount

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.request_discard(owner, self.amount)


class BurstPower(PowerInstance):
    """Next Skill is played an additional time. Decrements per use, removed at end of turn.

    C# hooks: ModifyCardPlayCount (card.Type == Skill => +1),
              AfterModifyingCardPlayCount (decrement),
              AfterTurnEnd (remove).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BURST, amount)

    def modify_card_play_count(self, owner: Creature, count: int, card: object) -> int:
        if self.amount <= 0:
            return count
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        card_owner = getattr(card, "owner", None)
        if card_owner is not owner:
            return count
        if card_type != CardType.SKILL:
            return count
        return count + 1

    def after_modifying_card_play_count(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        if getattr(card, "owner", None) is owner and card_type == CardType.SKILL and self.amount > 0:
            self.amount -= 1

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount = 0


class DuplicationPower(PowerInstance):
    """Next card is played an additional time. Decrements per use, removed at end of turn.

    C# hooks: ModifyCardPlayCount (any card => +1),
              AfterModifyingCardPlayCount (decrement),
              AfterTurnEnd (remove).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DUPLICATION, amount)

    def modify_card_play_count(self, owner: Creature, count: int, card: object) -> int:
        if self.amount <= 0:
            return count
        card_owner = getattr(card, "owner", None)
        if card_owner is not owner:
            return count
        return count + 1

    def after_modifying_card_play_count(self, owner: Creature, card: object, combat: CombatState) -> None:
        if getattr(card, "owner", None) is owner and self.amount > 0:
            self.amount -= 1

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount = 0


class MayhemPower(PowerInstance):
    """Auto-play Amount card(s) from top of draw pile at start of turn.

    C# hook: BeforeHandDrawLate (player == Owner.Player).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.MAYHEM, amount)

    def before_hand_draw_late(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.auto_play_from_draw(owner, self.amount)


class HelloWorldPower(PowerInstance):
    """Generate Amount random Common card(s) into hand at start of turn.

    C# hook: BeforeHandDraw (player == Owner.Player).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.HELLO_WORLD, amount)

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if not owner.is_player or self.amount <= 0:
            return
        from sts2_env.cards.factory import create_distinct_character_cards

        state = combat.combat_player_state_for(owner)
        if state is None:
            return
        generated = create_distinct_character_cards(
            state.character_id,
            combat.combat_card_generation_rng,
            self.amount,
            rarity="COMMON",
            generation_context="combat",
            is_multiplayer=combat.is_multiplayer,
        )
        combat._add_generated_cards_to_hand(generated, owner=owner)


class MachineLearningPower(PowerInstance):
    """Draw Amount extra cards each turn (permanent).

    C# hook: ModifyHandDraw (player == Owner.Player).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.MACHINE_LEARNING, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        return draw + self.amount


class LoopPower(PowerInstance):
    """Trigger the passive of the first orb Amount time(s) at start of turn.

    C# hook: AfterPlayerTurnStart.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.LOOP, amount)

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            for _ in range(self.amount):
                combat.trigger_first_orb_passive(owner)


class PanachePower(PowerInstance):
    """Every 5th card played deals Amount damage to all enemies.

    C# hooks: AfterCardPlayed (track cards, deal damage every 5th),
              AfterTurnEnd (reset counter).
    Uses internal counter; display amount shows cards remaining.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    _TRIGGER_EVERY = 5

    def __init__(self, amount: int):
        super().__init__(PowerId.PANACHE, amount)
        self._cards_left: int = self._TRIGGER_EVERY
        self._started: bool = False
        self._instances: list[tuple[int, int, bool]] = [(amount, self._TRIGGER_EVERY, False)]

    def _add_instance(self, amount: int) -> None:
        self._instances.append((amount, self._TRIGGER_EVERY, False))
        self.amount = amount

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount > 0 and self.amount != amount:
            self._add_instance(amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_owner = getattr(card, "owner", None)
        if card_owner is not None and card_owner is not owner:
            return
        updated_instances: list[tuple[int, int, bool]] = []
        for amount, cards_left, started in self._instances:
            if started:
                cards_left -= 1
            if cards_left <= 0:
                for enemy in combat.hittable_enemies:
                    combat.deal_damage(
                        dealer=owner,
                        target=enemy,
                        amount=amount,
                        props=ValueProp.UNPOWERED,
                    )
                cards_left = self._TRIGGER_EVERY
            updated_instances.append((amount, cards_left, True))
        self._instances = updated_instances
        if self._instances:
            _, self._cards_left, self._started = self._instances[-1]

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self._cards_left = self._TRIGGER_EVERY
            self._instances = [
                (amount, self._TRIGGER_EVERY, started)
                for amount, _, started in self._instances
            ]


class InfiniteBladesPower(PowerInstance):
    """Add Amount Shiv(s) to hand at the start of each turn.

    C# hook: BeforeHandDraw (player == Owner.Player).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.INFINITE_BLADES, amount)

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.add_shivs_to_hand(owner, self.amount)


class FanOfKnivesPower(PowerInstance):
    """Marker power (Single stack). No hook logic in C# — a flag checked by FanOfKnives card.

    C# source: StackType = Single, no hooks.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.FAN_OF_KNIVES, amount)


class DrumOfBattlePower(PowerInstance):
    """Exhaust Amount card(s) from the top of the draw pile at start of turn.

    C# hook: BeforeHandDrawLate (shuffles if needed, exhausts top card).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DRUM_OF_BATTLE, amount)

    def before_hand_draw_late(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.exhaust_from_draw_pile(owner, self.amount)


# =====================================================================
#  Turn-end powers
# =====================================================================


class PoisonPower(PowerInstance):
    """Deal Amount unblockable damage at start of owner's turn, then decrement.

    C# hook: AfterSideTurnStart (side == Owner.Side).
    Deals damage = Amount, then decrements by 1. Repeats according to
    opposing Accelerant stacks, capped by current poison amount.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.POISON, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        if owner.is_alive and self.amount > 0:
            opponents = [creature for creature in combat.get_enemies_of(owner) if creature.is_alive]
            trigger_count = min(
                self.amount,
                1 + sum(opponent.get_power_amount(PowerId.ACCELERANT) for opponent in opponents),
            )
            for _ in range(trigger_count):
                if not owner.is_alive or self.amount <= 0:
                    break
                current_amount = self.amount
                combat.deal_damage(
                    dealer=None,
                    target=owner,
                    amount=current_amount,
                    props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
                )
                if owner.is_alive:
                    self.amount -= 1


class ConstrictPower(PowerInstance):
    """Deal Amount damage to owner at end of their turn.

    C# hook: AfterTurnEnd (side == Owner.Side).
    Damage is unpowered. Removed if the applier dies.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CONSTRICT, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.deal_damage(
                dealer=owner,
                target=owner,
                amount=self.amount,
                props=ValueProp.UNPOWERED,
            )

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if not was_removal_prevented and creature is self.applier:
            owner.powers.pop(self.power_id, None)


class TheBombPower(PowerInstance):
    """Countdown timer: decrement each turn, deal 40 damage to all enemies when it hits 0.

    C# hook: BeforeTurnEnd (side == Owner.Side).
    Damage default is 40 (unpowered). Amount is the countdown turns.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    DEFAULT_DAMAGE = 40

    def __init__(self, amount: int):
        super().__init__(PowerId.THE_BOMB, amount)
        self.damage: int = self.DEFAULT_DAMAGE
        self._instances: list[tuple[int, int]] = [(amount, self.damage)]

    def add_instance(self, turns: int, damage: int) -> None:
        self._instances.append((turns, damage))
        self.amount = turns
        self.damage = damage

    def set_latest_damage(self, damage: int) -> None:
        turns, _ = self._instances[-1]
        self._instances[-1] = (turns, damage)
        self.damage = damage

    def before_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        remaining_instances: list[tuple[int, int]] = []
        for turns, damage in self._instances:
            if turns > 1:
                remaining_instances.append((turns - 1, damage))
                continue
            for enemy in combat.hittable_enemies:
                combat.deal_damage(
                    dealer=owner,
                    target=enemy,
                    amount=damage,
                    props=ValueProp.UNPOWERED,
                )
        self._instances = remaining_instances
        if remaining_instances:
            self.amount, self.damage = remaining_instances[-1]
        else:
            self.amount = 0


class TemporaryStrengthPower(PowerInstance):
    """Temporary Strength: removed at end of turn, reversing the Strength gain.

    C# hook: AfterTurnEnd (side == Owner.Side).
    On removal, applies -Amount Strength to owner.
    The initial Strength was granted when this power was first applied
    (BeforeApplied applies StrengthPower of same amount).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.TEMPORARY_STRENGTH, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, -self.amount, applier=owner)
            self.amount = 0


class FlexPotionPower(TemporaryStrengthPower):
    """Flex Potion: identical to TemporaryStrengthPower (positive temporary strength)."""

    def __init__(self, amount: int):
        # Call PowerInstance.__init__ directly to set the correct PowerId
        PowerInstance.__init__(self, PowerId.FLEX_POTION, amount)


class ShacklingPotionPower(PowerInstance):
    """Shackling Potion: negative temporary Strength (debuff version).

    Applies -Amount Strength on application. At end of turn, removes self
    and reverses the debuff by applying +Amount Strength.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.SHACKLING_POTION, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            # Reverse: was applied as -Amount STR, now restore +Amount STR
            owner.apply_power(PowerId.STRENGTH, self.amount, applier=owner)
            self.amount = 0


class TemporaryDexterityPower(PowerInstance):
    """Temporary Dexterity: removed at end of turn, reversing the Dexterity gain.

    C# hook: AfterTurnEnd (side == Owner.Side).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.TEMPORARY_DEXTERITY, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.DEXTERITY, amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.DEXTERITY, -self.amount, applier=owner)
            self.amount = 0


class SpeedPotionPower(TemporaryDexterityPower):
    """Speed Potion: identical to TemporaryDexterityPower (positive temporary dexterity)."""

    def __init__(self, amount: int):
        PowerInstance.__init__(self, PowerId.SPEED_POTION, amount)


class TemporaryFocusPower(PowerInstance):
    """Temporary Focus: removed at end of turn, reversing the Focus gain.

    C# hook: AfterTurnEnd (side == Owner.Side).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.TEMPORARY_FOCUS, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.FOCUS, amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.FOCUS, -self.amount, applier=owner)
            self.amount = 0


class BlockNextTurnPower(PowerInstance):
    """Gain Amount block when block is cleared (next turn start), then remove self.

    C# hook: AfterBlockCleared (creature == Owner).
    Gains unpowered block then removes self.
    We map this to after_side_turn_start since block clearing happens
    at turn start in the sim pipeline.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BLOCK_NEXT_TURN, amount)

    def after_block_cleared(self, owner: Creature, creature: Creature, combat: CombatState) -> None:
        if creature is owner and self.amount > 0:
            _gain_unpowered_block(owner, self.amount, combat)
            self.amount = 0

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side and owner.block == 0 and self.amount > 0:
            _gain_unpowered_block(owner, self.amount, combat)
            self.amount = 0


class RetainHandPower(PowerInstance):
    """Retain entire hand for Amount turn(s).

    C# hooks: ShouldFlush (returns false for owner's player),
              AfterTurnEnd (decrement).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.RETAIN_HAND, amount)

    def should_flush(self, owner: Creature, flushing_owner: Creature, combat: CombatState) -> bool | None:
        if flushing_owner is owner:
            return False
        return None

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount -= 1


class WellLaidPlansPower(PowerInstance):
    """Retain up to Amount card(s) at end of turn.

    C# hook: BeforeFlushLate — lets the player choose cards to retain.
    In the simulator, this is handled by the combat layer; this power
    just exposes the retain count.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int):
        super().__init__(PowerId.WELL_LAID_PLANS, amount)

    def should_flush(self, owner: Creature, flushing_owner: Creature, combat: CombatState) -> bool | None:
        if flushing_owner is owner and owner.is_player:
            return False
        return None


class WraithFormPower(PowerInstance):
    """Lose Amount Dexterity at the start of each turn.

    C# hook: AfterSideTurnStart (side == Owner.Side).
    Applies DexterityPower(-Amount).
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.WRAITH_FORM, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.apply_power_to(owner, PowerId.DEXTERITY, -self.amount, applier=owner)


class BiasedCognitionPower(PowerInstance):
    """Lose Amount Focus at the start of each turn.

    C# hook: AfterSideTurnStart (side == Owner.Side).
    Applies FocusPower(-Amount).
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BIASED_COGNITION, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.apply_power_to(owner, PowerId.FOCUS, -self.amount, applier=owner)


class CalcifyPower(PowerInstance):
    """Osty (pet) attacks deal Amount extra damage.

    C# hook: ModifyDamageAdditive — adds Amount when dealer is an Osty
    owned by the power's owner. Regent-specific power.
    In the simulator, we model this as a simple damage boost for pet attacks.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CALCIFY, amount)

    def modify_damage_additive(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> int:
        if (
            dealer is not None
            and getattr(dealer, "is_osty", False)
            and getattr(dealer, "pet_owner", None) is owner
            and props.is_powered_attack()
        ):
            return self.amount
        return 0


class CountdownPower(PowerInstance):
    """Apply Amount Doom to a random enemy at start of turn.

    C# hook: AfterSideTurnStart (side == Owner.Side).
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.COUNTDOWN, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            target = combat.random_enemy_of(owner)
            if target is not None:
                combat.apply_power_to(target, PowerId.DOOM, self.amount, applier=owner)


class DoomPower(PowerInstance):
    """At end of turn, if owner's HP <= Doom amount, kill the owner.

    C# hook: BeforeTurnEnd (side == Owner.Side).
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DOOM, amount)

    def before_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side or not owner.is_alive or owner.current_hp > self.amount:
            return
        side_creatures = (
            [combat.primary_player, *combat.alive_allies]
            if side == CombatSide.PLAYER
            else list(combat.alive_enemies)
        )
        doomed = [
            creature for creature in side_creatures
            if creature.is_alive
            and creature.get_power_amount(PowerId.DOOM) > 0
            and creature.current_hp <= creature.get_power_amount(PowerId.DOOM)
        ]
        if doomed and doomed[0] is owner:
            combat.kill_doomed_creatures(doomed)


# =====================================================================
#  Registration
# =====================================================================

from sts2_env.core.creature import register_power_class

_ALL_POWERS: dict[PowerId, type[PowerInstance]] = {
    PowerId.DEMON_FORM: DemonFormPower,
    PowerId.RITUAL: RitualPower,
    PowerId.REGEN: RegenPower,
    PowerId.NOXIOUS_FUMES: NoxiousFumesPower,
    PowerId.CREATIVE_AI: CreativeAiPower,
    PowerId.STORM: StormPower,
    PowerId.DRAW_CARDS_NEXT_TURN: DrawCardsNextTurnPower,
    PowerId.ENERGY_NEXT_TURN: EnergyNextTurnPower,
    PowerId.STAR_NEXT_TURN: StarNextTurnPower,
    PowerId.SUMMON_NEXT_TURN: SummonNextTurnPower,
    PowerId.TOOLS_OF_THE_TRADE: ToolsOfTheTradePower,
    PowerId.BURST: BurstPower,
    PowerId.DUPLICATION: DuplicationPower,
    PowerId.MAYHEM: MayhemPower,
    PowerId.HELLO_WORLD: HelloWorldPower,
    PowerId.MACHINE_LEARNING: MachineLearningPower,
    PowerId.LOOP: LoopPower,
    PowerId.PANACHE: PanachePower,
    PowerId.INFINITE_BLADES: InfiniteBladesPower,
    PowerId.FAN_OF_KNIVES: FanOfKnivesPower,
    PowerId.DRUM_OF_BATTLE: DrumOfBattlePower,
    PowerId.POISON: PoisonPower,
    PowerId.CONSTRICT: ConstrictPower,
    PowerId.THE_BOMB: TheBombPower,
    PowerId.TEMPORARY_STRENGTH: TemporaryStrengthPower,
    PowerId.FLEX_POTION: FlexPotionPower,
    PowerId.SHACKLING_POTION: ShacklingPotionPower,
    PowerId.TEMPORARY_DEXTERITY: TemporaryDexterityPower,
    PowerId.SPEED_POTION: SpeedPotionPower,
    PowerId.TEMPORARY_FOCUS: TemporaryFocusPower,
    PowerId.BLOCK_NEXT_TURN: BlockNextTurnPower,
    PowerId.RETAIN_HAND: RetainHandPower,
    PowerId.WELL_LAID_PLANS: WellLaidPlansPower,
    PowerId.WRAITH_FORM: WraithFormPower,
    PowerId.BIASED_COGNITION: BiasedCognitionPower,
    PowerId.CALCIFY: CalcifyPower,
    PowerId.COUNTDOWN: CountdownPower,
    PowerId.DOOM: DoomPower,
}

for _pid, _cls in _ALL_POWERS.items():
    register_power_class(_pid, _cls)
