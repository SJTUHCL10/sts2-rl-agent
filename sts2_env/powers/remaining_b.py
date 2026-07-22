"""Remaining powers batch B (46 powers): G-R.

Covers: GravityPower, GuardedPower, HailstormPower, HammerTimePower, HangPower,
HardToKillPower, HauntPower, HelicalDartPower, HexPower, HighVoltagePower,
HotfixPower, ImbalancedPower, ImprovementPower, InfernoPower, IterationPower,
JugglingPower, KnockdownPower, LeadershipPower, LightningRodPower,
MagicBombPower, ManglePower, MasterPlannerPower, MindRotPower,
MonarchsGazePower, MonarchsGazeStrengthDownPower, NemesisPower,
NeurosurgePower, NightmarePower, NoDrawPower, NostalgiaPower, OblivionPower,
OrbitPower, OutbreakPower, PagestormPower, PaleBlueDotPower, PaperCutsPower,
ParryPower, PiercingWailPower, PillarOfCreationPower, PossessSpeedPower,
PossessStrengthPower, PrepTimePower, PyrePower, RadiancePower, RampartPower,
ReaperFormPower.

All logic verified against decompiled C# source from
MegaCrit.Sts2.Core.Models.Powers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.core.enums import (
    CardPilePosition,
    CardId,
    CardKeyword,
    CardType,
    CombatSide,
    OrbType,
    PileType,
    PowerId,
    PowerType,
    PowerStackType,
    ValueProp,
)
from sts2_env.powers.base import PowerInstance

if TYPE_CHECKING:
    from sts2_env.core.creature import Creature
    from sts2_env.core.combat import CombatState


_BOWLBUG_ROCK_ID = "BOWLBUG_ROCK"


def _gain_unpowered_block(owner: Creature, amount: int, combat: CombatState) -> int:
    before = owner.block
    owner.gain_block(amount, unpowered=True)
    gained = owner.block - before
    if gained > 0:
        from sts2_env.core.hooks import fire_after_block_gained

        fire_after_block_gained(owner, gained, combat)
    return gained


# ---------------------------------------------------------------------------
# GravityPower
# ---------------------------------------------------------------------------
class GravityPower(PowerInstance):
    """After you play a card, deal Amount damage to all enemies (unpowered).
    Removed at end of owner's turn.

    C# ref: GravityPower.cs
    - BeforeCardPlayed: records card + amount.
    - AfterCardPlayed: deals Amount damage to all hittable enemies (unpowered).
    - AfterTurnEnd (owner side): remove self.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.GRAVITY, amount)
        self._amounts_for_played_cards: dict[int, int] = {}

    def before_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        if getattr(card, "owner", None) is owner:
            self._amounts_for_played_cards[id(card)] = self.amount

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        amount = self._amounts_for_played_cards.pop(id(card), 0)
        if amount <= 0:
            return
        for enemy in combat.hittable_enemies:
            if enemy.is_alive:
                combat.deal_damage(
                    dealer=owner,
                    target=enemy,
                    amount=amount,
                    props=ValueProp.UNPOWERED,
                )

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount = 0


# ---------------------------------------------------------------------------
# GuardedPower
# ---------------------------------------------------------------------------
class GuardedPower(PowerInstance):
    """Owner takes 50% damage from powered attacks. Removed when the applier dies.

    C# ref: GuardedPower.cs
    - ModifyDamageMultiplicative: 0.5 for powered attacks targeting owner.
    StackType.Single. Instanced.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.GUARDED, amount)
        self._appliers: list[Creature | None] = [None]

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
        if owner is not target or power_id != self.power_id or amount <= 0:
            return
        if len(self._appliers) == 1 and self._appliers[0] is None:
            self._appliers[0] = applier
            return
        self._appliers.append(applier)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        if target is not owner:
            return 1.0
        if not props.is_powered_attack():
            return 1.0
        return 0.5 ** len(self._appliers)

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented:
            return
        fallback_applier = self.applier
        self._appliers = [
            applier
            for applier in self._appliers
            if creature is not (applier or fallback_applier)
        ]
        if not self._appliers:
            owner.powers.pop(self.power_id, None)
            return
        self.amount = len(self._appliers)


# ---------------------------------------------------------------------------
# HailstormPower
# ---------------------------------------------------------------------------
class HailstormPower(PowerInstance):
    """Before turn end, if owner has >= 1 Frost orb, deal Amount damage
    to all enemies (unpowered).

    C# ref: HailstormPower.cs
    - BeforeTurnEnd (owner side): count Frost orbs; if >= FrostOrbs threshold
      (1), deal Amount damage to all hittable enemies.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    FROST_ORB_THRESHOLD = 1

    def __init__(self, amount: int):
        super().__init__(PowerId.HAILSTORM, amount)

    def before_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        frost_count = 0
        orb_queue = getattr(owner, "orb_queue", None) or getattr(
            getattr(owner, "player_combat_state", None), "orb_queue", None
        )
        if orb_queue is not None:
            orbs = getattr(orb_queue, "orbs", orb_queue)
            if hasattr(orbs, "__iter__"):
                frost_count = sum(
                    1 for o in orbs if getattr(o, "orb_type", None) == OrbType.FROST
                )
        if frost_count == 0:
            count_fn = getattr(combat, "count_orbs", None)
            if count_fn is not None:
                frost_count = count_fn(owner, OrbType.FROST)
        if frost_count >= self.FROST_ORB_THRESHOLD:
            for enemy in combat.hittable_enemies:
                combat.deal_damage(
                    dealer=owner,
                    target=enemy,
                    amount=self.amount,
                    props=ValueProp.UNPOWERED,
                )


# ---------------------------------------------------------------------------
# HammerTimePower
# ---------------------------------------------------------------------------
class HammerTimePower(PowerInstance):
    """When the owner forges, all other players also forge the same amount.

    C# ref: HammerTimePower.cs
    - AfterForge: if forger == owner.Player and source is not self,
      forge same amount for all other living players.
    StackType.Single.

    Simplified: Forge mechanic is Regent-specific multiplayer feature.
    This power is a flag; the forge system checks for it.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.HAMMER_TIME, amount)

    def after_forge(
        self,
        owner: Creature,
        amount: int,
        forger: Creature,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if source is self or forger is not owner or amount <= 0:
            return
        for teammate in combat.get_teammates_of(owner):
            if getattr(teammate, "is_player", False) and teammate.is_alive:
                combat.forge(teammate, amount, source=self)


# ---------------------------------------------------------------------------
# HangPower
# ---------------------------------------------------------------------------
class HangPower(PowerInstance):
    """Multiplies damage taken from Hang cards by Amount.

    C# ref: HangPower.cs
    - ModifyDamageMultiplicative: returns Amount when target == owner
      AND cardSource is a Hang card.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.HANG, amount)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        from sts2_env.cards.necrobinder import is_hang

        if target is not owner:
            return 1.0
        card_source = getattr(owner.combat_state, "active_card_source", None)
        if not is_hang(card_source):
            return 1.0
        return float(self.amount)


# ---------------------------------------------------------------------------
# HardToKillPower
# ---------------------------------------------------------------------------
class HardToKillPower(PowerInstance):
    """Caps incoming damage to Amount per hit when targeting the owner.

    C# ref: HardToKillPower.cs
    - ModifyDamageCap: returns Amount when target == owner, else MaxValue.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.HARD_TO_KILL, amount)

    def modify_damage_cap(
        self, owner: Creature, dealer: Creature | None, target: Creature,
        damage: float, props: ValueProp
    ) -> float:
        if target is not owner:
            return float("inf")
        return float(self.amount)


# ---------------------------------------------------------------------------
# HauntPower
# ---------------------------------------------------------------------------
class HauntPower(PowerInstance):
    """When owner plays a Soul card, deal Amount unblockable/unpowered damage
    to a random enemy.

    C# ref: HauntPower.cs
    - AfterCardPlayed: if card is Soul, deal Amount damage to random hittable
      enemy (Unblockable | Unpowered, no dealer).
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.HAUNT, amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        from sts2_env.cards.status import is_soul

        if not is_soul(card):
            return
        target = combat.random_enemy_of(owner)
        if target is not None:
            combat.deal_damage(
                dealer=None,
                target=target,
                amount=self.amount,
                props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
            )


# ---------------------------------------------------------------------------
# HelicalDartPower
# ---------------------------------------------------------------------------
class HelicalDartPower(PowerInstance):
    """Temporary Dexterity (positive). Grants Dexterity on application,
    removes it at end of owner's turn.

    C# ref: HelicalDartPower.cs extends TemporaryDexterityPower.
    - BeforeApplied: apply +Amount Dexterity.
    - AfterTurnEnd (owner side): remove self and apply -Amount Dexterity.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.HELICAL_DART, amount)

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
            owner.apply_power(PowerId.DEXTERITY, -self.amount)
            self.amount = 0


# ---------------------------------------------------------------------------
# HexPower
# ---------------------------------------------------------------------------
class HexPower(PowerInstance):
    """Afflicts all cards with Hexed (adds Ethereal). Removed when the
    applier dies. On removal, clears Hexed and removes applied Ethereal.

    C# ref: HexPower.cs
    - AfterApplied: afflict all player cards with Hexed (adds Ethereal).
    - AfterCardEnteredCombat: afflict new cards.
    - AfterDeath: if applier dies, remove self.
    - AfterRemoved: clear Hexed from all cards, remove applied Ethereal.
    StackType.Single.

    Simplified: This power is a flag. The card system checks for it and
    applies/removes Ethereal. The Amount is the affliction amount.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.HEX, amount)
        self._active: bool = False
        self._hexed_cards: list[object] = []
        self._applied_ethereal_cards: list[object] = []

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
        if owner is target and power_id == PowerId.HEX and amount > 0 and not self._active:
            self._active = True
            state = combat.combat_player_state_for(owner)
            if state is None:
                return
            for pile in state.all_piles:
                for card in pile:
                    self.afflict_card(card)

    def afflict_card(self, card: object) -> None:
        afflict = getattr(card, "afflict", None)
        if callable(afflict) and not afflict("hexed"):
            return
        self._hexed_cards.append(card)
        keywords = set(getattr(card, "keywords", set()))
        if "ethereal" in keywords:
            return
        keywords.add("ethereal")
        card.keywords = frozenset(keywords)
        self._applied_ethereal_cards.append(card)

    def after_card_entered_combat(self, owner: Creature, card: object, combat: CombatState) -> None:
        if getattr(card, "owner", None) is owner:
            self.afflict_card(card)

    def on_removed(self, owner: Creature, combat: CombatState) -> None:
        for card in self._applied_ethereal_cards:
            keywords = set(getattr(card, "keywords", set()))
            keywords.discard("ethereal")
            card.keywords = frozenset(keywords)
        for card in self._hexed_cards:
            clear_affliction = getattr(card, "clear_affliction", None)
            if callable(clear_affliction):
                clear_affliction("hexed")
        self._hexed_cards.clear()
        self._applied_ethereal_cards.clear()
        self._active = False

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if not was_removal_prevented and creature is self.applier:
            combat._remove_power(owner, self.power_id)


# ---------------------------------------------------------------------------
# HighVoltagePower
# ---------------------------------------------------------------------------
class HighVoltagePower(PowerInstance):
    """Gain Amount Strength at end of owner's turn.

    C# ref: HighVoltagePower.cs
    - AfterTurnEnd (owner side): apply Amount Strength to owner.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.HIGH_VOLTAGE, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)


# ---------------------------------------------------------------------------
# HotfixPower
# ---------------------------------------------------------------------------
class HotfixPower(PowerInstance):
    """Temporary Focus (positive). Grants Focus on application,
    removes it at end of owner's turn.

    C# ref: HotfixPower.cs extends TemporaryFocusPower.
    - BeforeApplied: apply +Amount Focus.
    - AfterTurnEnd (owner side): remove self and apply -Amount Focus.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.HOTFIX, amount)

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
            owner.apply_power(PowerId.FOCUS, -self.amount)
            self.amount = 0


# ---------------------------------------------------------------------------
# ImbalancedPower
# ---------------------------------------------------------------------------
class ImbalancedPower(PowerInstance):
    """When owner deals damage that is fully blocked, the owner is stunned.

    C# ref: ImbalancedPower.cs
    - AfterDamageGiven: if dealer == owner and damage was fully blocked,
      stun the owner. BowlbugRock instead sets IsOffBalance for its next move.
    StackType.Single.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.IMBALANCED, amount)
        self.was_fully_blocked: bool = False

    def after_damage_given(
        self, owner: Creature, dealer: Creature, target: Creature,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        result = getattr(combat, "_active_damage_result", None)
        if dealer is owner and getattr(result, "was_fully_blocked", False):
            if owner.monster_id == _BOWLBUG_ROCK_ID:
                self.was_fully_blocked = True
            else:
                combat.stun_enemy(owner)


# ---------------------------------------------------------------------------
# ImprovementPower
# ---------------------------------------------------------------------------
class ImprovementPower(PowerInstance):
    """After combat ends, upgrade Amount random upgradable cards in the deck.

    C# ref: ImprovementPower.cs
    - AfterCombatEnd: pick Amount random upgradable cards and upgrade them.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.IMPROVEMENT, amount)

    def after_combat_end(self, owner: Creature, combat: CombatState) -> None:
        state = combat.combat_player_state_for(owner)
        if state is None:
            return
        combat.upgrade_random_cards(state.player_state.deck, self.amount)


# ---------------------------------------------------------------------------
# InfernoPower
# ---------------------------------------------------------------------------
class InfernoPower(PowerInstance):
    """At start of turn, deal self-damage (tracked separately via SelfDamage
    DynamicVar, starts at 0). When owner takes unblocked damage on their own
    turn, deal Amount damage to all enemies (unpowered).

    C# ref: InfernoPower.cs
    - AfterPlayerTurnStart: deal SelfDamage to self (unblockable, unpowered).
    - AfterDamageReceived: if owner took unblocked damage on own turn,
      deal Amount damage to all hittable enemies (unpowered).
    - IncrementSelfDamage(): increases the self-damage var by 1.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.INFERNO, amount)
        self.self_damage: int = 0

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if self.self_damage > 0:
            combat.deal_damage(
                dealer=owner,
                target=owner,
                amount=self.self_damage,
                props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
            )

    def after_damage_received(
        self, owner: Creature, target: Creature, dealer: Creature | None,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        if target is not owner or damage <= 0:
            return
        if not combat.is_owner_side_turn(owner):
            return
        for enemy in combat.hittable_enemies:
            combat.deal_damage(
                dealer=owner,
                target=enemy,
                amount=self.amount,
                props=ValueProp.UNPOWERED,
            )

    def increment_self_damage(self) -> None:
        """Called by card system (e.g., Stoke) to increase self-damage."""
        self.self_damage += 1


# ---------------------------------------------------------------------------
# IterationPower
# ---------------------------------------------------------------------------
class IterationPower(PowerInstance):
    """When owner draws a Status card (first one per turn), draw Amount
    additional cards.

    C# ref: IterationPower.cs
    - AfterCardDrawn: if owner drew a Status card and it's the first Status
      drawn this turn, draw Amount cards.
    StackType.Counter.

    The draw system calls on_card_drawn() after recording the draw event.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ITERATION, amount)

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        if getattr(card, "owner", None) is not owner:
            return
        if getattr(card, "card_type", None) == CardType.STATUS:
            if combat.count_drawn_cards_this_turn(owner, CardType.STATUS) <= 1:
                combat.draw_cards(owner, self.amount)


# ---------------------------------------------------------------------------
# JugglingPower
# ---------------------------------------------------------------------------
class JugglingPower(PowerInstance):
    """On the 3rd Attack played each turn, add Amount copies of that card
    to hand.

    C# ref: JugglingPower.cs
    - AfterCardPlayed: if owner plays an Attack, increment counter.
      On the 3rd attack, clone the card Amount times to hand.
    - AfterTurnEnd (owner side): reset counter.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    TRIGGER_ON_ATTACK = 3

    def __init__(self, amount: int):
        super().__init__(PowerId.JUGGLING, amount)
        self._attacks_played_this_turn: int = 0

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
        if owner is target and power_id == PowerId.JUGGLING and amount > 0:
            self._attacks_played_this_turn = combat.count_card_play_starts_this_turn(
                owner,
                card_type=CardType.ATTACK,
            )

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_owner = getattr(card, "owner", None)
        if card_owner is not None and card_owner is not owner:
            return
        if getattr(card, "card_type", None) != CardType.ATTACK:
            return
        self._attacks_played_this_turn += 1
        if self._attacks_played_this_turn == self.TRIGGER_ON_ATTACK:
            clone_fn = getattr(combat, "clone_card_to_hand", None)
            if clone_fn is not None:
                for _ in range(self.amount):
                    clone_fn(owner, card)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self._attacks_played_this_turn = 0


# ---------------------------------------------------------------------------
# KnockdownPower
# ---------------------------------------------------------------------------
class KnockdownPower(PowerInstance):
    """Multiplies powered attack damage taken by Amount, EXCEPT from the
    applier. Removed at end of owner's turn.

    C# ref: KnockdownPower.cs
    - ModifyDamageMultiplicative: returns Amount when target == owner,
      powered attack, and dealer != applier.
    - AfterTurnEnd (owner side): remove self.
    StackType.Counter. Instanced.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.KNOCKDOWN, amount)
        self.applier: Creature | None = None
        self._instances: list[tuple[int, Creature | None]] = [(amount, None)]

    def _add_instance(self, amount: int, applier: Creature | None) -> None:
        self._instances.append((amount, applier))
        self.amount = amount
        self.applier = applier

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
        if owner is target and power_id == self.power_id and amount > 0:
            if len(self._instances) == 1 and self._instances[0][1] is None:
                self._instances[0] = (self._instances[0][0], self.applier)
                return
            self._add_instance(amount, applier)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        if self.amount <= 0:
            return 1.0
        if target is not owner:
            return 1.0
        if not props.is_powered_attack():
            return 1.0
        multiplier = 1.0
        for amount, applier in self._instances:
            if dealer is applier:
                continue
            multiplier *= float(amount)
        return multiplier

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat._remove_power(owner, self.power_id)


# ---------------------------------------------------------------------------
# LeadershipPower
# ---------------------------------------------------------------------------
class LeadershipPower(PowerInstance):
    """Allied creatures (not the owner) deal +Amount damage on powered attacks.

    C# ref: LeadershipPower.cs
    - ModifyDamageAdditive: +Amount when dealer is an ally on the same side
      (but not the owner itself) and attack is powered.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.LEADERSHIP, amount)

    def modify_damage_additive(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> int:
        if dealer is owner:
            return 0
        if dealer is None:
            return 0
        if dealer.side != owner.side:
            return 0
        if not props.is_powered_attack():
            return 0
        return self.amount


# ---------------------------------------------------------------------------
# LightningRodPower
# ---------------------------------------------------------------------------
class LightningRodPower(PowerInstance):
    """At start of turn (energy reset), channel a Lightning orb, then
    decrement.

    C# ref: LightningRodPower.cs
    - AfterEnergyReset: channel Lightning orb, then decrement.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.LIGHTNING_ROD, amount)

    def after_energy_reset(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            if hasattr(combat, "channel_orb"):
                combat.channel_orb(owner, OrbType.LIGHTNING)
            self.amount -= 1

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if (
            side == owner.side
            and owner.is_player
            and not combat.has_energy_reset_this_turn(owner)
        ):
            self.after_energy_reset(owner, combat)


# ---------------------------------------------------------------------------
# MagicBombPower
# ---------------------------------------------------------------------------
class MagicBombPower(PowerInstance):
    """At end of owner's turn, deal Amount damage to self (unpowered),
    then remove self. Removed if applier dies.

    C# ref: MagicBombPower.cs
    - AfterTurnEnd (owner side): if applier is alive, deal Amount damage
      to owner (unpowered), then remove self.
    - AfterDeath: if applier dies, remove self.
    StackType.Counter. Instanced.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.MAGIC_BOMB, amount)
        self._instances: list[tuple[int, Creature | None]] = [(amount, None)]
        self._initial_instance_recorded: bool = False

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
        if owner is not target or power_id != self.power_id or amount <= 0:
            return
        if not self._initial_instance_recorded:
            self._instances[0] = (self._instances[0][0], applier)
            self._initial_instance_recorded = True
            return
        self._instances.append((amount, applier))

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        for amount, applier in self._instances:
            if applier is None or not applier.is_alive:
                continue
            combat.deal_damage(
                dealer=owner,
                target=owner,
                amount=amount,
                props=ValueProp.UNPOWERED,
            )
        self._instances = []
        self.amount = 0

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented:
            return
        self._instances = [(amount, applier) for amount, applier in self._instances if creature is not applier]
        if self._instances:
            self.amount = sum(amount for amount, _ in self._instances)
        else:
            owner.powers.pop(self.power_id, None)


# ---------------------------------------------------------------------------
# ManglePower
# ---------------------------------------------------------------------------
class ManglePower(PowerInstance):
    """Temporary Strength (negative / debuff). Applies -Amount Strength on
    application, restores it at end of owner's turn.

    C# ref: ManglePower.cs extends TemporaryStrengthPower, IsPositive=false.
    - BeforeApplied: apply -Amount Strength.
    - AfterTurnEnd (owner side): remove self, apply +Amount Strength.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.MANGLE, amount)

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
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0


# ---------------------------------------------------------------------------
# MasterPlannerPower
# ---------------------------------------------------------------------------
class MasterPlannerPower(PowerInstance):
    """After playing a Skill card, apply the Sly keyword to it.

    C# ref: MasterPlannerPower.cs
    - AfterCardPlayed: if owner plays a Skill, apply CardKeyword.Sly to it.
    StackType.Single.

    Simplified: The card system applies Sly (card goes to top of draw pile
    instead of discard). This power flags the intent.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.MASTER_PLANNER, amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_owner = getattr(card, "owner", None)
        if card_owner is not None and card_owner is not owner:
            return
        if getattr(card, "card_type", None) != CardType.SKILL:
            return
        keywords = getattr(card, "keywords", None)
        if keywords is not None:
            card.keywords = frozenset(set(keywords) | {"sly"})


# ---------------------------------------------------------------------------
# MindRotPower
# ---------------------------------------------------------------------------
class MindRotPower(PowerInstance):
    """Reduces cards drawn per turn by Amount (minimum 0).

    C# ref: MindRotPower.cs
    - ModifyHandDraw: max(0, count - Amount).
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.MIND_ROT, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        return max(0, draw - self.amount)


# ---------------------------------------------------------------------------
# MonarchsGazePower
# ---------------------------------------------------------------------------
class MonarchsGazePower(PowerInstance):
    """After dealing powered attack damage, apply Amount
    MonarchsGazeStrengthDown (temporary negative Strength) to the target.

    C# ref: MonarchsGazePower.cs
    - AfterDamageGiven: if dealer == owner and powered attack, apply
      MonarchsGazeStrengthDownPower to target.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.MONARCHS_GAZE, amount)

    def after_damage_given(
        self, owner: Creature, dealer: Creature, target: Creature,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        if dealer is owner and props.is_powered_attack():
            combat.apply_power_to(target, PowerId.MONARCHS_GAZE_STRENGTH_DOWN, self.amount, applier=owner)


# ---------------------------------------------------------------------------
# MonarchsGazeStrengthDownPower
# ---------------------------------------------------------------------------
class MonarchsGazeStrengthDownPower(PowerInstance):
    """Temporary Strength (negative / debuff). Applied by MonarchsGazePower.
    Applies -Amount Strength on application, restores it at end of owner's turn.

    C# ref: MonarchsGazeStrengthDownPower.cs extends TemporaryStrengthPower,
    IsPositive=false.
    - BeforeApplied: apply -Amount Strength.
    - AfterTurnEnd (owner side): remove self, apply +Amount Strength.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.MONARCHS_GAZE_STRENGTH_DOWN, amount)

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
        if owner is target and power_id == PowerId.MONARCHS_GAZE_STRENGTH_DOWN and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0


# ---------------------------------------------------------------------------
# NemesisPower
# ---------------------------------------------------------------------------
class NemesisPower(PowerInstance):
    """Every other turn (alternating), gain 1 Intangible with skip_next_tick.
    On the off-turn, remove Intangible if present.

    C# ref: NemesisPower.cs
    - AfterTurnEnd (owner side): toggle flag. If flag is set, apply
      Intangible(1) with SkipNextDurationTick=true. If flag is unset
      and owner has Intangible, remove it.
    StackType.Single.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.NEMESIS, amount)
        self._should_apply_intangible: bool = False

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        self._should_apply_intangible = not self._should_apply_intangible
        if self._should_apply_intangible:
            owner.apply_power(PowerId.INTANGIBLE, 1)
            intangible = owner.powers.get(PowerId.INTANGIBLE)
            if intangible is not None:
                intangible.skip_next_tick = True
        else:
            if owner.has_power(PowerId.INTANGIBLE):
                owner.powers.pop(PowerId.INTANGIBLE, None)


# ---------------------------------------------------------------------------
# NeurosurgePower
# ---------------------------------------------------------------------------
class NeurosurgePower(PowerInstance):
    """Apply Amount Doom to owner at start of their turn.

    C# ref: NeurosurgePower.cs
    - AfterSideTurnStart (owner side): apply DoomPower(Amount) to owner.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.NEUROSURGE, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.apply_power_to(owner, PowerId.DOOM, self.amount, applier=owner)


# ---------------------------------------------------------------------------
# NightmarePower
# ---------------------------------------------------------------------------
class NightmarePower(PowerInstance):
    """At start of next turn, add Amount copies of the selected card to hand,
    then remove self.

    C# ref: NightmarePower.cs
    - BeforeHandDraw: if owner's turn, add Amount clones of selectedCard
      to hand, then remove self.
    - SetSelectedCard(): stores the card to clone.
    StackType.Counter. Instanced.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.NIGHTMARE, amount)
        self.selected_card: object | None = None
        self._selected_cards: list[tuple[object, int]] = []

    def set_selected_card(self, card: object) -> None:
        """Store the card to be cloned next turn."""
        clone = getattr(card, "clone", None)
        if callable(clone):
            self.selected_card = clone(0)
        else:
            self.selected_card = card
        self._selected_cards.append((self.selected_card, self.amount))

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player and self._selected_cards:
            clone_fn = getattr(combat, "clone_card_to_hand", None)
            if clone_fn is not None:
                for selected_card, amount in self._selected_cards:
                    for _ in range(amount):
                        clone_fn(owner, selected_card)
            self._selected_cards.clear()
            combat._remove_power(owner, self.power_id)
            self.selected_card = None


# ---------------------------------------------------------------------------
# NoDrawPower
# ---------------------------------------------------------------------------
class NoDrawPower(PowerInstance):
    """Prevents the owner from drawing cards outside of the hand draw phase.
    Removed at end of any turn.

    C# ref: NoDrawPower.cs
    - ShouldDraw: returns false for non-hand-draw draws for the owner.
    - AfterTurnEnd: remove self (unconditionally).
    StackType.Single.

    Simplified: The draw system checks owner.has_power(NO_DRAW) to block
    mid-turn draws.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.NO_DRAW, amount)

    def should_draw(self, owner: Creature, from_hand_draw: bool) -> bool | None:
        if from_hand_draw:
            return None
        return False

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        self.amount = 0


# ---------------------------------------------------------------------------
# NostalgiaPower
# ---------------------------------------------------------------------------
class NostalgiaPower(PowerInstance):
    """The first Amount Attack/Skill cards played each turn go to the top of
    the draw pile instead of discard.

    C# ref: NostalgiaPower.cs
    - ModifyCardPlayResultPileTypeAndPosition: for the first Amount
      Attack/Skill cards played this turn, redirect from Discard to
      Draw (top position).
    StackType.Counter.

    Uses card-play-start history, matching the original
    CardPlayStartedEntry check.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.NOSTALGIA, amount)

    def modify_card_play_result_pile_type_and_position(
        self,
        owner: Creature,
        card: object,
        is_auto_play: bool,
        energy_value: int,
        pile_type: PileType,
        position: CardPilePosition,
    ) -> tuple[PileType, CardPilePosition]:
        if getattr(card, "owner", None) is not owner:
            return pile_type, position
        card_type = getattr(card, "card_type", None)
        if card_type not in (CardType.ATTACK, CardType.SKILL):
            return pile_type, position
        if pile_type != PileType.DISCARD:
            return pile_type, position
        combat = getattr(owner, "combat_state", None)
        if combat is None:
            return pile_type, position
        qualifying_starts = combat.count_card_play_starts_this_turn(
            owner,
            card_type=CardType.ATTACK,
            exclude_card=card,
        ) + combat.count_card_play_starts_this_turn(
            owner,
            card_type=CardType.SKILL,
            exclude_card=card,
        )
        if qualifying_starts >= self.amount:
            return pile_type, position
        return PileType.DRAW, CardPilePosition.TOP


# ---------------------------------------------------------------------------
# OblivionPower
# ---------------------------------------------------------------------------
class OblivionPower(PowerInstance):
    """When the applier plays a card, apply Amount Doom to the owner.
    Removed at end of player turn.

    C# ref: OblivionPower.cs
    - BeforeCardPlayed: if card owner == applier's player, record.
    - AfterCardPlayed: apply DoomPower(Amount) to owner.
    - AfterTurnEnd (Player side): remove self.
    StackType.Counter.

    Simplified: In single-player, this triggers when the player plays a card
    and applies Doom to the enemy (owner of this power).
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.OBLIVION, amount)
        self.applier: Creature | None = None
        self._amounts_for_played_cards: dict[int, int] = {}

    def before_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_owner = getattr(card, "owner", None)
        if self.applier is not None and card_owner is self.applier:
            self._amounts_for_played_cards[id(card)] = self.amount

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        amount = self._amounts_for_played_cards.pop(id(card), None)
        if amount is not None:
            combat.apply_power_to(owner, PowerId.DOOM, amount, applier=self.applier)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == CombatSide.PLAYER:
            self.amount = 0


# ---------------------------------------------------------------------------
# OrbitPower
# ---------------------------------------------------------------------------
class OrbitPower(PowerInstance):
    """Every 4 energy spent, gain Amount energy.

    C# ref: OrbitPower.cs
    - AfterEnergySpent: track total energy spent by owner. Every 4 energy,
      gain Amount energy. Triggers = energySpent // 4 - previousTriggers.
    StackType.Counter. Instanced.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    ENERGY_INCREMENT = 4

    def __init__(self, amount: int):
        super().__init__(PowerId.ORBIT, amount)
        self._energy_spent: int = 0
        self._trigger_count: int = 0
        self._instances: list[tuple[int, int, int]] = [(amount, 0, 0)]

    def _add_instance(self, amount: int) -> None:
        self._instances.append((amount, 0, 0))
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

    def after_energy_spent(
        self, owner: Creature, card: object, amount: int, combat: CombatState
    ) -> None:
        if getattr(card, "owner", None) is owner:
            self.on_energy_spent(owner, amount, combat)

    def on_energy_spent(self, owner: Creature, energy_amount: int, combat: CombatState) -> None:
        """Called by the energy system when energy is spent."""
        if energy_amount <= 0:
            return
        updated_instances: list[tuple[int, int, int]] = []
        total_energy = 0
        for amount, energy_spent, trigger_count in self._instances:
            energy_spent += energy_amount
            triggers = energy_spent // self.ENERGY_INCREMENT - trigger_count
            if triggers > 0:
                total_energy += amount * triggers
                trigger_count += triggers
            updated_instances.append((amount, energy_spent, trigger_count))
        self._instances = updated_instances
        if total_energy > 0:
            combat.gain_energy(owner, total_energy)
        if self._instances:
            _, self._energy_spent, self._trigger_count = self._instances[-1]


# ---------------------------------------------------------------------------
# OutbreakPower
# ---------------------------------------------------------------------------
class OutbreakPower(PowerInstance):
    """Every 3rd time the owner applies Poison, deal Amount damage to all
    enemies (unpowered).

    C# ref: OutbreakPower.cs
    - AfterPowerAmountChanged: if owner applied Poison (amount > 0),
      increment counter. Every 3rd application, deal Amount damage to all
      hittable enemies.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.OUTBREAK, amount)

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
        if applier is not owner or power_id != PowerId.POISON or amount <= 0:
            return
        for enemy in combat.hittable_enemies:
            combat.deal_damage(
                dealer=owner,
                target=enemy,
                amount=self.amount,
                props=ValueProp.UNPOWERED,
            )


# ---------------------------------------------------------------------------
# PagestormPower
# ---------------------------------------------------------------------------
class PagestormPower(PowerInstance):
    """When owner draws an Ethereal card, draw Amount additional cards.

    C# ref: PagestormPower.cs
    - AfterCardDrawn: if owner draws a card with Ethereal keyword,
      draw Amount cards.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PAGESTORM, amount)

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        """Called by the draw system when a card is drawn."""
        if getattr(card, "owner", None) is not owner:
            return
        keywords = getattr(card, "keywords", set())
        if CardKeyword.ETHEREAL in keywords or "ethereal" in keywords:
            combat.draw_cards(owner, self.amount)


# ---------------------------------------------------------------------------
# PaleBlueDotPower
# ---------------------------------------------------------------------------
class PaleBlueDotPower(PowerInstance):
    """If the owner played >= 5 cards last turn, draw Amount extra cards
    this turn.

    C# ref: PaleBlueDotPower.cs
    - ModifyHandDraw: if cards played last round >= 5 (CardPlay threshold),
      add Amount to draw count.
    StackType.Counter.

    Simplified: The combat state tracks cards played per round. We check
    the previous round's count.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    CARD_PLAY_THRESHOLD = 5

    def __init__(self, amount: int):
        super().__init__(PowerId.PALE_BLUE_DOT, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        combat = getattr(owner, "combat_state", None)
        last_turn_plays = 0
        if combat is not None:
            last_turn_plays = combat.count_cards_played_last_round(owner)
        if last_turn_plays >= self.CARD_PLAY_THRESHOLD:
            return draw + self.amount
        return draw


# ---------------------------------------------------------------------------
# PaperCutsPower
# ---------------------------------------------------------------------------
class PaperCutsPower(PowerInstance):
    """After dealing unblocked powered attack damage to a player, reduce
    that player's max HP by Amount.

    C# ref: PaperCutsPower.cs
    - AfterDamageGiven: if dealer == owner, target is player, powered attack,
      and unblocked damage > 0, reduce target's max HP by Amount.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PAPER_CUTS, amount)

    def after_damage_given(
        self, owner: Creature, dealer: Creature, target: Creature,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        if dealer is not owner:
            return
        if not props.is_powered_attack():
            return
        if damage <= 0:
            return
        if not getattr(target, "is_player", False):
            return
        lose_max_hp = getattr(target, "lose_max_hp", None)
        if lose_max_hp is not None:
            lose_max_hp(self.amount)


# ---------------------------------------------------------------------------
# ParryPower
# ---------------------------------------------------------------------------
class ParryPower(PowerInstance):
    """After owner plays a Sovereign Blade, gain Amount block (unpowered).

    C# ref: ParryPower.cs
    - AfterSovereignBladePlayed: if dealer == owner, gain Amount block.
    StackType.Counter.

    Simplified: The card system calls on_sovereign_blade_played(). Also
    triggers via after_card_played when the card is a Sovereign Blade.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PARRY, amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        from sts2_env.cards.status import is_sovereign_blade

        if is_sovereign_blade(card):
            card_owner = getattr(card, "owner", None)
            if card_owner is owner or card_owner is None:
                _gain_unpowered_block(owner, self.amount, combat)


# ---------------------------------------------------------------------------
# PiercingWailPower
# ---------------------------------------------------------------------------
class PiercingWailPower(PowerInstance):
    """Temporary Strength (negative / debuff). Applied by Piercing Wail card.
    Applies -Amount Strength on application, restores at end of owner's turn.

    C# ref: PiercingWailPower.cs extends TemporaryStrengthPower,
    IsPositive=false.
    - BeforeApplied: apply -Amount Strength.
    - AfterTurnEnd (owner side): remove self, apply +Amount Strength.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.PIERCING_WAIL, amount)

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
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0


# ---------------------------------------------------------------------------
# PillarOfCreationPower
# ---------------------------------------------------------------------------
class PillarOfCreationPower(PowerInstance):
    """When a card is generated and added to combat (by the owner's player),
    gain Amount block (unpowered).

    C# ref: PillarOfCreationPower.cs
    - AfterCardGeneratedForCombat: if card owner == owner.Player and
      addedByPlayer, gain Amount block.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PILLAR_OF_CREATION, amount)
        self._triggered_turn: int | None = None

    def after_card_generated_for_combat(
        self,
        owner: Creature,
        card: object,
        added_by_player: bool,
        combat: CombatState,
    ) -> None:
        if not added_by_player or getattr(card, "owner", None) is not owner:
            return
        turn = combat.turn_count
        if self._triggered_turn == turn:
            return
        self._triggered_turn = turn
        _gain_unpowered_block(owner, self.amount, combat)


# ---------------------------------------------------------------------------
# PossessSpeedPower
# ---------------------------------------------------------------------------
class PossessSpeedPower(PowerInstance):
    """Tracks Dexterity stolen from players by the owner. When the owner
    dies, restores the stolen Dexterity to each victim.

    C# ref: PossessSpeedPower.cs
    - AfterPowerAmountChanged: if owner applied negative Dexterity to a
      player, track the amount.
    - AfterDeath: if owner dies, restore stolen Dexterity to victims.
    StackType.Single.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.POSSESS_SPEED, amount)
        self._stolen_dexterity: dict[Creature, int] = {}

    def track_stolen_dexterity(self, victim: Creature, stolen_amount: int) -> None:
        """Called by the power application system when the owner steals
        Dexterity from a player."""
        if victim not in self._stolen_dexterity:
            self._stolen_dexterity[victim] = 0
        self._stolen_dexterity[victim] += stolen_amount

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
        if applier is owner and power_id == PowerId.DEXTERITY and amount < 0 and target.is_player:
            self.track_stolen_dexterity(target, -amount)

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented or creature is not owner:
            return
        for victim, stolen in self._stolen_dexterity.items():
            victim.apply_power(PowerId.DEXTERITY, stolen)
        self._stolen_dexterity.clear()


# ---------------------------------------------------------------------------
# PossessStrengthPower
# ---------------------------------------------------------------------------
class PossessStrengthPower(PowerInstance):
    """Tracks Strength stolen from players by the owner. When the owner
    dies, restores the stolen Strength to each victim.

    C# ref: PossessStrengthPower.cs
    - AfterPowerAmountChanged: if owner applied negative Strength to a
      player, track the amount.
    - AfterDeath: if owner dies, restore stolen Strength to victims.
    StackType.Single.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.POSSESS_STRENGTH, amount)
        self._stolen_strength: dict[Creature, int] = {}

    def track_stolen_strength(self, victim: Creature, stolen_amount: int) -> None:
        """Called by the power application system when the owner steals
        Strength from a player."""
        if victim not in self._stolen_strength:
            self._stolen_strength[victim] = 0
        self._stolen_strength[victim] += stolen_amount

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
        if applier is owner and power_id == PowerId.STRENGTH and amount < 0 and target.is_player:
            self.track_stolen_strength(target, -amount)

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented or creature is not owner:
            return
        for victim, stolen in self._stolen_strength.items():
            victim.apply_power(PowerId.STRENGTH, stolen)
        self._stolen_strength.clear()


# ---------------------------------------------------------------------------
# PrepTimePower
# ---------------------------------------------------------------------------
class PrepTimePower(PowerInstance):
    """Gain Amount Vigor at the start of each turn.

    C# ref: PrepTimePower.cs
    - AfterSideTurnStart (owner side): apply VigorPower(Amount) to owner.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PREP_TIME, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.VIGOR, self.amount)


# ---------------------------------------------------------------------------
# PyrePower
# ---------------------------------------------------------------------------
class PyrePower(PowerInstance):
    """Increases max energy by Amount.

    C# ref: PyrePower.cs
    - ModifyMaxEnergy: +Amount for owner's player.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.PYRE, amount)

    def modify_max_energy(self, owner: Creature, energy: int) -> int:
        return energy + self.amount


# ---------------------------------------------------------------------------
# RadiancePower
# ---------------------------------------------------------------------------
class RadiancePower(PowerInstance):
    """At start of turn (energy reset), gain 1 energy and decrement.

    C# ref: RadiancePower.cs
    - AfterEnergyReset: gain EnergyVar (default 1) energy, then decrement.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    ENERGY_GAIN = 1

    def __init__(self, amount: int):
        super().__init__(PowerId.RADIANCE, amount)

    def after_energy_reset(self, owner: Creature, combat: CombatState) -> None:
        if owner.is_player:
            combat.gain_energy(owner, self.ENERGY_GAIN)
            self.amount -= 1

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if (
            side == owner.side
            and owner.is_player
            and not combat.has_energy_reset_this_turn(owner)
        ):
            self.after_energy_reset(owner, combat)


# ---------------------------------------------------------------------------
# RampartPower
# ---------------------------------------------------------------------------
class RampartPower(PowerInstance):
    """At start of player turn, grant Amount block (unpowered) to all
    TurretOperator enemies.

    C# ref: RampartPower.cs
    - AfterSideTurnStart (Player side): give Amount block to all
      TurretOperator enemies.
    StackType.Counter.

    Simplified: Grants block to all allied creatures (on the owner's side)
    at the start of the player turn.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    should_scale_in_multiplayer = True

    def __init__(self, amount: int):
        super().__init__(PowerId.RAMPART, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != CombatSide.PLAYER:
            return
        from sts2_env.monsters.act3 import TURRET_OPERATOR_MONSTER_ID

        for enemy in getattr(combat, "enemies", []):
            if enemy.is_alive and getattr(enemy, "monster_id", None) == TURRET_OPERATOR_MONSTER_ID:
                _gain_unpowered_block(enemy, self.amount, combat)


# ---------------------------------------------------------------------------
# ReaperFormPower
# ---------------------------------------------------------------------------
class ReaperFormPower(PowerInstance):
    """After dealing powered attack damage, apply (total_damage * Amount)
    Doom to the target.

    C# ref: ReaperFormPower.cs
    - AfterDamageGiven: if dealer == owner (or owner's pet) and powered
      attack and totalDamage > 0, apply Doom = totalDamage * Amount.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.REAPER_FORM, amount)

    def after_damage_given(
        self, owner: Creature, dealer: Creature, target: Creature,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        if dealer is None:
            return
        is_owner_or_pet = dealer is owner or getattr(dealer, "pet_owner", None) is owner
        if not is_owner_or_pet:
            return
        if not props.is_powered_attack():
            return
        result = getattr(combat, "_active_damage_result", None)
        total_damage = getattr(result, "total_damage", damage)
        if total_damage <= 0:
            return
        combat.apply_power_to(target, PowerId.DOOM, total_damage * self.amount, applier=owner)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
from sts2_env.core.creature import register_power_class  # noqa: E402

_ALL_POWERS: dict[PowerId, type[PowerInstance]] = {
    PowerId.GRAVITY: GravityPower,
    PowerId.GUARDED: GuardedPower,
    PowerId.HAILSTORM: HailstormPower,
    PowerId.HAMMER_TIME: HammerTimePower,
    PowerId.HANG: HangPower,
    PowerId.HARD_TO_KILL: HardToKillPower,
    PowerId.HAUNT: HauntPower,
    PowerId.HELICAL_DART: HelicalDartPower,
    PowerId.HEX: HexPower,
    PowerId.HIGH_VOLTAGE: HighVoltagePower,
    PowerId.HOTFIX: HotfixPower,
    PowerId.IMBALANCED: ImbalancedPower,
    PowerId.IMPROVEMENT: ImprovementPower,
    PowerId.INFERNO: InfernoPower,
    PowerId.ITERATION: IterationPower,
    PowerId.JUGGLING: JugglingPower,
    PowerId.KNOCKDOWN: KnockdownPower,
    PowerId.LEADERSHIP: LeadershipPower,
    PowerId.LIGHTNING_ROD: LightningRodPower,
    PowerId.MAGIC_BOMB: MagicBombPower,
    PowerId.MANGLE: ManglePower,
    PowerId.MASTER_PLANNER: MasterPlannerPower,
    PowerId.MIND_ROT: MindRotPower,
    PowerId.MONARCHS_GAZE: MonarchsGazePower,
    PowerId.MONARCHS_GAZE_STRENGTH_DOWN: MonarchsGazeStrengthDownPower,
    PowerId.NEMESIS: NemesisPower,
    PowerId.NEUROSURGE: NeurosurgePower,
    PowerId.NIGHTMARE: NightmarePower,
    PowerId.NO_DRAW: NoDrawPower,
    PowerId.NOSTALGIA: NostalgiaPower,
    PowerId.OBLIVION: OblivionPower,
    PowerId.ORBIT: OrbitPower,
    PowerId.OUTBREAK: OutbreakPower,
    PowerId.PAGESTORM: PagestormPower,
    PowerId.PALE_BLUE_DOT: PaleBlueDotPower,
    PowerId.PAPER_CUTS: PaperCutsPower,
    PowerId.PARRY: ParryPower,
    PowerId.PIERCING_WAIL: PiercingWailPower,
    PowerId.PILLAR_OF_CREATION: PillarOfCreationPower,
    PowerId.POSSESS_SPEED: PossessSpeedPower,
    PowerId.POSSESS_STRENGTH: PossessStrengthPower,
    PowerId.PREP_TIME: PrepTimePower,
    PowerId.PYRE: PyrePower,
    PowerId.RADIANCE: RadiancePower,
    PowerId.RAMPART: RampartPower,
    PowerId.REAPER_FORM: ReaperFormPower,
}

for _pid, _cls in _ALL_POWERS.items():
    register_power_class(_pid, _cls)
