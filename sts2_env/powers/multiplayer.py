"""Powers introduced with the v0.108.0 multiplayer card set."""

from __future__ import annotations

from sts2_env.cards.base import CardInstance, _get_next_id
from sts2_env.core.enums import CardId, CardType, CombatSide, PowerId, PowerStackType, PowerType, ValueProp
from sts2_env.powers.base import PowerInstance
from sts2_env.core.creature import register_power_class


class AmbergrisPower(PowerInstance):
    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.AMBERGRIS, amount)

    def should_take_extra_turn(self, owner, combat) -> bool:
        return self.amount > 0

    def after_taking_extra_turn(self, owner, combat) -> None:
        self.amount -= 1


class CacophonyPower(PowerInstance):
    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CACOPHONY, amount)
        self.cards_remaining = 33

    def on_card_drawn(self, owner, card, from_hand_draw: bool, combat) -> None:
        self.cards_remaining -= 1
        if self.cards_remaining > 0:
            return
        target = combat.random_enemy_of(owner)
        if target is not None:
            combat.deal_damage(owner, target, self.amount, ValueProp.UNPOWERED)
        self.cards_remaining = 33


class BorrowedTimePower(PowerInstance):
    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BORROWED_TIME_POWER, amount)

    def modify_card_cost(self, owner, card) -> int | None:
        return max(0, getattr(card, "cost", 0) + self.amount)

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side == owner.side:
            self.amount = 0


class NoEnergyGainPower(PowerInstance):
    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int):
        super().__init__(PowerId.NO_ENERGY_GAIN, amount)

    def modify_energy_gain(self, owner, target, amount: int) -> int:
        return 0 if owner is target else amount

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side == owner.side:
            self.amount = 0


class TaintedPower(PowerInstance):
    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.TAINTED, amount)

    def modify_damage_additive(self, owner, dealer, target, props: ValueProp) -> int:
        if target is owner and props.is_powered_attack():
            return self.amount
        return 0

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side == CombatSide.ENEMY:
            self.amount = 0

class ConcoctPower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.CONCOCT, amount)

    def after_damage_given(self, owner, dealer, target, damage: int, props: ValueProp, combat) -> None:
        if dealer is owner and damage > 0 and props.is_powered_attack():
            combat.apply_power_to(target, PowerId.POISON, self.amount, applier=owner)

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side != owner.side:
            self.amount = 0


class FadePower(PowerInstance):
    power_type = PowerType.BUFF
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.FADE, amount)

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.DEXTERITY, -self.amount, applier=owner)
            self.amount = 0


class HibernatePower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.HIBERNATE, amount)

    def after_player_turn_start(self, owner, combat) -> None:
        self.amount -= 1


class ImitationLearningPower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.IMITATION_LEARNING, amount)
        self.player_target = None
        self._clone: CardInstance | None = None

    def before_card_played(self, owner, card, combat) -> None:
        if (
            self.player_target is not None
            and getattr(card, "owner", None) is self.player_target
            and getattr(card, "card_type", None) is CardType.POWER
        ):
            self._clone = card.clone(_get_next_id())
            self._clone.owner = owner

    def after_card_played(self, owner, card, combat) -> None:
        if self._clone is not None:
            clone, self._clone = self._clone, None
            self.amount -= 1
            combat.auto_play_card(clone)


class OneForAllPower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.ONE_FOR_ALL, amount)

    def modify_damage_additive(self, owner, dealer, target, props: ValueProp) -> int:
        if dealer is owner and props.is_powered_attack():
            return self.amount
        return 0


class SoulboundPower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.SOULBOUND, amount)
        self.soul_creator = None
        self._adding = False

    def after_card_generated_for_combat(self, owner, card, added_by_player: bool, combat) -> None:
        if self._adding or card.card_id is not CardId.SOUL or card.owner is not self.soul_creator:
            return
        from sts2_env.cards.status import make_soul

        self._adding = True
        for _ in range(self.amount):
            combat.add_generated_card_to_creature_draw_pile(owner, make_soul(), random_position=True)
        self._adding = False


class UnderworldPower(PowerInstance):
    power_type = PowerType.BUFF

    def __init__(self, amount: int):
        super().__init__(PowerId.UNDERWORLD, amount)

    def after_damage_given(self, owner, dealer, target, damage: int, props: ValueProp, combat) -> None:
        if dealer is not owner and dealer.side == owner.side and damage > 0 and props.is_powered_attack():
            combat.apply_power_to(target, PowerId.DOOM, damage * self.amount, applier=owner)

    def after_turn_end(self, owner, side: CombatSide, combat) -> None:
        if side == CombatSide.ENEMY:
            self.amount = 0


class WitheringPresencePower(PowerInstance):
    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.WITHERING_PRESENCE, amount)

    def after_card_played(self, owner, card, combat) -> None:
        if getattr(card, "owner", None) is not owner:
            return
        self.amount -= 1
        if self.amount <= 0:
            from sts2_env.cards.status import make_wither

            combat.add_generated_card_to_creature_hand(owner, make_wither())
            self.amount = 6


for _power_id, _power_cls in {
    PowerId.AMBERGRIS: AmbergrisPower,
    PowerId.BORROWED_TIME_POWER: BorrowedTimePower,
    PowerId.CACOPHONY: CacophonyPower,
    PowerId.CONCOCT: ConcoctPower,
    PowerId.FADE: FadePower,
    PowerId.HIBERNATE: HibernatePower,
    PowerId.IMITATION_LEARNING: ImitationLearningPower,
    PowerId.NO_ENERGY_GAIN: NoEnergyGainPower,
    PowerId.ONE_FOR_ALL: OneForAllPower,
    PowerId.SOULBOUND: SoulboundPower,
    PowerId.TAINTED: TaintedPower,
    PowerId.UNDERWORLD: UnderworldPower,
    PowerId.WITHERING_PRESENCE: WitheringPresencePower,
}.items():
    register_power_class(_power_id, _power_cls)
