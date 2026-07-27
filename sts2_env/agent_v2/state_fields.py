"""Stable player-visible state projections for powers and relics."""

from __future__ import annotations

from typing import Any

from sts2_env.core.enums import PowerId
from sts2_env.relics.base import RelicId


_POWER_COUNTER_FIELDS = {
    "_cards_left": "cards_left",
    "_cards_played_this_turn": "cards_played_this_turn",
    "_slow_amount": "slow_amount",
    "_cards_left_this_turn": "cards_left_this_turn",
    "_block_value": "block_value",
}

_RELIC_COUNTER_FIELDS = {
    "_times_lifted": "times_lifted",
    "_attacks_played": "attacks_played",
    "_attacks_this_turn": "attacks_this_turn",
    "_attacks_played_this_turn": "attacks_played_this_turn",
    "_skills_played": "skills_played",
    "_skills_played_this_turn": "skills_played_this_turn",
    "_powers_played_this_turn": "powers_played_this_turn",
    "_cards_played_this_turn": "cards_played_this_turn",
    "_cards_played_last_turn": "cards_played_last_turn",
    "_times_used": "times_used",
    "times_used": "times_used",
    "_combats_seen": "combats_seen",
    "_combats_finished": "combats_finished",
    "_elites_defeated": "elites_defeated",
    "_turns_seen": "turns_seen",
    "_used_this_combat": "used_this_combat",
    "_used_this_turn": "used_this_turn",
    "_was_used": "was_used",
    "used": "used",
}


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _visible_counters(instance: object, fields: dict[str, str]) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    # Keep every scalar simulator state field: these are deterministic,
    # JSON-safe, and include implementation-specific counters that do not
    # have a dedicated public property yet.
    for attribute, value in vars(instance).items():
        if attribute in {
            "amount", "power_id", "relic_id", "applier", "enabled",
            "skip_next_tick", "ignore_next_instance",
        }:
            continue
        if isinstance(value, (bool, int, float, str)):
            counters[attribute.lstrip("_")] = value
    for attribute, key in fields.items():
        value = getattr(instance, attribute, None)
        if isinstance(value, (bool, int, float, str)):
            counters[key] = value
    return counters


def serialize_power_fields(power: object) -> dict[str, Any]:
    """Return fields corresponding to the game's public PowerModel surface."""
    amount = int(getattr(power, "amount", 0))
    power_id = getattr(power, "power_id", PowerId.STRENGTH)
    counters = _visible_counters(power, _POWER_COUNTER_FIELDS)
    display_amount = amount
    if power_id == PowerId.SLOW and "slow_amount" in counters:
        display_amount = int(counters["slow_amount"]) * 10
    elif "cards_left" in counters:
        display_amount = int(counters["cards_left"])
    elif "cards_played_this_turn" in counters and power_id in {
        PowerId.SLOTH,
        PowerId.TENDER,
    }:
        display_amount = int(counters["cards_played_this_turn"])
    counters.setdefault("display_amount", display_amount)
    return {
        "power_type": _enum_name(getattr(power, "power_type", "BUFF")),
        "stack_type": _enum_name(getattr(power, "stack_type", "COUNTER")),
        "amount": amount,
        "display_amount": display_amount,
        "amount_on_turn_start": int(
            getattr(power, "amount_on_turn_start", amount)
        ),
        "skip_next_duration_tick": bool(
            getattr(power, "skip_next_tick", False)
        ),
        "is_visible": True,
        "dynamic_vars": {},
        "counters": counters,
    }


def serialize_relic_fields(relic: object) -> dict[str, Any]:
    """Return stable relic status plus audited, player-visible counters."""
    relic_id = getattr(relic, "relic_id", None)
    counters = _visible_counters(relic, _RELIC_COUNTER_FIELDS)
    display_amount = 0
    show_counter = False

    if relic_id == RelicId.GIRYA:
        display_amount = int(counters.get("times_lifted", 0))
        show_counter = True
    elif relic_id in {RelicId.NUNCHAKU, RelicId.PEN_NIB}:
        display_amount = int(counters.get("attacks_played", 0)) % 10
        show_counter = True
    elif relic_id == RelicId.ORNAMENTAL_FAN:
        display_amount = int(counters.get("attacks_this_turn", 0)) % 3
        show_counter = True
    elif relic_id == RelicId.WINGED_BOOTS:
        times_used = int(counters.get("times_used", 0))
        display_amount = max(0, 3 - times_used)
        counters["uses_remaining"] = display_amount
        show_counter = display_amount > 0
    elif counters:
        # A counter-bearing implementation remains observable even when the
        # simulator does not yet define a specialized display transform.
        display_amount = int(next(
            (value for value in counters.values() if isinstance(value, int)),
            0,
        ))
        show_counter = True

    counters["display_amount"] = display_amount
    is_used_up = bool(getattr(relic, "is_used_up", False))
    status = "Normal"
    if not getattr(relic, "enabled", True) or is_used_up:
        status = "Disabled"
    elif relic_id in {
        RelicId.NUNCHAKU,
        RelicId.PEN_NIB,
    } and display_amount == 9:
        status = "Active"
    elif relic_id == RelicId.ORNAMENTAL_FAN and display_amount == 2:
        status = "Active"
    return {
        "rarity": _enum_name(getattr(relic, "rarity", "COMMON")),
        "stack_count": int(getattr(relic, "stack_count", 1)),
        "status": status,
        "is_used_up": is_used_up,
        "is_melted": bool(getattr(relic, "is_melted", False)),
        "is_wax": bool(getattr(relic, "is_wax", False)),
        "show_counter": show_counter,
        "display_amount": display_amount,
        "floor_added": int(getattr(relic, "floor_added", 0)),
        "dynamic_vars": {},
        "counters": counters,
    }
