#!/usr/bin/env python3
"""Interactive command-line interface for a full STS2 run."""

from __future__ import annotations

import argparse
import random
import re
import sys
from typing import Any

import sts2_env.events  # noqa: F401
import sts2_env.potions.effects  # noqa: F401

from sts2_env.core.combat import CombatState
from sts2_env.core.enums import IntentType, PotionTargetType, TargetType
from sts2_env.core.rng import INT_MAX
from sts2_env.run.reward_objects import CardBundlesReward, CardReward, PotionReward, RelicReward
from sts2_env.run.run_manager import RunManager, SUPPORTED_CHARACTER_IDS
from sts2_env.run.shop import is_shop_entry_available


HEADER_WIDTH = 72
DEFAULT_CHARACTER_INDEX = 0
CHARACTERS = SUPPORTED_CHARACTER_IDS
QUIT_COMMANDS = {"q", "quit", "exit"}
HELP_COMMANDS = {"h", "help", "?"}
CONFIRM_COMMANDS = {"c", "confirm", "e", "end", "skip"}
AUTO_COMMANDS = {"a", "auto"}
INSPECT_COMMANDS = {"i", "inspect", "deck"}
PHASE_LABELS = {
    RunManager.PHASE_MAP_CHOICE: "Map",
    RunManager.PHASE_COMBAT: "Combat",
    RunManager.PHASE_CARD_REWARD: "Reward",
    RunManager.PHASE_BOSS_RELIC: "Boss Relic",
    RunManager.PHASE_SHOP: "Shop",
    RunManager.PHASE_REST_SITE: "Rest Site",
    RunManager.PHASE_EVENT: "Event",
    RunManager.PHASE_TREASURE: "Treasure",
    RunManager.PHASE_RUN_OVER: "Run Over",
}
ROOM_LABELS = {
    "MONSTER": "Monster",
    "ELITE": "Elite",
    "BOSS": "Boss",
    "SHOP": "Shop",
    "TREASURE": "Treasure",
    "REST_SITE": "Rest",
    "UNKNOWN": "?",
    "ANCIENT": "Event",
    "UNASSIGNED": "?",
}
INTENT_LABELS = {
    IntentType.ATTACK: "Attack",
    IntentType.MULTI_ATTACK: "Attack",
    IntentType.DEFEND: "Block",
    IntentType.BUFF: "Buff",
    IntentType.DEBUFF: "Debuff",
    IntentType.DEBUFF_STRONG: "Strong debuff",
    IntentType.SLEEP: "Sleep",
    IntentType.SUMMON: "Summon",
    IntentType.ESCAPE: "Escape",
    IntentType.UNKNOWN: "Unknown",
    IntentType.STATUS_CARD: "Status",
    IntentType.STUN: "Stun",
    IntentType.HEAL: "Heal",
    IntentType.DEATH_BLOW: "Death blow",
    IntentType.CARD_DEBUFF: "Card debuff",
}


def display_name(value: object) -> str:
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text.title()


def display_text(text: str) -> str:
    text = re.sub(
        r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
        lambda match: display_name(match.group(0)),
        text,
    )
    return re.sub(
        r"\b[A-Z][A-Z0-9]{2,}\b",
        lambda match: display_name(match.group(0)),
        text,
    )


def describe_card(card: object) -> str:
    return display_text(repr(card))


def describe_action(action: dict[str, Any]) -> str:
    kind = action.get("action")
    if kind == "move":
        coord = action.get("coord")
        return f"Go to {coord}  [{ROOM_LABELS.get(str(action.get('point_type')), display_name(action.get('point_type')))}]"
    if kind == "event_choice":
        label = action.get("label", action.get("option_id", "choice"))
        description = action.get("description", "")
        return f"{label}" + (f" - {description}" if description else "")
    if kind == "choose":
        name = display_name(action.get("card_id", action.get("index")))
        source = action.get("source_pile")
        selected = " [selected]" if action.get("selected") else ""
        return f"Choose {name}" + (f" from {source}" if source else "") + selected
    if kind == "confirm_choice":
        return "Confirm / skip"
    if kind == "play_card":
        target = action.get("target_name")
        text = f"Play {display_name(action.get('card_id'))} from hand[{action.get('hand_index')}]"
        if target is not None:
            text += f" -> {display_name(target)}"
        return text
    if kind == "use_potion":
        target = action.get("target_name")
        text = f"Use {display_name(action.get('potion_id'))} from slot {action.get('slot_index')}"
        if target is not None:
            text += f" -> {display_name(target)}"
        return text
    if kind == "end_turn":
        return "End turn"
    if kind == "pick_card":
        name = display_name(action.get("card_id"))
        rarity = action.get("rarity")
        upgraded = "+" if action.get("upgraded") else ""
        return f"Take card {name}{upgraded}" + (f" [{rarity}]" if rarity else "")
    if kind == "pick_card_bundle":
        names = ", ".join(display_name(card_id) for card_id in action.get("card_ids", []))
        return f"Take card bundle: {names}"
    if kind == "reroll_card_reward":
        return f"Reroll card reward ({action.get('rerolls_remaining')} left)"
    if kind == "sacrifice_card_reward":
        return "Sacrifice card reward"
    if kind == "skip":
        return "Skip"
    if kind == "pick_potion":
        return f"Take potion {display_name(action.get('potion_id'))}"
    if kind == "skip_potion":
        return f"Skip potion {display_name(action.get('potion_id'))}"
    if kind == "pick_relic_reward":
        return f"Take relic {display_name(action.get('relic_id'))}"
    if kind == "skip_relic":
        return f"Skip relic {display_name(action.get('relic_id'))}"
    if kind == "pick_relic":
        return f"Take boss relic {display_name(action.get('relic_id'))}"
    if kind == "buy_card":
        return f"Buy card {display_name(action.get('card_id'))} for {action.get('price')}g"
    if kind == "buy_relic":
        return f"Buy relic {display_name(action.get('relic_id', action.get('rarity')))} for {action.get('price')}g"
    if kind == "buy_potion":
        return f"Buy potion {display_name(action.get('potion_id', action.get('rarity')))} for {action.get('price')}g"
    if kind == "remove_card":
        return f"Remove a card for {action.get('price')}g"
    if kind == "leave_shop":
        return "Leave shop"
    if kind == "rest_option":
        label = action.get("label", action.get("option_id"))
        description = action.get("description", "")
        return f"{label}" + (f" - {description}" if description else "")
    if kind == "collect":
        relic = action.get("relic_id")
        return "Collect treasure" + (f" ({display_name(relic)})" if relic else "")
    if kind == "select_player":
        selected = " [selected]" if action.get("selected") else ""
        return f"Act as {action.get('character_id')} player {action.get('player_id')}{selected}"
    return str(action)


def describe_intent(intent: object) -> str:
    intent_type = getattr(intent, "intent_type", IntentType.UNKNOWN)
    label = INTENT_LABELS.get(intent_type, display_name(intent_type))
    if intent_type == IntentType.ATTACK:
        return f"{label} {getattr(intent, 'damage', 0)}"
    if intent_type == IntentType.MULTI_ATTACK:
        return f"{label} {getattr(intent, 'damage', 0)}x{getattr(intent, 'hits', 1)}"
    return label


def describe_enemy_intents(ai: object | None) -> str:
    if ai is None:
        return "Unknown"
    move = getattr(ai, "current_move", None)
    intents = getattr(move, "intents", [])
    if not intents:
        return "Unknown"
    return ", ".join(describe_intent(intent) for intent in intents)


def display_header(mgr: RunManager) -> None:
    summary = mgr.summary()
    print()
    print("=" * HEADER_WIDTH)
    print(
        f"  Act {summary['act'] + 1}  Floor {summary['floor']}  |  "
        f"{PHASE_LABELS.get(summary['phase'], summary['phase'])}  |  HP {summary['hp']}  Gold {summary['gold']}  "
        f"Deck {summary['deck_size']}"
    )
    relics = summary["relics"]
    if relics:
        print(f"  Relics: {', '.join(display_name(relic) for relic in relics)}")
    print("=" * HEADER_WIDTH)


def display_map(mgr: RunManager, actions: list[dict[str, Any]]) -> None:
    run_state = mgr.run_state
    act_map = run_state.map
    visited = set(run_state.visited_map_coords)
    reachable = {tuple(action["coord"]) for action in actions if action.get("action") == "move"}
    if act_map is None:
        return
    print("\n  MAP:")
    if run_state.visited_map_coords:
        current = run_state.visited_map_coords[-1]
        print(f"    Current: ({current.col},{current.row})   * reachable   x visited")
    else:
        print("    Current: start   * reachable   x visited")
    max_row = max((point.row for point in act_map.all_points()), default=0)
    for row in range(max_row, 0, -1):
        points = act_map.get_row(row)
        if not points:
            continue
        parts: list[str] = []
        for point in points:
            coord = (point.col, point.row)
            marker = "*" if coord in reachable else "x" if point.coord in visited else " "
            parts.append(f"{marker}({point.col},{point.row}) {ROOM_LABELS.get(point.point_type.name, display_name(point.point_type.name))}")
        print("    " + "   ".join(parts))
    if reachable:
        print("\n  NEXT PATHS:")
        for point in sorted(
            (act_map.get_point(action_coord) for action_coord in run_state.get_available_next_coords()),
            key=lambda item: (item.col, item.row) if item is not None else (-1, -1),
        ):
            if point is None:
                continue
            children = ", ".join(
                f"({child.col},{child.row}) {ROOM_LABELS.get(child.point_type.name, display_name(child.point_type.name))}"
                for child in sorted(point.children, key=lambda child: child.col)
            )
            suffix = f" -> {children}" if children else ""
            print(
                f"    ({point.col},{point.row}) "
                f"{ROOM_LABELS.get(point.point_type.name, display_name(point.point_type.name))}{suffix}"
            )


def display_combat(combat: CombatState) -> None:
    player = combat.primary_player
    print()
    print(f"  Round {combat.round_number}  Energy {combat.energy}  HP {player.current_hp}/{player.max_hp}  Block {player.block}")
    print("\n  ENEMIES:")
    for i, enemy in enumerate(combat.enemies):
        alive = "" if enemy.is_alive else " [dead]"
        powers = f"  Powers: {', '.join(str(p) for p in enemy.powers.values())}" if enemy.powers else ""
        state = ""
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is not None:
            state = f"  Intent: {describe_enemy_intents(ai)}"
        print(f"    [{i}] {display_name(enemy.monster_id)}  HP {enemy.current_hp}/{enemy.max_hp}  Block {enemy.block}{alive}{state}{display_text(powers)}")

    if player.powers:
        print(f"\n  YOUR POWERS: {', '.join(str(power) for power in player.powers.values())}")

    print("\n  HAND:")
    for i, card in enumerate(combat.hand):
        playable = "*" if combat.can_play_card(card) else " "
        target_hint = " -> pick target" if card.target_type_for(player) in {TargetType.ANY_ENEMY, TargetType.ANY_ALLY} else ""
        print(f"    {playable} [{i}] {describe_card(card)}{target_hint}")

    if any(potion is not None for potion in combat.potions):
        print("\n  POTIONS:")
        for i, potion in enumerate(combat.potions):
            if potion is None:
                continue
            target_hint = " -> pick enemy" if potion.target_type == PotionTargetType.ANY_ENEMY else ""
            print(f"    [p{i}] {display_name(potion.potion_id)}{target_hint}")

    print(
        f"\n  Draw {len(combat.draw_pile)}  Discard {len(combat.discard_pile)}  "
        f"Exhaust {len(combat.exhaust_pile)}"
    )


def display_reward(mgr: RunManager) -> None:
    reward = getattr(mgr, "_current_reward", None)
    if reward is None:
        return
    print("\n  REWARD:")
    if isinstance(reward, CardReward):
        print("    Choose a card reward.")
        for i, card in enumerate(getattr(mgr, "_offered_cards", [])):
            print(f"    [{i}] {describe_card(card)}")
        return
    if isinstance(reward, CardBundlesReward):
        print("    Choose a card bundle.")
        for i, bundle in enumerate(getattr(mgr, "_offered_card_bundles", [])):
            names = ", ".join(describe_card(card) for card in bundle)
            print(f"    [{i}] {names}")
        return
    if isinstance(reward, PotionReward):
        potion = getattr(mgr, "_offered_potion", None)
        name = potion.potion_id if potion is not None else reward.potion_id
        print(f"    Potion: {display_name(name)}")
        return
    if isinstance(reward, RelicReward):
        print(f"    Relic: {display_name(reward.relic_id)}")


def display_boss_relics(mgr: RunManager) -> None:
    relics = getattr(mgr, "_boss_relics", [])
    if not relics:
        return
    print("\n  BOSS RELICS:")
    for i, relic_id in enumerate(relics):
        print(f"    [{i}] {display_name(relic_id)}")


def display_shop(mgr: RunManager) -> None:
    inv = getattr(mgr, "_shop_inventory", None)
    if inv is None:
        return
    print("\n  SHOP:")
    print("    Cards:")
    for i, entry in enumerate(inv.cards):
        sold = " [sold]" if not is_shop_entry_available(entry) else ""
        card = describe_card(entry.card) if entry.card is not None else display_name(entry.card_id)
        sale = " sale" if entry.on_sale else ""
        print(f"    [{i}] {card} - {entry.price}g{sale}{sold}")
    offset = len(inv.cards)
    print("    Colorless:")
    for i, entry in enumerate(inv.colorless_cards):
        sold = " [sold]" if not is_shop_entry_available(entry) else ""
        card = describe_card(entry.card) if entry.card is not None else display_name(entry.card_id)
        sale = " sale" if entry.on_sale else ""
        print(f"    [{offset + i}] {card} - {entry.price}g{sale}{sold}")
    print("    Relics:")
    for i, entry in enumerate(inv.relics):
        sold = " [sold]" if not is_shop_entry_available(entry) else ""
        print(f"    [{i}] {display_name(entry.relic_id)} - {entry.price}g [{entry.relic_rarity.name}]{sold}")
    print("    Potions:")
    for i, entry in enumerate(inv.potions):
        sold = " [sold]" if not is_shop_entry_available(entry) else ""
        print(f"    [{i}] {display_name(entry.potion_id)} - {entry.price}g [{entry.potion_rarity.name}]{sold}")
    removal_status = "used" if inv.removal_used else f"{inv.removal_cost}g"
    print(f"    Remove card: {removal_status}")


def display_rest_site(mgr: RunManager) -> None:
    options = getattr(mgr, "_rest_options", [])
    if not options:
        return
    print("\n  REST SITE:")
    for option in options:
        status = "" if option.enabled else " [disabled]"
        description = f" - {option.description}" if option.description else ""
        print(f"    {option.label}{description}{status}")


def display_event(mgr: RunManager) -> None:
    event = getattr(mgr, "_event_model", None)
    if event is None:
        return
    print(f"\n  EVENT: {display_name(event.event_id or event.__class__.__name__)}")
    for option in getattr(mgr, "_event_options", []):
        description = f" - {option.description}" if option.description else ""
        status = "" if option.enabled else " [disabled]"
        print(f"    {option.label}{description}{status}")


def display_treasure(mgr: RunManager) -> None:
    reward = getattr(mgr, "_current_reward", None)
    print("\n  TREASURE:")
    if isinstance(reward, RelicReward):
        print(f"    Chest contains {display_name(reward.relic_id)}.")
    else:
        print("    Open the chest.")


def display_pending_choices(mgr: RunManager) -> None:
    choice = mgr.run_state.pending_choice
    if choice is None:
        combat = mgr.get_combat_state()
        choice = combat.pending_choice if combat is not None else None
    if choice is None:
        event = getattr(mgr, "_event_model", None)
        choice = event.pending_choice if event is not None else None
    if choice is None:
        return
    print()
    print(f"  >>> {choice.prompt} <<<")
    if choice.is_multi:
        print(f"  Select {choice.min_choices}-{choice.max_choices}; selected: {sorted(choice.selected_indices)}")
    for i, option in enumerate(choice.options):
        selected = "[x]" if i in choice.selected_indices else "[ ]"
        print(f"    {selected} [{i}] {describe_card(option.card)}  ({option.source_pile})")


def display_inventory(mgr: RunManager) -> None:
    player = mgr.run_state.player
    print()
    print("  DECK:")
    for i, card in enumerate(player.deck):
        print(f"    [{i}] {describe_card(card)}")
    print("\n  RELICS:")
    if player.relics:
        for relic in player.relics:
            print(f"    {display_name(relic)}")
    else:
        print("    None")
    print("\n  POTIONS:")
    potions = player.potions
    if potions and any(potion is not None for potion in potions):
        for i, potion in enumerate(potions):
            if potion is not None:
                print(f"    [{i}] {display_name(potion.potion_id)}")
    else:
        print("    None")


def display_actions(actions: list[dict[str, Any]]) -> None:
    print("\n  ACTIONS:")
    for i, action in enumerate(actions):
        print(f"    [{i}] {describe_action(action)}")
    print("    h help   i inspect   q quit")


def display_state(mgr: RunManager, actions: list[dict[str, Any]]) -> None:
    display_header(mgr)
    combat = mgr.get_combat_state()
    if combat is not None:
        display_combat(combat)
    elif mgr.phase == RunManager.PHASE_MAP_CHOICE:
        display_map(mgr, actions)
    elif mgr.phase == RunManager.PHASE_CARD_REWARD:
        display_reward(mgr)
    elif mgr.phase == RunManager.PHASE_BOSS_RELIC:
        display_boss_relics(mgr)
    elif mgr.phase == RunManager.PHASE_SHOP:
        display_shop(mgr)
    elif mgr.phase == RunManager.PHASE_REST_SITE:
        display_rest_site(mgr)
    elif mgr.phase == RunManager.PHASE_EVENT:
        display_event(mgr)
    elif mgr.phase == RunManager.PHASE_TREASURE:
        display_treasure(mgr)
    display_pending_choices(mgr)
    display_actions(actions)


def choose_action(raw: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = raw.strip().lower()
    if not raw:
        return None
    if raw in QUIT_COMMANDS:
        raise KeyboardInterrupt
    if raw in HELP_COMMANDS:
        print_help()
        return None
    if raw in AUTO_COMMANDS:
        return actions[0] if actions else None
    if raw in CONFIRM_COMMANDS:
        for action in actions:
            if action.get("action") in {"confirm_choice", "skip", "skip_potion", "skip_relic", "end_turn", "leave_shop", "collect"}:
                return action
    try:
        index = int(raw)
    except ValueError:
        print("  Enter an action number, h for help, or q to quit.")
        return None
    if 0 <= index < len(actions):
        return actions[index]
    print(f"  Invalid action number. Choose 0-{len(actions) - 1}.")
    return None


def print_help() -> None:
    print()
    print("  Pick an action by number.")
    print("  Common shortcuts: a = first action, c/e = confirm/skip/end/leave, q = quit.")
    print("  Inspect: i = show deck, relics, and potions.")
    print("  Combat targets and reward choices are listed as separate numbered actions.")


def choose_character() -> str:
    print("\n  Choose your character:")
    for i, character in enumerate(CHARACTERS):
        print(f"    [{i}] {character}")
    while True:
        raw = input(f"  Character (default={DEFAULT_CHARACTER_INDEX}): ").strip().lower()
        if raw in QUIT_COMMANDS:
            raise KeyboardInterrupt
        if raw == "":
            return CHARACTERS[DEFAULT_CHARACTER_INDEX]
        try:
            index = int(raw)
        except ValueError:
            print("  Enter a character number or q to quit.")
            continue
        if 0 <= index < len(CHARACTERS):
            return CHARACTERS[index]
        print(f"  Invalid character number. Choose 0-{len(CHARACTERS) - 1}.")


def run_interactive(mgr: RunManager) -> int:
    last_description = ""
    while not mgr.is_over:
        actions = mgr.get_available_actions()
        if not actions:
            print("\n  No available actions; stopping.")
            return 1
        display_state(mgr, actions)
        if last_description:
            print(f"\n  Last: {display_text(last_description)}")
        while True:
            raw = input("\n  Action> ")
            if raw.strip().lower() in INSPECT_COMMANDS:
                display_inventory(mgr)
                continue
            try:
                action = choose_action(raw, actions)
            except KeyboardInterrupt:
                print("\n  Quit.")
                return 0
            if action is not None:
                break
        result = mgr.take_action(action)
        last_description = result.get("description", "")
        if last_description:
            print(f"\n  {display_text(last_description)}")

    display_header(mgr)
    print("  YOU WIN!" if mgr.player_won else "  YOU DIED.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a full STS2 run in the terminal.")
    parser.add_argument("--character", choices=CHARACTERS)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-neow",
        action="store_true",
        help="Start directly on the map, matching the old RunManager default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = args.seed if args.seed is not None else random.randint(0, INT_MAX)
    try:
        character = args.character or choose_character()
    except KeyboardInterrupt:
        print("\n  Quit.")
        return 0
    print("\n  === STS2 Full Run ===")
    print(f"  Character: {character}  Ascension: {args.ascension}  Seed: {seed}")
    mgr = RunManager(
        seed=seed,
        character_id=character,
        ascension_level=args.ascension,
        start_with_neow=not args.skip_neow,
        act1_variant="random",
    )
    return run_interactive(mgr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
