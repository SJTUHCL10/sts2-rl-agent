"""Small local web UI for playing a full STS2 run."""

from __future__ import annotations

import argparse
import json
import random
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import sts2_env.events  # noqa: F401
import sts2_env.potions.effects  # noqa: F401

from sts2_env.cli.play_run import (
    CHARACTERS,
    DEFAULT_CHARACTER_INDEX,
    PHASE_LABELS,
    ROOM_LABELS,
    describe_action,
    describe_card,
    describe_enemy_intents,
    display_name,
    display_text,
)
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import PotionTargetType, TargetType
from sts2_env.core.rng import INT_MAX
from sts2_env.run.reward_objects import CardBundlesReward, CardReward, PotionReward, RelicReward
from sts2_env.run.run_manager import RunManager
from sts2_env.run.shop import is_shop_entry_available


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATE_SESSION_ID = "default"
START_PHASE = "START"
START_PHASE_LABEL = "Start"
START_SCREEN_TITLE = "Start Run"
START_NOTICE = "Ready to start."


class RunSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mgr: RunManager | None = None
        self.seed: int | None = None
        self.character = CHARACTERS[DEFAULT_CHARACTER_INDEX]
        self.ascension = 0
        self.last_description = ""

    def start(self, *, character: str, seed: int | None, ascension: int = 0, skip_neow: bool = False) -> dict:
        with self.lock:
            self.seed = seed if seed is not None else random.randint(0, INT_MAX)
            self.character = character if character in CHARACTERS else CHARACTERS[DEFAULT_CHARACTER_INDEX]
            self.ascension = ascension
            self.last_description = ""
            self.mgr = RunManager(
                seed=self.seed,
                character_id=self.character,
                ascension_level=self.ascension,
                start_with_neow=not skip_neow,
                act1_variant="random",
            )
            return self.state()

    def take_action(self, action_index: int) -> dict:
        with self.lock:
            if self.mgr is None:
                self.last_description = "Start a new run first."
                return self.state()
            mgr = self.mgr
            actions = mgr.get_available_actions()
            if action_index < 0 or action_index >= len(actions):
                self.last_description = "Invalid action."
                return self.state()
            result = mgr.take_action(actions[action_index])
            self.last_description = result.get("description", "")
            return self.state()

    def state(self) -> dict:
        if self.mgr is None:
            return serialize_start_state(
                seed=self.seed,
                character=self.character,
                ascension=self.ascension,
                last_description=self.last_description,
            )
        mgr = self.mgr
        actions = mgr.get_available_actions()
        return serialize_run(
            mgr,
            actions,
            seed=self.seed,
            character=self.character,
            ascension=self.ascension,
            last_description=self.last_description,
        )


def serialize_start_state(
    *,
    seed: int | None,
    character: str,
    ascension: int,
    last_description: str,
) -> dict:
    return {
        "session_id": STATE_SESSION_ID,
        "seed": seed,
        "character": character,
        "ascension": ascension,
        "phase": START_PHASE,
        "phase_label": START_PHASE_LABEL,
        "act": "-",
        "floor": "-",
        "hp": "-",
        "gold": "-",
        "deck_size": "-",
        "relics": [],
        "is_over": False,
        "player_won": False,
        "last_description": last_description or START_NOTICE,
        "actions": [],
        "screen": {"type": "start", "title": START_SCREEN_TITLE, "items": []},
        "inventory": {"deck": [], "relics": [], "potions": []},
    }


def serialize_run(
    mgr: RunManager,
    actions: list[dict[str, Any]],
    *,
    seed: int | None,
    character: str,
    ascension: int,
    last_description: str,
) -> dict:
    summary = mgr.summary()
    combat = mgr.get_combat_state()
    return {
        "session_id": STATE_SESSION_ID,
        "seed": seed,
        "character": character,
        "ascension": ascension,
        "phase": summary["phase"],
        "phase_label": PHASE_LABELS.get(summary["phase"], summary["phase"]),
        "act": summary["act"] + 1,
        "floor": summary["floor"],
        "hp": summary["hp"],
        "gold": summary["gold"],
        "deck_size": summary["deck_size"],
        "relics": [display_name(relic) for relic in summary["relics"]],
        "is_over": summary["is_over"],
        "player_won": summary["player_won"],
        "last_description": last_description,
        "actions": [
            {"index": index, "label": describe_action(action), "kind": action.get("action", "")}
            for index, action in enumerate(actions)
        ],
        "screen": _screen(mgr, actions, combat),
        "inventory": _inventory(mgr),
    }


def _screen(mgr: RunManager, actions: list[dict[str, Any]], combat: CombatState | None) -> dict:
    if any(action.get("action") in {"choose", "confirm_choice"} for action in actions):
        return _choice_screen(actions)
    if combat is not None:
        return _combat_screen(combat, actions)
    phase = mgr.phase
    if phase == RunManager.PHASE_MAP_CHOICE:
        return _map_screen(mgr, actions)
    if phase == RunManager.PHASE_CARD_REWARD:
        return _reward_screen(mgr, actions)
    if phase == RunManager.PHASE_BOSS_RELIC:
        return {
            "type": "boss_relic",
            "title": "Boss Relics",
            "items": [
                {
                    "name": display_name(relic_id),
                    "action_index": _find_action_index(actions, action="pick_relic", index=index),
                }
                for index, relic_id in enumerate(getattr(mgr, "_boss_relics", []))
            ],
        }
    if phase == RunManager.PHASE_SHOP:
        return _shop_screen(mgr, actions)
    if phase == RunManager.PHASE_REST_SITE:
        return _rest_screen(mgr, actions)
    if phase == RunManager.PHASE_EVENT:
        return _event_screen(mgr, actions)
    if phase == RunManager.PHASE_TREASURE:
        return _treasure_screen(mgr, actions)
    return {"type": "message", "title": PHASE_LABELS.get(phase, phase), "items": []}


def _choice_screen(actions: list[dict[str, Any]]) -> dict:
    prompt = next(
        (action.get("prompt") for action in actions if action.get("prompt")),
        "Choose",
    )
    items = []
    for index, action in enumerate(actions):
        action_type = action.get("action")
        if action_type == "confirm_choice":
            selected_count = action.get("selected_count", 0)
            items.append({
                "name": "Confirm" if selected_count else "Skip",
                "description": f"{selected_count} selected",
                "selected": False,
                "action_index": index,
            })
        elif action_type == "choose":
            items.append({
                "name": display_name(action.get("card_id", action.get("index"))),
                "description": action.get("source_pile", ""),
                "selected": bool(action.get("selected")),
                "action_index": index,
            })
    return {
        "type": "choice",
        "title": display_text(str(prompt)),
        "items": items,
    }


def _combat_screen(combat: CombatState, actions: list[dict[str, Any]]) -> dict:
    player = combat.primary_player
    end_turn_action_index = _find_action_index(actions, action="end_turn")
    enemies = []
    for index, enemy in enumerate(combat.enemies):
        ai = combat.enemy_ais.get(enemy.combat_id)
        enemies.append({
            "index": index,
            "name": display_name(enemy.monster_id),
            "hp": f"{enemy.current_hp}/{enemy.max_hp}",
            "block": enemy.block,
            "intent": describe_enemy_intents(ai),
            "alive": enemy.is_alive,
            "powers": [str(power) for power in enemy.powers.values()],
        })
    return {
        "type": "combat",
        "title": "Combat",
        "round": combat.round_number,
        "energy": combat.energy,
        "end_turn_action_index": end_turn_action_index,
        "player": {
            "hp": f"{player.current_hp}/{player.max_hp}",
            "block": player.block,
            "powers": [str(power) for power in player.powers.values()],
        },
        "enemies": enemies,
        "hand": [
            {
                "index": index,
                "name": describe_card(card),
                "playable": combat.can_play_card(card),
                "targeted": card.target_type_for(player) in {TargetType.ANY_ENEMY, TargetType.ANY_ALLY},
                "actions": _combat_card_actions(actions, index),
            }
            for index, card in enumerate(combat.hand)
        ],
        "potions": [
            {
                "index": index,
                "name": display_name(potion.potion_id),
                "targeted": potion.target_type == PotionTargetType.ANY_ENEMY,
                "actions": _combat_potion_actions(actions, index),
            }
            for index, potion in enumerate(combat.potions)
            if potion is not None
        ],
        "piles": {
            "draw": len(combat.draw_pile),
            "discard": len(combat.discard_pile),
            "exhaust": len(combat.exhaust_pile),
        },
    }


def _combat_card_actions(actions: list[dict[str, Any]], hand_index: int) -> list[dict[str, Any]]:
    card_actions = []
    for index, action in enumerate(actions):
        if action.get("action") != "play_card" or action.get("hand_index") != hand_index:
            continue
        target_name = action.get("target_name")
        card_actions.append({
            "action_index": index,
            "target": display_name(target_name) if target_name else "Play",
        })
    return card_actions


def _combat_potion_actions(actions: list[dict[str, Any]], slot_index: int) -> list[dict[str, Any]]:
    potion_actions = []
    for index, action in enumerate(actions):
        if action.get("action") != "use_potion" or action.get("slot_index") != slot_index:
            continue
        target_name = action.get("target_name")
        potion_actions.append({
            "action_index": index,
            "target": display_name(target_name) if target_name else "Use",
        })
    return potion_actions


def _map_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    run_state = mgr.run_state
    act_map = run_state.map
    if act_map is None:
        return {"type": "map", "title": "Map", "columns": [], "rows": [], "paths": []}
    visited = set(run_state.visited_map_coords)
    reachable = {
        tuple(action["coord"]): index
        for index, action in enumerate(actions)
        if action.get("action") == "move"
    }
    rows = []
    all_points = act_map.all_points()
    max_row = max((point.row for point in all_points), default=0)
    columns = sorted({point.col for point in all_points})
    for row_index in range(max_row, 0, -1):
        row = []
        for point in act_map.get_row(row_index):
            coord = (point.col, point.row)
            row.append({
                "coord": coord,
                "label": ROOM_LABELS.get(point.point_type.name, display_name(point.point_type.name)),
                "reachable": coord in reachable,
                "action_index": reachable.get(coord),
                "visited": point.coord in visited,
            })
        if row:
            rows.append(row)
    paths = []
    for coord in run_state.get_available_next_coords():
        point = act_map.get_point(coord)
        if point is None:
            continue
        paths.append({
            "coord": (point.col, point.row),
            "label": ROOM_LABELS.get(point.point_type.name, display_name(point.point_type.name)),
            "next": [
                {
                    "coord": (child.col, child.row),
                    "label": ROOM_LABELS.get(child.point_type.name, display_name(child.point_type.name)),
                }
                for child in sorted(point.children, key=lambda child: child.col)
            ],
        })
    current = None
    if run_state.visited_map_coords:
        coord = run_state.visited_map_coords[-1]
        current = (coord.col, coord.row)
    return {"type": "map", "title": "Map", "current": current, "columns": columns, "rows": rows, "paths": paths}


def _reward_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    reward = getattr(mgr, "_current_reward", None)
    alternative_items = _reward_alternative_items(actions)
    if isinstance(reward, CardReward):
        return {
            "type": "reward",
            "title": "Card Reward",
            "items": alternative_items + [
                {
                    "name": describe_card(card),
                    "action_index": _find_action_index(actions, action="pick_card", index=index),
                }
                for index, card in enumerate(getattr(mgr, "_offered_cards", []))
            ],
        }
    if isinstance(reward, CardBundlesReward):
        return {
            "type": "reward",
            "title": "Card Bundle",
            "items": alternative_items + [
                {
                    "name": ", ".join(describe_card(card) for card in bundle),
                    "action_index": _find_action_index(actions, action="pick_card_bundle", index=index),
                }
                for index, bundle in enumerate(getattr(mgr, "_offered_card_bundles", []))
            ],
        }
    if isinstance(reward, PotionReward):
        potion = getattr(mgr, "_offered_potion", None)
        return {
            "type": "reward",
            "title": "Potion Reward",
            "items": alternative_items + [
                {
                    "name": display_name(potion.potion_id if potion else reward.potion_id),
                    "action_index": _find_action_index(actions, action="pick_potion"),
                },
            ],
        }
    if isinstance(reward, RelicReward):
        return {
            "type": "reward",
            "title": "Relic Reward",
            "items": alternative_items + [
                {
                    "name": display_name(reward.relic_id),
                    "action_index": _find_action_index(actions, action="pick_relic_reward"),
                },
            ],
        }
    return {"type": "reward", "title": "Reward", "items": []}


def _reward_alternative_items(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alternatives = []
    labels = {
        "skip": "Skip",
        "skip_potion": "Skip potion",
        "skip_relic": "Skip relic",
        "reroll_card_reward": "Reroll",
        "sacrifice_card_reward": "Sacrifice",
    }
    for index, action in enumerate(actions):
        action_type = action.get("action")
        if action_type in labels:
            alternatives.append({"name": labels[action_type], "action_index": index})
    return alternatives


def _shop_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    inv = getattr(mgr, "_shop_inventory", None)
    if inv is None:
        return {"type": "shop", "title": "Shop", "sections": []}
    return {
        "type": "shop",
        "title": "Shop",
        "sections": [
            {
                "title": "Exit",
                "items": [
                    {
                        "name": "Leave shop",
                        "action_index": _find_action_index(actions, action="leave_shop"),
                    },
                ],
            },
            {
                "title": "Cards",
                "items": [
                    _shop_item(
                        describe_card(entry.card) if entry.card is not None else display_name(entry.card_id),
                        entry.price,
                        entry,
                        action_index=_find_action_index(actions, action="buy_card", index=index),
                    )
                    for index, entry in enumerate([*inv.cards, *inv.colorless_cards])
                ],
            },
            {
                "title": "Relics",
                "items": [
                    _shop_item(
                        display_name(entry.relic_id),
                        entry.price,
                        entry,
                        action_index=_find_action_index(actions, action="buy_relic", index=index),
                    )
                    for index, entry in enumerate(inv.relics)
                ],
            },
            {
                "title": "Potions",
                "items": [
                    _shop_item(
                        display_name(entry.potion_id),
                        entry.price,
                        entry,
                        action_index=_find_action_index(actions, action="buy_potion", index=index),
                    )
                    for index, entry in enumerate(inv.potions)
                ],
            },
            {
                "title": "Services",
                "items": [
                    {
                        "name": "Remove card",
                        "price": inv.removal_cost,
                        "sold": inv.removal_used,
                        "action_index": _find_action_index(actions, action="remove_card"),
                    },
                ],
            },
        ],
    }


def _shop_item(name: str, price: int, entry: object, *, action_index: int | None) -> dict:
    return {
        "name": name,
        "price": price,
        "sold": not is_shop_entry_available(entry),
        "action_index": action_index,
    }


def _rest_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    return {
        "type": "rest",
        "title": "Rest Site",
        "items": [
            {
                "name": option.label,
                "description": option.description,
                "enabled": option.enabled,
                "action_index": _find_action_index(actions, action="rest_option", option_id=option.option_id),
            }
            for option in getattr(mgr, "_rest_options", [])
        ],
    }


def _event_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    event = getattr(mgr, "_event_model", None)
    return {
        "type": "event",
        "title": display_name(event.event_id if event is not None else "Event"),
        "items": [
            {
                "name": option.label,
                "description": option.description,
                "enabled": option.enabled,
                "action_index": _find_action_index(actions, action="event_choice", option_id=option.option_id),
            }
            for option in getattr(mgr, "_event_options", [])
        ],
    }


def _treasure_screen(mgr: RunManager, actions: list[dict[str, Any]]) -> dict:
    reward = getattr(mgr, "_current_reward", None)
    item = display_name(reward.relic_id) if isinstance(reward, RelicReward) else "Chest"
    return {
        "type": "treasure",
        "title": "Treasure",
        "items": [{"name": item, "action_index": _find_action_index(actions, action="collect")}],
    }


def _find_action_index(actions: list[dict[str, Any]], **criteria: Any) -> int | None:
    for index, action in enumerate(actions):
        if all(action.get(key) == value for key, value in criteria.items()):
            return index
    return None


def _inventory(mgr: RunManager) -> dict:
    player = mgr.run_state.player
    return {
        "deck": [describe_card(card) for card in player.deck],
        "relics": [display_name(relic) for relic in player.relics],
        "potions": [
            {"index": index, "name": display_name(potion.potion_id)}
            for index, potion in enumerate(player.potions)
            if potion is not None
        ],
    }


class PlayRunHandler(BaseHTTPRequestHandler):
    session = RunSession()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(STATIC_DIR / "play_run.html", "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(self.session.state())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/new":
            query = parse_qs(parsed.query)
            seed = _optional_int(query.get("seed", [""])[0])
            ascension = _optional_int(query.get("ascension", ["0"])[0]) or 0
            character = query.get("character", [CHARACTERS[DEFAULT_CHARACTER_INDEX]])[0]
            skip_neow = query.get("skip_neow", ["0"])[0] in {"1", "true", "yes"}
            self._send_json(self.session.start(
                character=character,
                seed=seed,
                ascension=ascension,
                skip_neow=skip_neow,
            ))
            return
        if parsed.path == "/api/action":
            query = parse_qs(parsed.query)
            action_index = _optional_int(query.get("index", ["-1"])[0])
            self._send_json(self.session.take_action(action_index if action_index is not None else -1))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), PlayRunHandler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the STS2 full-run web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = make_server(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving STS2 full-run web UI at {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
