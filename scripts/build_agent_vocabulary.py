"""Build the collision-free agent vocabulary from simulator and Codex data."""

from __future__ import annotations

import argparse
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any

from sts2_env.agent_v2.categorical_vocabulary import (
    VOCABULARY_VERSION,
    canonical_token,
)
from sts2_env.agent_v2.tensorizer import ENTITY_TYPES
from sts2_env.core import enums
from sts2_env.relics.base import RelicId

STRUCTURAL_TOKENS = {
    "PAD", "UNKNOWN", "PLAYER_OWNER", "ENEMY_OWNER", "OTHER_OWNER",
    "PLAYER", "CARD", "CREATURE", "POWER", "RELIC", "POTION",
    "MAP_NODE", "CRYSTAL_CELL", "CHOICE", "OVERFLOW",
    "MAP_CHOICE", "COMBAT", "CARD_REWARD", "BOSS_RELIC", "SHOP",
    "REST_SITE", "EVENT", "TREASURE", "RUN_OVER",
    "COMBAT_ACTION", "RUN_COMPLETE", "GAME_STATE",
    "END_TURN_OR_CONFIRM", "USE_POTION", "PLAY_OR_CHOOSE", "MOVE",
    "CARD_REWARD_EXTRA", "SELECT_PLAYER", "TREASURE_OR_REROLL",
    "HAND", "DRAW", "DISCARD", "EXHAUST", "DECK", "CANDIDATE",
    "TRUE", "FALSE", "NONE", "COMPOSITE",
    "PLAY_CARD", "END_TURN", "CHOOSE", "PICK_CARD", "SKIP",
    "MAP_SELECT", "BUY_CARD", "BUY_POTION", "BUY_RELIC", "EVENT_CHOICE",
    "SELECT_CARD", "LEAVE_SHOP", "CARD_SELECT", "REMOVE_CARD",
    "CONFIRM_CHOICE", "USE_POTION", "REST_OPTION", "COLLECT",
    "PICK_RELIC", "PICK_RELIC_REWARD", "SKIP_RELIC", "PICK_POTION",
    "SKIP_POTION", "REROLL_CARD_REWARD", "PICK_CARD_BUNDLE",
    "CONFIRM", "SMITH", "REST", "LIFT", "DIG", "RECALL",
}

RELEVANT_KEYS = {
    "id", "type", "rarity", "target", "stack_type", "color", "pool",
    "card_type", "node_type", "action", "usage", "side", "zone",
    "character_id", "option_id", "keyword", "keywords", "tags",
    "applicable_to", "afflictions", "enchantments",
}


def _collect_json_tokens(value: Any, tokens: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in RELEVANT_KEYS or child_key.endswith("_id"):
                _collect_json_tokens(child, tokens, child_key)
            elif isinstance(child, (dict, list)):
                _collect_json_tokens(child, tokens, child_key)
        return
    if isinstance(value, list):
        for child in value:
            _collect_json_tokens(child, tokens, key)
        return
    if key in RELEVANT_KEYS or key.endswith("_id"):
        token = canonical_token(value)
        if token:
            tokens.add(token)


def build_tokens(codex_data: Path) -> list[str]:
    tokens = set(STRUCTURAL_TOKENS)
    tokens.update(ENTITY_TYPES)
    for _, enum_type in inspect.getmembers(enums, inspect.isclass):
        if issubclass(enum_type, Enum):
            tokens.update(member.name for member in enum_type)
    tokens.update(member.name for member in RelicId)
    for path in sorted(codex_data.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            _collect_json_tokens(json.load(handle), tokens)
    for index in range(1, 158):
        tokens.add(f"ACTION_SLOT:{index}")
    for level in range(17):
        tokens.add(f"UPGRADE:{level}")
    for count in range(1, 17):
        tokens.add(f"COUNT:{count}")
    for exponent in range(4, 64):
        tokens.add(f"COUNT_LOG:{exponent}")
    return sorted(canonical_token(token) for token in tokens if token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codex-data",
        type=Path,
        default=Path("../spire-codex/data-beta/v0.110.0/eng"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "sts2_env/agent_v2/categorical_vocabulary_v1.json"
        ),
    )
    args = parser.parse_args()
    tokens = build_tokens(args.codex_data)
    payload = {"version": VOCABULARY_VERSION, "tokens": tokens}
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tokens)} unique tokens to {args.output}")


if __name__ == "__main__":
    main()
