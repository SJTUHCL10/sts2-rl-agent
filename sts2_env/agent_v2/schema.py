"""Stable schema and vocabulary metadata for the v2 agent interface.

The v1 float-vector layouts remain untouched.  V2 uses JSON-compatible typed
entities and semantic action candidates; tensors are a model-layer concern.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Iterable

from sts2_env.core.enums import CardId, PowerId
from sts2_env.relics.base import RelicId

PROTOCOL_VERSION = 2
OBSERVATION_SCHEMA_VERSION = "sts2-entity-v2"
ACTION_SCHEMA_VERSION = "candidate-v2"

# Hard limits are batching/validation limits, not action semantics.  Increasing
# one creates a new feature-layout hash and therefore fails checkpoint loading.
ENTITY_LIMITS = {
    "players": 7,
    "creatures": 16,
    "cards": 256,
    "powers": 256,
    "relics": 128,
    "potions": 16,
    "map_nodes": 128,
    "map_edges": 256,
    "crystal_cells": 121,
    "candidates": 256,
}

ENTITY_FIELDS = {
    "global": (
        "phase", "character_id", "act", "act_floor", "floor", "ascension",
        "room_type", "is_over", "player_won",
    ),
    "player": (
        "entity_id", "player_id", "character_id", "hp", "max_hp", "block",
        "energy", "max_energy", "gold", "stars", "max_potion_slots",
    ),
    "card": (
        "entity_id", "card_id", "owner_id", "zone", "zone_index", "cost",
        "original_cost", "card_type", "target_type", "rarity", "base_damage",
        "base_block", "upgrade_level", "upgraded", "playable", "keywords",
        "tags", "enchantments", "afflictions", "dynamic_vars", "effect_vars",
        "combat_vars",
    ),
    "creature": (
        "entity_id", "owner_id", "monster_id", "side", "hp", "max_hp",
        "block", "alive", "stars", "intents",
    ),
    "power": (
        "entity_id", "owner_id", "power_id", "power_type", "stack_type",
        "amount", "display_amount", "amount_on_turn_start",
        "skip_next_duration_tick", "is_visible", "dynamic_vars", "counters",
    ),
    "relic": (
        "entity_id", "owner_id", "relic_id", "rarity", "stack_count",
        "status", "is_used_up", "is_melted", "is_wax", "show_counter",
        "display_amount", "floor_added", "dynamic_vars", "counters", "state",
    ),
    "potion": (
        "entity_id", "owner_id", "slot", "potion_id", "rarity", "usage",
        "target_type", "can_use",
    ),
    "map_node": (
        "entity_id", "row", "col", "node_type", "visited", "reachable",
    ),
    "crystal_cell": (
        "entity_id", "x", "y", "hidden", "clickable", "revealed_item_id",
    ),
    "crystal_minigame": (
        "grid_width", "grid_height", "divinations_remaining", "tool",
        "finished", "placed_all_items", "revealed_items",
    ),
    "candidate": (
        "candidate_id", "action_type", "source_id", "target_id", "enabled",
        "payload", "features",
    ),
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vocabulary_hash(values: Iterable[str]) -> str:
    """Hash a sorted, duplicate-free vocabulary."""
    return _canonical_hash(sorted(set(values)))


def enum_vocabulary(enum_type: type[Enum]) -> tuple[str, ...]:
    return tuple(member.name for member in enum_type)


def schema_manifest() -> dict[str, Any]:
    """Return checkpoint/replay compatibility metadata."""
    vocabularies = {
        "cards": enum_vocabulary(CardId),
        "powers": enum_vocabulary(PowerId),
        "relics": enum_vocabulary(RelicId),
    }
    feature_layout = {
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "entity_limits": ENTITY_LIMITS,
        "entity_fields": ENTITY_FIELDS,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "feature_layout_hash": _canonical_hash(feature_layout),
        "vocabulary_hashes": {
            name: vocabulary_hash(values)
            for name, values in vocabularies.items()
        },
        "vocabulary_sizes": {
            name: len(values)
            for name, values in vocabularies.items()
        },
    }


def validate_v2_envelope(state: dict[str, Any]) -> None:
    """Raise a descriptive error when a v2 message is incompatible."""
    expected = schema_manifest()
    for key in ("protocol_version", "observation_schema", "action_schema"):
        if state.get(key) != expected[key]:
            raise ValueError(
                f"Incompatible {key}: expected {expected[key]!r}, "
                f"got {state.get(key)!r}"
            )
    supplied_hash = state.get("feature_layout_hash")
    if supplied_hash is not None and supplied_hash != expected["feature_layout_hash"]:
        raise ValueError(
            "Incompatible feature_layout_hash: "
            f"expected {expected['feature_layout_hash']}, got {supplied_hash}"
        )
