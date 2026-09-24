"""Fixed-shape tensors for the entity-v2 agent interface.

The wire protocol intentionally stays JSON-shaped. This module is the model
boundary: it converts a snapshot into padded, permutation-friendly entity and
candidate sets without changing the legacy v1 environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from gymnasium import spaces

from sts2_env.agent_v2.categorical_vocabulary import (
    UNKNOWN_ID,
    VOCABULARY_HASH,
    VOCABULARY_SIZE,
    categorical_id as _uncached_categorical_id,
    canonical_token,
)
from sts2_env.agent_v2.unknown_diagnostics import UnknownTokenDiagnostics

from sts2_env.core.constants import (
    ACTION_END_TURN,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    POTION_ACTION_START,
    POTION_TARGET_OPTIONS,
)
from sts2_env.gym_env.run_env import (
    ENTITY_TOTAL_ACTIONS,
    TOTAL_ACTIONS,
    _BOSS_RELIC_START,
    _CARD_RWD_EXTRA_START,
    _CARD_RWD_REROLL,
    _CARD_RWD_START,
    _COMBAT_SIZE,
    _EVENT_START,
    _MAP_START,
    _PLAYER_SELECT_START,
    _REST_START,
    _SHOP_START,
    _TREASURE_START,
)

TENSOR_ENCODING_VERSION = "typed-set-tensor-v6"
MAX_CURRENT_INTENTS = 3
_BRIDGE_PHASES = {
    "map_select": "MAP_CHOICE",
    "combat_action": "COMBAT",
    "card_reward": "CARD_REWARD",
    "reward_screen": "CARD_REWARD",
    "card_bundle": "CARD_REWARD",
    "card_select": "CARD_REWARD",
    "boss_relic": "BOSS_RELIC",
    "shop": "SHOP",
    "rest_site": "REST_SITE",
    "event": "EVENT",
    "crystal_sphere": "EVENT",
    "treasure": "TREASURE",
    "run_complete": "RUN_OVER",
}


def _snapshot_phase(snapshot: dict[str, Any]) -> str:
    global_state = snapshot.get("global")
    global_phase = global_state.get("phase") if isinstance(global_state, dict) else None
    return str(
        snapshot.get("phase") or global_phase
        or _BRIDGE_PHASES.get(str(snapshot.get("type", "")).casefold(), "")
    ).upper()


@dataclass(frozen=True, slots=True)
class TensorizerConfig:
    """Shape and vocabulary contract stored with a v2 checkpoint."""

    max_entities: int = 384
    num_actions: int = ENTITY_TOTAL_ACTIONS
    categorical_vocab_size: int = VOCABULARY_SIZE
    categorical_vocabulary_hash: str = VOCABULARY_HASH
    entity_categorical_fields: int = 16
    entity_numeric_fields: int = 45
    candidate_categorical_fields: int = 7
    candidate_numeric_fields: int = 15
    global_categorical_fields: int = 4
    global_numeric_fields: int = 16

    def __post_init__(self) -> None:
        if self.categorical_vocab_size != VOCABULARY_SIZE:
            raise ValueError(
                "Categorical vocabulary size mismatch: "
                f"expected {VOCABULARY_SIZE}, got "
                f"{self.categorical_vocab_size}"
            )
        if self.categorical_vocabulary_hash != VOCABULARY_HASH:
            raise ValueError(
                "Categorical vocabulary hash mismatch: "
                f"expected {VOCABULARY_HASH}, got "
                f"{self.categorical_vocabulary_hash}"
            )

    def feature_layout_hash(self) -> str:
        return _feature_layout_hash(self, TENSOR_ENCODING_VERSION)


def _feature_layout_hash(config: TensorizerConfig, version: str) -> str:
    payload = json.dumps(
        {
            "encoding_version": version,
            "config": asdict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LegacyV5TensorizerConfig(TensorizerConfig):
    """Read-only compatibility for tracing pre-modifier-fix checkpoints."""

    entity_numeric_fields: int = 43

    def feature_layout_hash(self) -> str:
        return _feature_layout_hash(self, "typed-set-tensor-v5")


DEFAULT_TENSORIZER_CONFIG = TensorizerConfig()

ENTITY_TYPES = (
    "PAD",
    "PLAYER",
    "CARD",
    "CREATURE",
    "POWER",
    "RELIC",
    "POTION",
    "MAP_NODE",
    "CRYSTAL_CELL",
    "CHOICE",
    "OVERFLOW",
)
ENTITY_TYPE_TO_ID = {
    name: _uncached_categorical_id(name)
    for name in ENTITY_TYPES
}


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _ratio(value: Any, scale: float) -> float:
    return max(-10.0, min(10.0, _finite(value) / scale))


def _signed_log(value: Any, scale: float = 5.0) -> float:
    number = _finite(value)
    transformed = math.copysign(math.log1p(abs(number)) / scale, number)
    return max(-10.0, min(10.0, transformed))


@lru_cache(maxsize=8192)
def _cached_categorical_id(value: Any, strict: bool) -> int:
    """Cache the highly repetitive scalar vocabulary lookups."""
    return _uncached_categorical_id(value, strict=strict)


def categorical_id(value: Any, *, strict: bool = False) -> int:
    """Encode a scalar category, falling back for an unhashable value."""
    try:
        return _cached_categorical_id(value, strict)
    except TypeError:
        return _uncached_categorical_id(value, strict=strict)


def _categorical_fields(
    values: list[Any],
    names: tuple[str, ...],
    context: dict[str, Any],
    diagnostics: UnknownTokenDiagnostics | None,
) -> np.ndarray:
    if diagnostics is None:
        return np.asarray([categorical_id(value) for value in values], dtype=np.int32)
    encoded = []
    for name, value in zip(names, values, strict=True):
        category = categorical_id(value)
        if category == UNKNOWN_ID:
            diagnostics.record(name, value, context)
        encoded.append(category)
    return np.asarray(encoded, dtype=np.int32)


_GLOBAL_CATEGORY_FIELDS = (
    "global.phase", "global.character", "global.room_type", "global.screen_type",
)
_ENTITY_CATEGORY_FIELDS = (
    "entity.type", "entity.content", "entity.zone", "entity.subtype",
    "entity.rarity", "entity.owner", "entity.upgrade", "entity.affliction_0",
    "entity.affliction_1", "entity.enchantment_0", "entity.enchantment_1",
    "entity.status", "entity.count", "entity.intent_0", "entity.intent_1",
    "entity.intent_2",
)
_CANDIDATE_CATEGORY_FIELDS = (
    "candidate.action_type", "candidate.action_slot", "candidate.source_content",
    "candidate.target_content", "candidate.phase", "candidate.option",
    "candidate.source_zone_or_action",
)


def _content_id(entity: dict[str, Any]) -> Any:
    for key in (
        "model_content", "card_id", "monster_id", "power_id", "relic_id", "potion_id",
        "node_type", "revealed_item_id", "character_id", "id", "type",
    ):
        if entity.get(key) not in (None, ""):
            return entity[key]
    return None


def _owner_token(owner_id: Any) -> str | None:
    if owner_id is None:
        return None
    text = str(owner_id).casefold()
    if text.startswith("player:"):
        return "PLAYER_OWNER"
    if text.startswith("enemy:"):
        return "ENEMY_OWNER"
    return "OTHER_OWNER"


def _count_token(count: int) -> str:
    if count <= 1:
        return "COUNT:1"
    if count <= 16:
        return f"COUNT:{count}"
    return f"COUNT_LOG:{min(63, int(math.log2(count)))}"


def _modifier_tokens(value: Any, limit: int = 2) -> list[str | None]:
    if isinstance(value, dict):
        raw = value.keys()
    elif isinstance(value, (list, tuple, set)):
        raw = value
    elif value in (None, ""):
        raw = ()
    else:
        raw = (value,)
    tokens = sorted({canonical_token(item) for item in raw if item})[:limit]
    return [*tokens, *([None] * (limit - len(tokens)))]


def _single_modifier(value: Any) -> tuple[str | None, float]:
    """Extract the game's one modifier type and its stack amount."""
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, (list, tuple, set)):
        items = [(item, 1) for item in value]
    elif value in (None, ""):
        items = []
    else:
        items = [(value, 1)]
    if len(items) > 1:
        raise ValueError(f"A card has multiple modifiers of one kind: {items!r}")
    if not items:
        return None, 0.0
    name, amount = items[0]
    return canonical_token(name), _finite(amount, 1)


def _semantic_card_key(card: dict[str, Any]) -> str:
    ignored = {"entity_id", "zone_index", "count", "playable"}
    payload = {
        key: value for key, value in card.items()
        if key not in ignored
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )


def _aggregate_pile_cards(
    cards: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate semantically identical cards within non-actionable piles."""
    result: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for raw in cards:
        card = dict(raw)
        zone = str(card.get("zone", "")).casefold()
        if zone not in {"deck", "draw", "discard", "exhaust"}:
            card.setdefault("count", 1)
            result.append(card)
            continue
        key = _semantic_card_key(card)
        if key not in groups:
            card["count"] = 1
            digest = hashlib.sha1(key.encode()).hexdigest()[:16]
            card["entity_id"] = f"pile-group:{digest}"
            groups[key] = card
        else:
            groups[key]["count"] += 1
    result.extend(groups[key] for key in sorted(groups))
    return result


def _iter_snapshot_entities(
    snapshot: dict[str, Any],
    *,
    legacy_v5: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Collect current-screen entities first, then persistent run entities."""
    collected: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()

    def extend(container: dict[str, Any], key: str, entity_type: str) -> None:
        if key == "hand" and container.get("cards"):
            return  # Simulator already supplies the same instances in cards.
        for index, raw in enumerate(container.get(key, []) or []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if legacy_v5 and entity_type == "CARD":
                # v5 simulator snapshots always emitted an empty mapping.
                item["afflictions"] = {}
            if entity_type == "CREATURE":
                item.setdefault("combat_index", index)
            elif key == "hand":
                item.setdefault("zone", "hand")
                item.setdefault("zone_index", index)
            identity = str(
                item.get("entity_id")
                or _fallback_entity_id(key, item, index)
            )
            item.setdefault("entity_id", identity)
            marker = (entity_type, identity)
            if marker in seen:
                continue
            seen.add(marker)
            collected.append((entity_type, item))

    current_specs = (
        ("players", "PLAYER"),
        ("creatures", "CREATURE"),
        ("enemies", "CREATURE"),
        ("hand", "CARD"),
        ("cards", "CARD"),
        ("powers", "POWER"),
        ("relics", "RELIC"),
        ("potions", "POTION"),
        ("crystal_cells", "CRYSTAL_CELL"),
        ("map_nodes", "MAP_NODE"),
    )
    for key, entity_type in current_specs:
        extend(snapshot, key, entity_type)

    run_state = snapshot.get("run_state")
    if isinstance(run_state, dict):
        for key, entity_type in current_specs:
            extend(run_state, key, entity_type)

    for key in ("options", "nodes", "bundles"):
        if legacy_v5 and snapshot.get("reward_item_type") and key == "options":
            continue  # Old checkpoints saw no offered potion/relic choice row.
        for index, raw in enumerate(snapshot.get(key, []) or []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item.setdefault("zone", "candidate")
            item.setdefault("zone_index", index)
            item.setdefault(
                "entity_id",
                _fallback_entity_id(key, item, index),
            )
            collected.append(("CHOICE", item))

    if not legacy_v5:
        event_context = snapshot.get("event_context") or []
        if not event_context and _snapshot_phase(snapshot) == "EVENT":
            event_id = snapshot.get("event_id") or next(
                (
                    item.get("event_id")
                    for item in snapshot.get("options", []) or []
                    if isinstance(item, dict) and item.get("event_id")
                ),
                None,
            )
            if event_id:
                event_context = [{
                    "entity_id": f"event:{event_id}",
                    "id": event_id,
                    "zone": "event",
                }]
        for item in event_context:
            collected.append(("CHOICE", item))

    # Preserve current-screen order and priority. Only persistent deck copies
    # move into the compressed tail; hand/draw/discard/exhaust instances stay
    # individually represented and cannot be displaced by map/deck bulk.
    result: list[tuple[str, dict[str, Any]]] = []
    compressible_cards: list[dict[str, Any]] = []
    for entity_type, item in collected:
        if (
            entity_type == "CARD"
            and str(item.get("zone", "")).casefold()
            in {"deck", "draw", "discard", "exhaust"}
        ):
            compressible_cards.append(item)
        else:
            result.append((entity_type, item))
    result.extend(
        ("CARD", card)
        for card in _aggregate_pile_cards(compressible_cards)
    )
    return result


def _fallback_entity_id(key: str, item: dict[str, Any], index: int) -> str:
    prefix = "enemy" if key == "enemies" else key.rstrip("s")
    return f"{prefix}:{item.get('id', 'unknown')}:{item.get('index', index)}"


def _entity_row(
    entity_type: str,
    entity: dict[str, Any],
    config: TensorizerConfig,
    diagnostics: UnknownTokenDiagnostics | None = None,
    phase: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, int(_finite(entity.get("count"), 1)))
    legacy_v5 = isinstance(config, LegacyV5TensorizerConfig)
    if legacy_v5:
        # Simulator v5 snapshots accidentally omitted card.affliction.
        afflictions = [None, None] if entity_type == "CARD" else _modifier_tokens(entity.get("afflictions"))
        enchantments = _modifier_tokens(entity.get("enchantments"))
        affliction_amount = enchantment_amount = 0.0
    else:
        affliction, affliction_amount = _single_modifier(entity.get("afflictions"))
        enchantment, enchantment_amount = _single_modifier(entity.get("enchantments"))
        afflictions = [affliction, None]
        enchantments = [enchantment, None]
    intents = (entity.get("intents") or []) if entity_type == "CREATURE" else []
    if not intents and entity_type == "CREATURE" and entity.get("intent"):
        intents = [{
            "intent_type": entity["intent"],
            "damage": entity.get("intent_damage", 0),
            "hits": entity.get("intent_hits", 1),
        }]
    if len(intents) > MAX_CURRENT_INTENTS:
        raise ValueError(
            f"Creature has {len(intents)} current intents; "
            f"tensor supports {MAX_CURRENT_INTENTS}"
        )
    padded_intents = [*intents, *([{}] * (MAX_CURRENT_INTENTS - len(intents)))]
    categorical_values = [
        entity_type,
        _content_id(entity),
        entity.get("zone"),
        (
            entity.get("card_type")
            or entity.get("power_type")
            or entity.get("target_type")
            or entity.get("side")
        ),
        entity.get("rarity") or entity.get("stack_type"),
        _owner_token(entity.get("owner_id")),
        f"UPGRADE:{min(16, max(0, int(_finite(entity.get('upgrade_level'), 0))))}",
        afflictions[0],
        afflictions[1],
        enchantments[0],
        enchantments[1],
        entity.get("status"),
        _count_token(count),
        *(intent.get("intent_type") for intent in padded_intents),
    ]
    categorical = _categorical_fields(
        categorical_values,
        _ENTITY_CATEGORY_FIELDS,
        {
            "phase": phase,
            "entity_type": entity_type,
            "entity_id": str(entity.get("entity_id", ""))[:160],
            "content_id": str(_content_id(entity) or "")[:160],
            "zone": str(entity.get("zone", ""))[:80],
        },
        diagnostics,
    )

    hp = _finite(entity.get("hp"))
    max_hp = max(1.0, _finite(entity.get("max_hp"), 1.0))
    counters = (
        entity.get("counters")
        if isinstance(entity.get("counters"), dict)
        else {}
    )
    numeric = np.asarray([
        hp / max_hp,
        _ratio(max_hp, 300.0),
        _ratio(entity.get("block"), 100.0),
        _ratio(entity.get("energy"), 10.0),
        _ratio(entity.get("max_energy"), 10.0),
        _ratio(entity.get("gold"), 1000.0),
        _ratio(entity.get("cost"), 10.0),
        _ratio(entity.get("original_cost"), 10.0),
        _ratio(entity.get("base_damage"), 100.0),
        _ratio(entity.get("base_block"), 100.0),
        _signed_log(entity.get("amount")),
        _signed_log(entity.get("display_amount")),
        _signed_log(count),
        _ratio(entity.get("zone_index"), 256.0),
        _ratio(entity.get("slot"), 16.0),
        _ratio(entity.get("row", entity.get("y")), 20.0),
        _ratio(entity.get("col", entity.get("x")), 10.0),
        float(bool(entity.get("alive", entity.get("is_alive", False)))),
        float(bool(entity.get("playable", False))),
        float(bool(entity.get("reachable", False))),
        float(bool(entity.get("visited", False))),
        float(bool(entity.get("upgraded", False))),
        float(bool(entity.get("can_use", False))),
        _signed_log(sum(_finite(value) for value in counters.values())),
        _signed_log(len([item for item in afflictions if item])),
        _signed_log(len([item for item in enchantments if item])),
        _signed_log(entity.get("stack_count", 1)),
        _ratio(entity.get("floor_added"), 60.0),
        float(bool(entity.get("is_used_up", False))),
        float(bool(entity.get("is_melted", False))),
        float(bool(entity.get("is_wax", False))),
        float(bool(entity.get("show_counter", False))),
        float(bool(entity.get("hidden", False))),
        float(bool(entity.get("clickable", False))),
        _ratio(
            int(entity["combat_index"]) + 1
            if entity_type == "CREATURE" and entity.get("combat_index") is not None
            else 0,
            MAX_ENEMIES,
        ),
        *(
            value
            for intent in padded_intents
            for value in (
                _ratio(intent.get("damage"), 100.0),
                _ratio(intent.get("hits"), 10.0),
            )
        ),
        _ratio(len(intents), MAX_CURRENT_INTENTS),
        _ratio(
            sum(
                _finite(intent.get("damage")) * _finite(intent.get("hits"), 1)
                for intent in intents
            ),
            300.0,
        ),
        *([] if legacy_v5 else [
            _signed_log(affliction_amount),
            _signed_log(enchantment_amount),
        ]),
    ], dtype=np.float32)
    return categorical, numeric


def _combat_candidate_slot(candidate: dict[str, Any]) -> int | None:
    payload = candidate.get("payload") or {}
    action = str(payload.get("action", "")).casefold()
    if action == "end_turn":
        return ACTION_END_TURN
    if action == "confirm_choice":
        return ACTION_END_TURN
    if action in {"play", "play_card"}:
        hand = int(payload.get("card_index", payload.get("hand_index", -1)))
        target = int(payload.get("target_index", -1))
        if hand < 0:
            return None
        if target < 0:
            return 1 + hand
        return 1 + MAX_HAND_SIZE + hand * MAX_ENEMIES + target
    if action in {"potion", "use_potion"}:
        potion_slot = int(payload.get("slot", -1))
        target = int(payload.get("target_index", -1))
        if potion_slot < 0:
            return None
        return (
            POTION_ACTION_START
            + potion_slot * POTION_TARGET_OPTIONS
            + max(0, target + 1)
        )
    return None


def _candidate_slots(
    snapshot: dict[str, Any],
    action_mask: np.ndarray,
    *,
    legacy_v5: bool = False,
) -> dict[int, dict[str, Any]]:
    if legacy_v5 and snapshot.get("reward_item_type"):
        return {}  # Preserve the exact missing-candidate v5 projection.
    entity_lookup: dict[str, dict[str, Any]] = {}
    containers = [snapshot]
    if isinstance(snapshot.get("run_state"), dict):
        containers.append(snapshot["run_state"])
    for container in containers:
        for key in (
            "players", "creatures", "enemies", "hand", "cards", "powers", "relics", "potions",
            "map_nodes", "crystal_cells", "options", "nodes", "bundles",
        ):
            for index, item in enumerate(container.get(key, []) or []):
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("entity_id") or item.get("candidate_id")
                if entity_id is None:
                    entity_id = _fallback_entity_id(key, item, index)
                entity_lookup.setdefault(str(entity_id), item)

    candidates = []
    for raw in snapshot.get("candidates", []) or []:
        if not isinstance(raw, dict) or not raw.get("enabled", True):
            continue
        item = dict(raw)
        features = dict(item.get("features") or {})
        source = entity_lookup.get(str(item.get("source_id")))
        target = entity_lookup.get(str(item.get("target_id")))
        if source is not None:
            features["model_source_content"] = _content_id(source)
            features["model_source_zone"] = source.get("zone")
        if target is not None:
            features["model_target_content"] = _content_id(target)
        item["features"] = features
        candidates.append(item)
    aligned: dict[int, dict[str, Any]] = {}
    if str(snapshot.get("type", "")).casefold() == "combat_action":
        for candidate in candidates:
            slot = _combat_candidate_slot(candidate)
            if slot is not None and 0 <= slot < TOTAL_ACTIONS:
                aligned[slot] = candidate

    phase = _snapshot_phase(snapshot)
    remaining = [
        candidate for candidate in candidates
        if candidate not in aligned.values()
    ]

    def take(action_type: str, limit: int | None = None) -> list[dict[str, Any]]:
        selected = [
            item for item in remaining
            if str(item.get("action_type", "")).upper() == action_type
        ]
        if limit is not None:
            selected = selected[:limit]
        for item in selected:
            remaining.remove(item)
        return selected

    def align_options(options: list[dict[str, Any]], slots: Iterable[int]) -> None:
        for position, (option, slot) in enumerate(zip(options, slots)):
            if not action_mask[slot]:
                continue
            option_index = int(option.get("index", position))
            match = next((
                item for item in remaining
                if (item.get("payload") or {}).get("index") == option_index
            ), None)
            if match is not None:
                aligned[slot] = match
                remaining.remove(match)

    state_type = str(snapshot.get("type", "")).casefold()
    if state_type == "reward_screen":
        options = [item for item in snapshot.get("options", []) if item.get("enabled", True)]
        picks = [item for item in options if str(item.get("action", "")).casefold() != "proceed"]
        proceeds = [item for item in options if str(item.get("action", "")).casefold() == "proceed"]
        align_options(picks[:3], range(_CARD_RWD_START, _CARD_RWD_START + 3))
        align_options(proceeds[:1], [_CARD_RWD_START + 3])
        align_options(picks[3:], range(TOTAL_ACTIONS, len(action_mask)))
    elif state_type == "shop":
        options = [item for item in snapshot.get("options", []) if item.get("enabled", True)]
        buys = [item for item in options if str(item.get("action", "")).casefold() != "leave_shop"]
        leaves = [item for item in options if str(item.get("action", "")).casefold() == "leave_shop"]
        align_options(leaves[:1], [_SHOP_START])
        align_options(buys[:9], range(_SHOP_START + 1, _SHOP_START + 10))
        align_options(buys[9:], range(TOTAL_ACTIONS, len(action_mask)))
    elif state_type == "crystal_sphere":
        options = [item for item in snapshot.get("options", []) if item.get("enabled", True)]
        minigame = snapshot.get("minigame")
        if isinstance(minigame, dict) and (
            minigame.get("finished") or minigame.get("divinations_remaining") == 0
        ):
            proceeds = [item for item in options if str(item.get("action", "")).casefold() == "proceed"]
            options = proceeds or options
        slots = [*range(_EVENT_START, _EVENT_START + 4), *range(TOTAL_ACTIONS, len(action_mask))]
        align_options(options, slots)
    elif state_type == "card_select":
        cards = [item for item in snapshot.get("cards", []) if item.get("enabled", True)]
        choice_slots = [*range(1, _COMBAT_SIZE), *range(TOTAL_ACTIONS, len(action_mask))]
        align_options(cards, choice_slots)
        if action_mask[0]:
            command_type = "CONFIRM" if snapshot.get("run_state_available") else "SKIP"
            matches = take(command_type, 1)
            if matches:
                aligned[0] = matches[0]
    elif phase == "CARD_REWARD":
        if snapshot.get("reward_item_type"):
            picks = take("CHOOSE", 1)
            skips = take("SKIP", 1)
            if picks:
                aligned[_CARD_RWD_START] = picks[0]
            if skips:
                aligned[_CARD_RWD_START + 3] = skips[0]
            return aligned
        pick_slots = [
            _CARD_RWD_START,
            _CARD_RWD_START + 1,
            _CARD_RWD_START + 2,
            _CARD_RWD_EXTRA_START,
            _CARD_RWD_EXTRA_START + 1,
            _CARD_RWD_EXTRA_START + 2,
        ]
        picks = take("PICK_CARD", len(pick_slots))
        if not picks:
            picks = take("PICK_CARD_BUNDLE", len(pick_slots))
        aligned.update(zip(pick_slots, picks))
        skips = take("SKIP")
        if skips:
            aligned[_CARD_RWD_START + 3] = skips[0]
        rerolls = take("REROLL")
        if rerolls:
            aligned[_CARD_RWD_REROLL] = rerolls[0]
    elif phase == "BOSS_RELIC":
        aligned.update(
            (_BOSS_RELIC_START + index, item)
            for index, item in enumerate(take("PICK_RELIC", 3))
        )

    valid_unassigned = [
        int(slot) for slot in np.flatnonzero(action_mask)
        if int(slot) not in aligned
    ]
    for slot, candidate in zip(valid_unassigned, remaining):
        aligned[slot] = candidate
    return aligned


def _action_family(slot: int) -> str:
    if slot >= TOTAL_ACTIONS:
        return "CHOOSE"
    if slot < _COMBAT_SIZE:
        if slot == ACTION_END_TURN:
            return "END_TURN_OR_CONFIRM"
        if slot >= POTION_ACTION_START:
            return "USE_POTION"
        return "PLAY_OR_CHOOSE"
    if slot < _MAP_START + 5:
        return "MOVE"
    if slot < _CARD_RWD_START + 4:
        return "CARD_REWARD"
    if slot < _BOSS_RELIC_START:
        return "CARD_REWARD_EXTRA"
    if slot < _SHOP_START:
        return "BOSS_RELIC"
    if slot < _REST_START:
        return "SHOP"
    if slot < _EVENT_START:
        return "REST"
    if slot < _TREASURE_START:
        return "EVENT"
    if slot == _TREASURE_START:
        return "TREASURE_OR_REROLL"
    if slot >= _PLAYER_SELECT_START:
        return "SELECT_PLAYER"
    return "CHOOSE"


def _candidate_row(
    slot: int,
    candidate: dict[str, Any] | None,
    valid: bool,
    phase: Any,
    config: TensorizerConfig,
    diagnostics: UnknownTokenDiagnostics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    item = candidate or {}
    features = (
        item.get("features")
        if isinstance(item.get("features"), dict)
        else {}
    )
    payload = (
        item.get("payload")
        if isinstance(item.get("payload"), dict)
        else {}
    )
    semantic_option = features.get("id") or features.get("option_id")
    if semantic_option is None and any(
        features.get(key) for key in ("cards", "options", "relics")
    ):
        semantic_option = "COMPOSITE"
    categorical = _categorical_fields(
        [
            item.get("action_type") or _action_family(slot),
            f"ACTION_SLOT:{slot + 1}",
            features.get("model_source_content"),
            features.get("model_target_content"),
            phase,
            semantic_option,
            features.get("model_source_zone") or payload.get("action"),
        ],
        _CANDIDATE_CATEGORY_FIELDS,
        {
            "phase": phase,
            "slot": slot,
            "valid": valid,
            "candidate_id": str(item.get("candidate_id", ""))[:160],
            "source_id": str(item.get("source_id", ""))[:160],
            "target_id": str(item.get("target_id", ""))[:160],
        },
        diagnostics,
    )
    numeric = np.asarray([
        float(valid),
        _ratio(features.get("price"), 1000.0),
        _ratio(features.get("cost"), 10.0),
        _ratio(features.get("x"), 11.0),
        _ratio(features.get("y"), 11.0),
        _ratio(payload.get("index"), 256.0),
        _ratio(
            payload.get("card_index", payload.get("hand_index")),
            MAX_HAND_SIZE,
        ),
        _ratio(payload.get("target_index"), MAX_ENEMIES),
        _ratio(payload.get("slot"), 16.0),
        _signed_log(len(features.get("affected_cell_ids", []) or [])),
        float(bool(features.get("selected", False))),
        _ratio(features.get("selected_count"), 16.0),
        _ratio(features.get("min_select"), 16.0),
        _ratio(features.get("max_select"), 16.0),
        float(bool(features.get("can_confirm", False))),
    ], dtype=np.float32)
    return categorical, numeric


@lru_cache(maxsize=16)
def observation_space(
    config: TensorizerConfig = DEFAULT_TENSORIZER_CONFIG,
) -> spaces.Dict:
    categorical_high = config.categorical_vocab_size - 1
    return spaces.Dict({
        "global_categorical": spaces.Box(
            0,
            categorical_high,
            shape=(config.global_categorical_fields,),
            dtype=np.int32,
        ),
        "global_numeric": spaces.Box(
            -10.0,
            10.0,
            shape=(config.global_numeric_fields,),
            dtype=np.float32,
        ),
        "entity_categorical": spaces.Box(
            0,
            categorical_high,
            shape=(
                config.max_entities,
                config.entity_categorical_fields,
            ),
            dtype=np.int32,
        ),
        "entity_numeric": spaces.Box(
            -10.0,
            10.0,
            shape=(config.max_entities, config.entity_numeric_fields),
            dtype=np.float32,
        ),
        "entity_mask": spaces.MultiBinary(config.max_entities),
        "candidate_categorical": spaces.Box(
            0,
            categorical_high,
            shape=(
                config.num_actions,
                config.candidate_categorical_fields,
            ),
            dtype=np.int32,
        ),
        "candidate_numeric": spaces.Box(
            -10.0,
            10.0,
            shape=(
                config.num_actions,
                config.candidate_numeric_fields,
            ),
            dtype=np.float32,
        ),
        "candidate_source_row": spaces.Box(
            -1, config.max_entities - 1,
            shape=(config.num_actions,), dtype=np.int32,
        ),
        "candidate_target_row": spaces.Box(
            -1, config.max_entities - 1,
            shape=(config.num_actions,), dtype=np.int32,
        ),
    })


def tensorize_snapshot(
    snapshot: dict[str, Any],
    action_mask: np.ndarray,
    config: TensorizerConfig = DEFAULT_TENSORIZER_CONFIG,
    *,
    validate: bool = False,
    unknown_diagnostics: UnknownTokenDiagnostics | None = None,
) -> dict[str, np.ndarray]:
    """Convert a v2 snapshot and fixed action mask to padded numpy tensors.

    Full Gymnasium-space validation scans every padded array, so it is opt-in
    for tests and diagnostics rather than part of the production hot path.
    """
    if action_mask.shape != (config.num_actions,):
        raise ValueError(
            f"Expected action mask {(config.num_actions,)}, got {action_mask.shape}"
        )
    if isinstance(config, LegacyV5TensorizerConfig) and "legacy_event_options" in snapshot:
        snapshot = {
            **snapshot,
            "options": snapshot["legacy_event_options"],
            "candidates": snapshot["legacy_event_candidates"],
        }
    phase = _snapshot_phase(snapshot)
    run_state = snapshot.get("run_state")
    run = run_state if isinstance(run_state, dict) else snapshot
    players = run.get("players", []) or snapshot.get("players", []) or []
    player = players[0] if players else snapshot.get("player", {}) or {}
    deck_size = sum(
        int(item.get("count", 1))
        for item in run.get("cards", []) or []
        if str(item.get("zone", "")).casefold() == "deck"
    )

    global_state = snapshot.get("global")
    global_state = global_state if isinstance(global_state, dict) else {}
    global_categorical = _categorical_fields(
        [
            phase,
            run.get("character_id") or player.get("character_id"),
            snapshot.get("room_type") or global_state.get("room_type"),
            snapshot.get("type"),
        ],
        _GLOBAL_CATEGORY_FIELDS,
        {"phase": phase, "screen_type": str(snapshot.get("type", ""))},
        unknown_diagnostics,
    )
    hp = _finite(player.get("hp"))
    max_hp = max(1.0, _finite(player.get("max_hp"), 1.0))
    crystal = snapshot.get("crystal_minigame")
    crystal = crystal if isinstance(crystal, dict) else {}
    global_numeric = np.asarray([
        _ratio(run.get("act"), 4.0),
        _ratio(run.get("act_floor"), 20.0),
        _ratio(run.get("floor"), 60.0),
        _ratio(run.get("ascension"), 20.0),
        hp / max_hp,
        _ratio(max_hp, 100.0),
        _ratio(player.get("gold"), 1000.0),
        _signed_log(deck_size),
        _ratio(len(run.get("relics", []) or []), 30.0),
        _ratio(len(run.get("potions", []) or []), 10.0),
        _ratio(player.get("max_potion_slots"), 10.0),
        _ratio(snapshot.get("round"), 100.0),
        _ratio(crystal.get("divinations_remaining"), 6.0),
        float(bool(run.get("is_over", False))),
        float(bool(run.get("player_won", False))),
        float(bool(
            snapshot.get("run_state_available", isinstance(run_state, dict))
        )),
    ], dtype=np.float32)

    entity_categorical = np.zeros(
        (config.max_entities, config.entity_categorical_fields),
        dtype=np.int32,
    )
    entity_numeric = np.zeros(
        (config.max_entities, config.entity_numeric_fields),
        dtype=np.float32,
    )
    entity_mask = np.zeros(config.max_entities, dtype=np.int8)
    entities = _iter_snapshot_entities(
        snapshot, legacy_v5=isinstance(config, LegacyV5TensorizerConfig),
    )
    overflow = max(0, len(entities) - config.max_entities)
    if overflow:
        entities = entities[: config.max_entities - 1]
        entities.append(
            ("OVERFLOW", {"id": "OVERFLOW", "count": overflow + 1})
        )
    for index, (entity_type, entity) in enumerate(entities):
        categorical, numeric = _entity_row(
            entity_type, entity, config, unknown_diagnostics, phase,
        )
        entity_categorical[index] = categorical
        entity_numeric[index] = numeric
        entity_mask[index] = 1
    entity_rows: dict[str, int] = {}
    for index, (_, entity) in enumerate(entities):
        if entity.get("entity_id") is not None:
            entity_rows.setdefault(str(entity["entity_id"]), index)

    candidate_categorical = np.zeros(
        (config.num_actions, config.candidate_categorical_fields),
        dtype=np.int32,
    )
    candidate_numeric = np.zeros(
        (config.num_actions, config.candidate_numeric_fields),
        dtype=np.float32,
    )
    candidate_source_row = np.full(config.num_actions, -1, dtype=np.int32)
    candidate_target_row = np.full(config.num_actions, -1, dtype=np.int32)
    aligned = _candidate_slots(
        snapshot, action_mask,
        legacy_v5=isinstance(config, LegacyV5TensorizerConfig),
    )
    for slot in range(config.num_actions):
        categorical, numeric = _candidate_row(
            slot,
            aligned.get(slot),
            bool(action_mask[slot]),
            phase,
            config,
            unknown_diagnostics,
        )
        candidate_categorical[slot] = categorical
        candidate_numeric[slot] = numeric
        candidate = aligned.get(slot)
        if candidate is not None and action_mask[slot]:
            for field, rows in (
                ("source_id", candidate_source_row),
                ("target_id", candidate_target_row),
            ):
                entity_id = candidate.get(field)
                if entity_id is not None:
                    rows[slot] = entity_rows.get(str(entity_id), -1)

    result = {
        "global_categorical": global_categorical,
        "global_numeric": global_numeric,
        "entity_categorical": entity_categorical,
        "entity_numeric": entity_numeric,
        "entity_mask": entity_mask,
        "candidate_categorical": candidate_categorical,
        "candidate_numeric": candidate_numeric,
        "candidate_source_row": candidate_source_row,
        "candidate_target_row": candidate_target_row,
    }
    if validate and not observation_space(config).contains(result):
        raise ValueError(
            "Tensorized v2 observation violates its Gymnasium space"
        )
    if unknown_diagnostics is not None:
        unknown_diagnostics.note_observation()
    return result
