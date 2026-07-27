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

from sts2_env.core.constants import (
    ACTION_END_TURN,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    POTION_ACTION_START,
    POTION_TARGET_OPTIONS,
)
from sts2_env.gym_env.run_env import (
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

TENSOR_ENCODING_VERSION = "typed-set-tensor-v2"


@dataclass(frozen=True, slots=True)
class TensorizerConfig:
    """Shape and vocabulary contract stored with a v2 checkpoint."""

    max_entities: int = 384
    num_actions: int = TOTAL_ACTIONS
    categorical_buckets: int = 4096
    entity_categorical_fields: int = 10
    entity_numeric_fields: int = 24
    candidate_categorical_fields: int = 7
    candidate_numeric_fields: int = 15
    global_categorical_fields: int = 4
    global_numeric_fields: int = 16

    def feature_layout_hash(self) -> str:
        payload = json.dumps(
            {
                "encoding_version": TENSOR_ENCODING_VERSION,
                "config": asdict(self),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


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
ENTITY_TYPE_TO_ID = {name: index for index, name in enumerate(ENTITY_TYPES)}


def _stable_bucket(value: Any, buckets: int) -> int:
    """Return a stable non-zero categorical bucket."""
    if value is None or value == "":
        return 0
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        value = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        )
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "little") % (buckets - 1)


def _enum_bucket(value: Any, buckets: int) -> int:
    return _stable_bucket(
        str(value).upper() if value is not None else None, buckets
    )


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _ratio(value: Any, scale: float) -> float:
    return float(np.clip(_finite(value) / scale, -10.0, 10.0))


def _signed_log(value: Any, scale: float = 5.0) -> float:
    number = _finite(value)
    return float(np.clip(
        math.copysign(math.log1p(abs(number)) / scale, number), -10.0, 10.0
    ))


def _content_id(entity: dict[str, Any]) -> Any:
    for key in (
        "card_id", "monster_id", "power_id", "relic_id", "potion_id",
        "node_type", "revealed_item_id", "character_id", "id", "type",
    ):
        if entity.get(key) not in (None, ""):
            return entity[key]
    return None


def _owner_bucket(owner_id: Any, buckets: int) -> int:
    if owner_id is None:
        return 0
    text = str(owner_id)
    if text.startswith("player:"):
        return 1 + int(text.rsplit(":", 1)[-1]) % 16
    if text.startswith("enemy:"):
        return 32 + _stable_bucket(text.rsplit(":", 1)[-1], 32)
    return _stable_bucket(text, buckets)


def _count_bucket(count: int, buckets: int) -> int:
    if count <= 1:
        return 1
    if count <= 16:
        return count
    return min(buckets - 1, 16 + int(math.log2(count)))


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
) -> list[tuple[str, dict[str, Any]]]:
    """Collect current-screen entities first, then persistent run entities."""
    collected: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()

    def extend(container: dict[str, Any], key: str, entity_type: str) -> None:
        for index, raw in enumerate(container.get(key, []) or []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            identity = str(
                item.get("entity_id") or f"{key}:{index}:{_content_id(item)}"
            )
            marker = (entity_type, identity)
            if marker in seen:
                continue
            seen.add(marker)
            collected.append((entity_type, item))

    current_specs = (
        ("players", "PLAYER"),
        ("creatures", "CREATURE"),
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
        for index, raw in enumerate(snapshot.get(key, []) or []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item.setdefault("zone", "candidate")
            item.setdefault("zone_index", index)
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


def _entity_row(
    entity_type: str,
    entity: dict[str, Any],
    config: TensorizerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = config.categorical_buckets
    count = max(1, int(_finite(entity.get("count"), 1)))
    categorical = np.asarray([
        ENTITY_TYPE_TO_ID.get(entity_type, ENTITY_TYPE_TO_ID["CHOICE"]),
        _stable_bucket(_content_id(entity), buckets),
        _enum_bucket(entity.get("zone"), buckets),
        _enum_bucket(
            entity.get("card_type")
            or entity.get("power_type")
            or entity.get("target_type")
            or entity.get("side"),
            buckets,
        ),
        _enum_bucket(entity.get("rarity") or entity.get("stack_type"), buckets),
        _owner_bucket(entity.get("owner_id"), buckets),
        min(
            buckets - 1,
            max(0, int(_finite(entity.get("upgrade_level"), 0))) + 1,
        ),
        _stable_bucket(entity.get("afflictions") or (), buckets),
        _stable_bucket(entity.get("enchantments") or (), buckets),
        _count_bucket(count, buckets),
    ], dtype=np.int32)

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
        _ratio(entity.get("row"), 20.0),
        _ratio(entity.get("col"), 10.0),
        float(bool(entity.get("alive", entity.get("is_alive", False)))),
        float(bool(entity.get("playable", False))),
        float(bool(entity.get("reachable", False))),
        float(bool(entity.get("visited", False))),
        float(bool(entity.get("upgraded", False))),
        float(bool(entity.get("can_use", False))),
        _signed_log(sum(_finite(value) for value in counters.values())),
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
) -> dict[int, dict[str, Any]]:
    entity_lookup: dict[str, dict[str, Any]] = {}
    containers = [snapshot]
    if isinstance(snapshot.get("run_state"), dict):
        containers.append(snapshot["run_state"])
    for container in containers:
        for key in (
            "players", "creatures", "cards", "powers", "relics", "potions",
            "map_nodes", "crystal_cells", "options", "nodes", "bundles",
        ):
            for item in container.get(key, []) or []:
                if isinstance(item, dict) and item.get("entity_id"):
                    entity_lookup[str(item["entity_id"])] = item

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

    phase = str(
        snapshot.get("phase")
        or snapshot.get("global", {}).get("phase")
        or ""
    ).upper()
    remaining = [
        candidate for candidate in candidates
        if candidate not in aligned.values()
    ]

    def take(action_type: str) -> list[dict[str, Any]]:
        selected = [
            item for item in remaining
            if str(item.get("action_type", "")).upper() == action_type
        ]
        for item in selected:
            remaining.remove(item)
        return selected

    if phase == "CARD_REWARD":
        pick_slots = [
            _CARD_RWD_START,
            _CARD_RWD_START + 1,
            _CARD_RWD_START + 2,
            _CARD_RWD_EXTRA_START,
            _CARD_RWD_EXTRA_START + 1,
            _CARD_RWD_EXTRA_START + 2,
        ]
        aligned.update(zip(pick_slots, take("PICK_CARD")))
        skips = take("SKIP")
        if skips:
            aligned[_CARD_RWD_START + 3] = skips[0]
        rerolls = take("REROLL")
        if rerolls:
            aligned[_CARD_RWD_REROLL] = rerolls[0]
    elif phase == "BOSS_RELIC":
        aligned.update(
            (_BOSS_RELIC_START + index, item)
            for index, item in enumerate(take("PICK_RELIC")[:3])
        )

    valid_unassigned = [
        int(slot) for slot in np.flatnonzero(action_mask)
        if int(slot) not in aligned
    ]
    for slot, candidate in zip(valid_unassigned, remaining):
        aligned[slot] = candidate
    return aligned


def _action_family(slot: int) -> str:
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
    return "UNKNOWN"


def _candidate_row(
    slot: int,
    candidate: dict[str, Any] | None,
    valid: bool,
    phase: Any,
    config: TensorizerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = config.categorical_buckets
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
    categorical = np.asarray([
        _stable_bucket(item.get("action_type") or _action_family(slot), buckets),
        slot + 1,
        _stable_bucket(features.get("model_source_content"), buckets),
        _stable_bucket(features.get("model_target_content"), buckets),
        _stable_bucket(phase, buckets),
        _stable_bucket(
            features.get("cards")
            or features.get("options")
            or features.get("relics")
            or features.get("id"),
            buckets,
        ),
        _stable_bucket(
            features.get("model_source_zone") or payload.get("action"),
            buckets,
        ),
    ], dtype=np.int32)
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
    categorical_high = config.categorical_buckets - 1
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
    })


def tensorize_snapshot(
    snapshot: dict[str, Any],
    action_mask: np.ndarray,
    config: TensorizerConfig = DEFAULT_TENSORIZER_CONFIG,
) -> dict[str, np.ndarray]:
    """Convert a v2 snapshot and fixed action mask to padded numpy tensors."""
    if action_mask.shape != (config.num_actions,):
        raise ValueError(
            f"Expected action mask {(config.num_actions,)}, got {action_mask.shape}"
        )
    phase = snapshot.get("phase") or snapshot.get("global", {}).get("phase")
    run_state = snapshot.get("run_state")
    run = run_state if isinstance(run_state, dict) else snapshot
    players = run.get("players", []) or snapshot.get("players", []) or []
    player = players[0] if players else snapshot.get("player", {}) or {}
    deck_size = sum(
        int(item.get("count", 1))
        for item in run.get("cards", []) or []
        if str(item.get("zone", "")).casefold() == "deck"
    )

    global_categorical = np.asarray([
        _stable_bucket(phase, config.categorical_buckets),
        _stable_bucket(
            run.get("character_id") or player.get("character_id"),
            config.categorical_buckets,
        ),
        _stable_bucket(
            snapshot.get("room_type")
            or snapshot.get("global", {}).get("room_type"),
            config.categorical_buckets,
        ),
        _stable_bucket(snapshot.get("type"), config.categorical_buckets),
    ], dtype=np.int32)
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
    entities = _iter_snapshot_entities(snapshot)
    overflow = max(0, len(entities) - config.max_entities)
    if overflow:
        entities = entities[: config.max_entities - 1]
        entities.append(
            ("OVERFLOW", {"id": "OVERFLOW", "count": overflow + 1})
        )
    for index, (entity_type, entity) in enumerate(entities):
        categorical, numeric = _entity_row(entity_type, entity, config)
        entity_categorical[index] = categorical
        entity_numeric[index] = numeric
        entity_mask[index] = 1

    candidate_categorical = np.zeros(
        (config.num_actions, config.candidate_categorical_fields),
        dtype=np.int32,
    )
    candidate_numeric = np.zeros(
        (config.num_actions, config.candidate_numeric_fields),
        dtype=np.float32,
    )
    aligned = _candidate_slots(snapshot, action_mask)
    for slot in range(config.num_actions):
        categorical, numeric = _candidate_row(
            slot,
            aligned.get(slot),
            bool(action_mask[slot]),
            phase,
            config,
        )
        candidate_categorical[slot] = categorical
        candidate_numeric[slot] = numeric

    result = {
        "global_categorical": global_categorical,
        "global_numeric": global_numeric,
        "entity_categorical": entity_categorical,
        "entity_numeric": entity_numeric,
        "entity_mask": entity_mask,
        "candidate_categorical": candidate_categorical,
        "candidate_numeric": candidate_numeric,
    }
    if not observation_space(config).contains(result):
        raise ValueError(
            "Tensorized v2 observation violates its Gymnasium space"
        )
    return result
