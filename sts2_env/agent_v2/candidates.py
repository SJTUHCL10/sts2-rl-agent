"""Semantic, state-local action candidates.

Candidate ordering is deterministic but is not part of action semantics.
Callers should return ``candidate_id`` together with ``decision_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    candidate_id: str
    action_type: str
    source_id: str | None = None
    target_id: str | None = None
    enabled: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "enabled": self.enabled,
            "payload": dict(self.payload),
            "features": dict(self.features),
        }


def _entity_id(item: dict[str, Any], prefix: str, index: int) -> str:
    return str(
        item.get("entity_id")
        or item.get("candidate_id")
        or f"{prefix}:{item.get('id', 'unknown')}:{item.get('index', index)}"
    )


def _canonical_target(value: Any) -> str:
    return str(value or "").replace("_", "").replace(" ", "").casefold()


def _combat_candidates(state: dict[str, Any]) -> list[ActionCandidate]:
    candidates = [
        ActionCandidate(
            candidate_id="combat:end_turn",
            action_type="END_TURN",
            payload={"action": "end_turn"},
        )
    ]
    enemies = list(state.get("enemies", []))
    alive = [
        (index, enemy, _entity_id(enemy, "enemy", index))
        for index, enemy in enumerate(enemies)
        if bool(enemy.get("is_alive", enemy.get("alive", False)))
    ]
    for hand_index, card in enumerate(state.get("hand", [])):
        if not bool(card.get("playable", True)):
            continue
        card_entity = _entity_id(card, "hand", hand_index)
        target_type = _canonical_target(card.get("target") or card.get("target_type"))
        base_payload = {"action": "play", "card_index": hand_index}
        if target_type in {"anyenemy", "randomenemy"}:
            for target_index, _, target_entity in alive:
                candidates.append(ActionCandidate(
                    candidate_id=f"combat:play:{card_entity}:{target_entity}",
                    action_type="PLAY_CARD",
                    source_id=card_entity,
                    target_id=target_entity,
                    payload={**base_payload, "target_index": target_index},
                    features={"cost": card.get("cost", 0)},
                ))
        else:
            candidates.append(ActionCandidate(
                candidate_id=f"combat:play:{card_entity}:none",
                action_type="PLAY_CARD",
                source_id=card_entity,
                payload={**base_payload, "target_index": -1},
                features={"cost": card.get("cost", 0)},
            ))

    available = {
        str(value).upper()
        for value in state.get("available_actions", [])
    }
    if not available or "POTION" in available:
        for list_index, potion in enumerate(state.get("potions", [])):
            if not potion or not bool(potion.get("can_use", True)):
                continue
            slot = int(potion.get("slot", list_index))
            potion_entity = _entity_id(potion, "potion", slot)
            targeted = bool(potion.get("requires_target", False)) or _canonical_target(
                potion.get("target") or potion.get("target_type")
            ) == "anyenemy"
            if targeted:
                for target_index, _, target_entity in alive:
                    candidates.append(ActionCandidate(
                        candidate_id=f"combat:potion:{potion_entity}:{target_entity}",
                        action_type="USE_POTION",
                        source_id=potion_entity,
                        target_id=target_entity,
                        payload={
                            "action": "potion",
                            "slot": slot,
                            "target_index": target_index,
                        },
                    ))
            else:
                candidates.append(ActionCandidate(
                    candidate_id=f"combat:potion:{potion_entity}:none",
                    action_type="USE_POTION",
                    source_id=potion_entity,
                    payload={"action": "potion", "slot": slot, "target_index": -1},
                ))
    return candidates


def _choice_items(state: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for key in ("nodes", "options", "cards", "bundles", "relics"):
        if key in state:
            return key, list(state.get(key) or [])
    return "options", []


def _choice_candidates(state: dict[str, Any]) -> list[ActionCandidate]:
    state_type = str(state.get("type", "choice"))
    key, items = _choice_items(state)
    candidates: list[ActionCandidate] = []
    selection_features = {
        "selected_count": state.get("selected_count", 0),
        "min_select": state.get("min_select"),
        "max_select": state.get("max_select"),
        "can_confirm": state.get("can_confirm", False),
    }
    if bool(state.get("can_confirm", False)):
        candidates.append(ActionCandidate(
            candidate_id=f"{state_type}:confirm",
            action_type="CONFIRM",
            payload={"action": "confirm_choice"},
            features=selection_features,
        ))
    for position, item in enumerate(items):
        if not bool(item.get("enabled", True)):
            continue
        external_index = int(item.get("index", position))
        source_id = _entity_id(item, key.rstrip("s"), position)
        action_type = str(item.get("action") or {
            "map_select": "MOVE",
            "card_reward": "PICK_CARD",
            "card_select": "SELECT_CARD",
            "card_bundle": "PICK_CARD_BUNDLE",
            "boss_relic": "PICK_RELIC",
        }.get(state_type, "CHOOSE")).upper()
        if state_type == "crystal_sphere":
            x = item.get("x")
            y = item.get("y")
            if x is not None and y is not None:
                source_id = str(
                    item.get("entity_id") or f"crystal-cell:{int(x)}:{int(y)}"
                )
                candidate_id = f"crystal_sphere:divine:{int(x)}:{int(y)}"
            else:
                candidate_id = "crystal_sphere:proceed"
        else:
            candidate_id = f"{state_type}:{action_type.casefold()}:{source_id}"
        candidates.append(ActionCandidate(
            candidate_id=candidate_id,
            action_type=action_type,
            source_id=source_id,
            payload={"action": "choose", "index": external_index},
            features={
                **selection_features,
                key: item.get("id") or item.get("type"),
                "price": item.get("price"),
                "selected": item.get("selected", False),
                "x": item.get("x"),
                "y": item.get("y"),
                "affected_cell_ids": item.get("affected_cell_ids", []),
            },
        ))
    if bool(state.get("can_skip")) or (
        state_type == "card_select"
        and int(state.get("min_select", 1) or 0) == 0
    ):
        candidates.append(ActionCandidate(
            candidate_id=f"{state_type}:skip",
            action_type="SKIP",
            payload={"action": "skip"},
        ))
    return candidates


def build_action_candidates(state: dict[str, Any]) -> list[ActionCandidate]:
    """Build deterministic semantic candidates from a simulator/live snapshot."""
    supplied = state.get("candidates")
    if isinstance(supplied, list) and supplied:
        return [
            ActionCandidate(
                candidate_id=str(item["candidate_id"]),
                action_type=str(item["action_type"]),
                source_id=item.get("source_id"),
                target_id=item.get("target_id"),
                enabled=bool(item.get("enabled", True)),
                payload=dict(item.get("payload", {})),
                features=dict(item.get("features", {})),
            )
            for item in supplied
        ]
    if state.get("type") == "combat_action":
        return _combat_candidates(state)
    return _choice_candidates(state)


def resolve_candidate(
    state: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """Resolve a semantic ID against the exact state that produced it."""
    matches = [
        candidate
        for candidate in build_action_candidates(state)
        if candidate.candidate_id == candidate_id and candidate.enabled
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one enabled candidate {candidate_id!r}, found {len(matches)}"
        )
    return dict(matches[0].payload)
