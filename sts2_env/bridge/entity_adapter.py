"""Live/simulator adapter for the versioned entity-agent contract.

This layer intentionally returns structured Python data.  Padding and tensor
projection belong to the future model implementation and can evolve without
changing the wire/replay schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts2_env.agent_v2.candidates import (
    ActionCandidate,
    build_action_candidates,
    resolve_candidate,
)
from sts2_env.agent_v2.schema import validate_v2_envelope


@dataclass(frozen=True, slots=True)
class EntityDecision:
    episode_id: str | None
    decision_id: str | int | None
    phase: str
    global_state: dict[str, Any]
    players: tuple[dict[str, Any], ...]
    cards: tuple[dict[str, Any], ...]
    creatures: tuple[dict[str, Any], ...]
    powers: tuple[dict[str, Any], ...]
    relics: tuple[dict[str, Any], ...]
    potions: tuple[dict[str, Any], ...]
    map_nodes: tuple[dict[str, Any], ...]
    map_edges: tuple[dict[str, Any], ...]
    crystal_cells: tuple[dict[str, Any], ...]
    minigame: dict[str, Any]
    candidates: tuple[ActionCandidate, ...]


class EntityStateAdapter:
    """Validate v2 messages and merge their complete run/combat entities."""

    def decode(self, state: dict[str, Any]) -> EntityDecision:
        validate_v2_envelope(state)
        run = state.get("run_state")
        run = run if isinstance(run, dict) else {}
        combat_cards = list(state.get("cards", []))
        if not combat_cards:
            # Live combat messages keep v1 hand at the top level.
            combat_cards = list(state.get("hand", []))
        cards = self._merge_entities(
            list(run.get("cards", run.get("deck", []))),
            combat_cards,
        )
        players = self._merge_entities(
            list(run.get("players", [])),
            list(state.get("players", [])),
        )
        if not players and isinstance(state.get("player"), dict):
            player = dict(state["player"])
            player.setdefault("entity_id", "player:local")
            players = [player]
        creatures = list(state.get("creatures", state.get("enemies", [])))
        powers = list(state.get("powers", []))
        if not powers:
            powers = self._flatten_legacy_powers(players, creatures)
        global_state = {
            "phase": state.get("phase") or str(state.get("type", "")).upper(),
            "character_id": run.get("character_id"),
            "act": state.get("act", run.get("act")),
            "act_floor": state.get("act_floor", run.get("act_floor")),
            "floor": state.get("floor", run.get("floor")),
            "ascension": state.get("ascension", run.get("ascension")),
            "room_type": run.get("room_type"),
            "run_state_available": state.get("run_state_available", bool(run)),
        }
        return EntityDecision(
            episode_id=state.get("episode_id"),
            decision_id=state.get("decision_id"),
            phase=str(global_state["phase"]),
            global_state=global_state,
            players=tuple(players),
            cards=tuple(cards),
            creatures=tuple(creatures),
            powers=tuple(powers),
            relics=tuple(run.get("relics", state.get("relics", []))),
            potions=tuple(self._merge_entities(
                list(run.get("potions", [])),
                list(state.get("potions", [])),
            )),
            map_nodes=tuple(run.get("map_nodes", state.get("map_nodes", []))),
            map_edges=tuple(run.get("map_edges", state.get("map_edges", []))),
            crystal_cells=tuple(state.get("crystal_cells", [])),
            minigame=dict(state.get("minigame", {})),
            candidates=tuple(build_action_candidates(state)),
        )

    def resolve(self, state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
        validate_v2_envelope(state)
        payload = resolve_candidate(state, candidate_id)
        payload["candidate_id"] = candidate_id
        if state.get("decision_id") is not None:
            payload["decision_id"] = state["decision_id"]
        return payload

    @staticmethod
    def _merge_entities(
        base: list[dict[str, Any]],
        overlay: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        anonymous: list[dict[str, Any]] = []
        for item in [*base, *overlay]:
            entity_id = item.get("entity_id")
            if entity_id is None:
                anonymous.append(dict(item))
                continue
            merged[str(entity_id)] = {**merged.get(str(entity_id), {}), **item}
        return [*merged.values(), *anonymous]

    @staticmethod
    def _flatten_legacy_powers(
        players: list[dict[str, Any]],
        creatures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        powers: list[dict[str, Any]] = []
        for owner_index, owner in enumerate([*players, *creatures]):
            owner_id = str(owner.get("entity_id", f"owner:{owner_index}"))
            for power in owner.get("powers", []):
                power_id = str(power.get("power_id", power.get("id", "UNKNOWN")))
                powers.append({
                    "entity_id": f"power:{owner_id}:{power_id}",
                    "owner_id": owner_id,
                    "power_id": power_id,
                    "amount": int(power.get("amount", 0)),
                })
        return powers
