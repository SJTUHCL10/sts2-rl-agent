"""Canonical full snapshots shared by simulator and live adapters."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from sts2_env.agent_v2.candidates import build_action_candidates
from sts2_env.agent_v2.categorical_vocabulary import DEFAULT_CATEGORICAL_VOCABULARY
from sts2_env.agent_v2.schema import schema_manifest
from sts2_env.agent_v2.state_fields import (
    serialize_power_fields,
    serialize_relic_fields,
)

if TYPE_CHECKING:
    from sts2_env.map.generator import ActMap
    from sts2_env.cards.base import CardInstance
    from sts2_env.core.combat import CombatState
    from sts2_env.run.run_state import PlayerState, RunState
    from sts2_env.run.run_manager import RunManager


_MapNodeTemplate = tuple[int, int, str]
_MapEdgeTemplate = tuple[int, int, int, int]
_MAP_TOPOLOGY_CACHE: WeakKeyDictionary[
    Any,
    tuple[tuple[_MapNodeTemplate, ...], tuple[_MapEdgeTemplate, ...]],
] = WeakKeyDictionary()


def _map_topology(
    act_map: ActMap,
) -> tuple[tuple[_MapNodeTemplate, ...], tuple[_MapEdgeTemplate, ...]]:
    """Return immutable topology data cached for the lifetime of an act map."""
    cached = _MAP_TOPOLOGY_CACHE.get(act_map)
    if cached is not None:
        return cached
    points = sorted(
        act_map.all_points(),
        key=lambda point: (point.row, point.col),
    )
    nodes = tuple(
        (point.col, point.row, point.point_type.name)
        for point in points
    )
    edges = tuple(
        (point.col, point.row, child.col, child.row)
        for point in points
        for child in point.children
    )
    result = (nodes, edges)
    _MAP_TOPOLOGY_CACHE[act_map] = result
    return result


def _name(value: Any) -> str:
    return value.name if isinstance(value, Enum) else str(value)


def serialize_card(
    card: CardInstance,
    *,
    zone: str,
    zone_index: int,
    owner_id: str,
    playable: bool | None = None,
) -> dict[str, Any]:
    """Serialize all currently modeled card-instance state."""
    affliction = card.affliction
    return {
        "entity_id": f"card:{owner_id}:{card.instance_id}",
        "card_id": card.card_id.name,
        "id": card.card_id.name,  # v1/live interoperability
        "owner_id": owner_id,
        "zone": zone,
        "zone_index": zone_index,
        "cost": card.cost,
        "original_cost": card.original_cost,
        "card_type": card.card_type.name,
        "type": card.card_type.name.title(),
        "target_type": card.target_type.name,
        "target": card.target_type.name.title().replace("_", ""),
        "rarity": card.rarity.name,
        "base_damage": card.base_damage,
        "base_block": card.base_block,
        "upgrade_level": 1 if card.upgraded else 0,
        "upgraded": card.upgraded,
        "playable": playable,
        "keywords": sorted(_name(value) for value in card.keywords),
        "tags": sorted(_name(value) for value in card.tags),
        "enchantments": dict(sorted(card.enchantments.items())),
        "afflictions": {affliction: 1} if affliction is not None else {},
        "effect_vars": dict(sorted(card.effect_vars.items())),
        "combat_vars": {
            key: value
            for key, value in sorted(card.combat_vars.items())
            if isinstance(value, (bool, int, float, str)) or value is None
        },
    }


def _serialize_powers(creature: Any, entity_id: str) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": f"power:{entity_id}:{power_id.name}",
            "owner_id": entity_id,
            "power_id": power_id.name,
            "id": power_id.name,
            "amount": power.amount,
            **serialize_power_fields(power),
        }
        for power_id, power in sorted(
            creature.powers.items(),
            key=lambda pair: pair[0].name,
        )
        if power.amount != 0
    ]


def build_combat_snapshot(
    combat: CombatState,
    *,
    include_candidates: bool = True,
) -> dict[str, Any]:
    owner = combat.primary_player
    owner_id = f"player:{getattr(owner, 'combat_id', 0)}"
    player_state = combat.combat_player_state_for(owner)
    hand = player_state.hand if player_state is not None else combat.hand
    zones = {
        "hand": hand,
        "draw": player_state.draw if player_state is not None else combat.draw_pile,
        "discard": player_state.discard if player_state is not None else combat.discard_pile,
        "exhaust": player_state.exhaust if player_state is not None else combat.exhaust_pile,
    }
    cards: list[dict[str, Any]] = []
    for zone, pile in zones.items():
        for index, card in enumerate(pile):
            cards.append(serialize_card(
                card,
                zone=zone,
                zone_index=index,
                owner_id=owner_id,
                playable=combat.can_play_card(card) if zone == "hand" else None,
            ))
    enemies: list[dict[str, Any]] = []
    powers = _serialize_powers(owner, owner_id)
    for index, enemy in enumerate(combat.enemies):
        entity_id = f"enemy:{enemy.combat_id}"
        ai = combat.enemy_ais.get(enemy.combat_id)
        intents = []
        if ai is not None and enemy.is_alive:
            intents = [
                {
                    "intent_type": intent.intent_type.name,
                    "damage": intent.damage,
                    "hits": intent.hits,
                }
                for intent in ai.current_move.intents
            ]
        enemies.append({
            "entity_id": entity_id,
            "combat_index": index,
            "owner_id": None,
            "monster_id": enemy.monster_id or "UNKNOWN",
            "id": enemy.monster_id or "UNKNOWN",
            "side": enemy.side.name,
            "hp": enemy.current_hp,
            "max_hp": enemy.max_hp,
            "block": enemy.block,
            "alive": enemy.is_alive,
            "is_alive": enemy.is_alive,
            "stars": enemy.stars,
            "intents": intents,
        })
        powers.extend(_serialize_powers(enemy, entity_id))

    potions = []
    owner_potions = player_state.potions if player_state is not None else combat.potions
    for slot, potion in enumerate(owner_potions):
        if potion is None:
            continue
        target_index = next(
            (i for i, enemy in enumerate(combat.enemies) if enemy.is_alive),
            None,
        ) if potion.target_type.name == "ANY_ENEMY" else None
        potions.append({
            "entity_id": f"potion:{owner_id}:{slot}",
            "owner_id": owner_id,
            "slot": slot,
            "potion_id": potion.potion_id,
            "id": potion.potion_id,
            "rarity": potion.rarity.name,
            "usage": potion.usage_type.name,
            "target_type": potion.target_type.name,
            "target": potion.target_type.name.title().replace("_", ""),
            "requires_target": potion.target_type.name == "ANY_ENEMY",
            "can_use": combat.can_use_potion(
                slot,
                target_index=target_index,
                owner=owner,
            ),
        })

    snapshot: dict[str, Any] = {
        "type": "combat_action",
        "phase": "COMBAT",
        "global": {"phase": "COMBAT", "round": combat.round_number},
        "players": [{
            "entity_id": owner_id,
            "player_id": getattr(owner, "combat_id", 0),
            "character_id": combat.character_id,
            "hp": owner.current_hp,
            "max_hp": owner.max_hp,
            "block": owner.block,
            "energy": player_state.energy if player_state is not None else combat.energy,
            "max_energy": player_state.base_max_energy if player_state is not None else combat.max_energy,
            "stars": owner.stars,
        }],
        "cards": cards,
        "creatures": enemies,
        "powers": powers,
        "relics": [],
        "potions": potions,
        "map_nodes": [],
        "map_edges": [],
        # Compatibility projections make fixtures directly consumable by v1.
        "player": {
            "hp": owner.current_hp,
            "max_hp": owner.max_hp,
            "block": owner.block,
            "energy": player_state.energy if player_state is not None else combat.energy,
            "max_energy": player_state.base_max_energy if player_state is not None else combat.max_energy,
            "powers": [
                {"id": item["power_id"], "amount": item["amount"]}
                for item in powers if item["owner_id"] == owner_id
            ],
        },
        "hand": [card for card in cards if card["zone"] == "hand"],
        "enemies": enemies,
        "available_actions": ["PLAY", "END_TURN", "POTION"],
        "draw_pile_count": len(zones["draw"]),
        "discard_pile_count": len(zones["discard"]),
        "exhaust_pile_count": len(zones["exhaust"]),
        "round": combat.round_number,
    }
    if include_candidates:
        snapshot["candidates"] = [
            candidate.to_dict()
            for candidate in build_action_candidates(snapshot)
        ]
    return attach_v2_envelope(snapshot)


def _serialize_player(player: PlayerState) -> dict[str, Any]:
    owner_id = f"player:{player.player_id}"
    return {
        "entity_id": owner_id,
        "player_id": player.player_id,
        "character_id": player.character_id,
        "hp": player.current_hp,
        "max_hp": player.max_hp,
        "gold": player.gold,
        "max_energy": player.max_energy,
        "max_potion_slots": player.max_potion_slots,
        "base_orb_slot_count": player.base_orb_slot_count,
    }


def build_run_snapshot(run_state: RunState) -> dict[str, Any]:
    """Serialize persistent, player-observable run state."""
    players = [_serialize_player(player) for player in run_state.players]
    cards: list[dict[str, Any]] = []
    relics: list[dict[str, Any]] = []
    potions: list[dict[str, Any]] = []
    for player in run_state.players:
        owner_id = f"player:{player.player_id}"
        cards.extend(
            serialize_card(
                card,
                zone="deck",
                zone_index=index,
                owner_id=owner_id,
            )
            for index, card in enumerate(player.deck)
        )
        for index, relic_id in enumerate(player.relics):
            relic_object = (
                player.relic_objects[index]
                if index < len(player.relic_objects)
                else None
            )
            state = (
                serialize_relic_fields(relic_object)
                if relic_object is not None
                else {}
            )
            relics.append({
                "entity_id": f"relic:{owner_id}:{index}:{relic_id}",
                "owner_id": owner_id,
                "relic_id": relic_id,
                "id": relic_id,
                "state": state,
                **state,
            })
        for slot, potion in enumerate(player.potions):
            if potion is None:
                continue
            potions.append({
                "entity_id": f"potion:{owner_id}:{slot}",
                "owner_id": owner_id,
                "slot": slot,
                "potion_id": potion.potion_id,
                "id": potion.potion_id,
                "rarity": potion.rarity.name,
                "usage": potion.usage_type.name,
                "target_type": potion.target_type.name,
                "can_use": potion.can_use_out_of_combat(),
            })

    map_nodes: list[dict[str, Any]] = []
    map_edges: list[dict[str, Any]] = []
    if run_state.map is not None:
        visited = {(coord.col, coord.row) for coord in run_state.visited_map_coords}
        node_templates, edge_templates = _map_topology(run_state.map)
        reachable = set()
        if run_state.visited_map_coords:
            last = run_state.visited_map_coords[-1]
            current = run_state.map.get_point(last)
            if current is not None:
                reachable = {(child.col, child.row) for child in current.children}
        else:
            reachable = {
                (col, row)
                for col, row, _ in node_templates
                if row == 0
            }
        for col, row, node_type in node_templates:
            entity_id = f"map:{col}:{row}"
            map_nodes.append({
                "entity_id": entity_id,
                "row": row,
                "col": col,
                "node_type": node_type,
                "visited": (col, row) in visited,
                "reachable": (col, row) in reachable,
            })
        map_edges.extend({
            "source_id": f"map:{source_col}:{source_row}",
            "target_id": f"map:{target_col}:{target_row}",
        } for source_col, source_row, target_col, target_row in edge_templates)

    return {
        "character_id": run_state.player.character_id,
        "act": run_state.current_act_index + 1,
        "act_id": run_state.current_act.act_id,
        "act_floor": run_state.act_floor,
        "floor": run_state.total_floor,
        "ascension": run_state.ascension_level,
        "is_over": run_state.is_over,
        "player_won": run_state.player_won,
        "players": players,
        "cards": cards,
        "relics": relics,
        "potions": potions,
        "map_nodes": map_nodes,
        "map_edges": map_edges,
    }


def attach_v2_envelope(
    state: dict[str, Any],
    *,
    episode_id: str | None = None,
    decision_id: str | int | None = None,
) -> dict[str, Any]:
    """Attach compatibility metadata without mutating the caller's mapping."""
    result = dict(state)
    result.update(schema_manifest())
    if episode_id is not None:
        result["episode_id"] = episode_id
    if decision_id is not None:
        result["decision_id"] = decision_id
    return result


def build_run_decision_snapshot(manager: RunManager) -> dict[str, Any]:
    """Serialize the current simulator decision using the same v2 wire shape."""
    combat = manager.get_combat_state()
    if combat is not None and combat.pending_choice is not None:
        from sts2_env.parity.bridge_replay import combat_state_to_bridge_state

        state = combat_state_to_bridge_state(combat)
        combat_entities = build_combat_snapshot(
            combat,
            include_candidates=False,
        )
        for key in ("players", "creatures", "powers", "relics", "potions"):
            state[key] = combat_entities.get(key, [])
    elif combat is not None:
        state = build_combat_snapshot(combat, include_candidates=False)
    elif manager.is_over:
        # Gymnasium requires the observation returned by the terminal step.
        # RUN_OVER deliberately has no candidates; the environment's action
        # mask supplies its single sampling-safe fallback slot.
        state = {
            "type": "run_complete",
            "phase": "RUN_OVER",
            "candidates": [],
        }
    else:
        # Reuse the parity-proven phase projection, then enrich it rather than
        # maintaining another independent option-order implementation.
        from sts2_env.parity.bridge_replay import run_manager_to_bridge_state

        state = run_manager_to_bridge_state(manager)
    run_snapshot = build_run_snapshot(manager.run_state)
    result = dict(state)
    result["run_state"] = run_snapshot
    result["run_state_available"] = True
    result["global"] = {
        "phase": _name(manager.phase),
        **{
            key: run_snapshot[key]
            for key in (
                "character_id", "act", "act_id", "act_floor", "floor", "ascension",
                "is_over", "player_won",
            )
        },
    }
    if manager.phase == "EVENT":
        event = getattr(manager, "_event_model", None)
        event_id = getattr(event, "event_id", None)
        if event_id:
            result["event_id"] = event_id
            result["event_context"] = [{
                "entity_id": f"event:{event_id}",
                "id": event_id,
                "zone": "event",
            }]
        if result.get("type") == "event":
            event_actions = [
                item for item in manager.get_available_actions()
                if item.get("action") == "event_choice"
            ]
            if len(event_actions) == len(result.get("options", [])):
                result["legacy_event_options"] = [
                    dict(option) for option in result["options"]
                ]
                result["legacy_event_candidates"] = [
                    candidate.to_dict()
                    for candidate in build_action_candidates(result)
                ]
                for option, action in zip(result["options"], event_actions, strict=True):
                    option["id"] = action.get("option_id")
                    option["option_id"] = action.get("option_id")
                    option["label"] = action.get("label")
                    option["description"] = action.get("description")
                    option["model_content"] = next(
                        (
                            value for value in (action.get("label"), action.get("option_id"))
                            if value and DEFAULT_CATEGORICAL_VOCABULARY.contains(value)
                        ),
                        "EVENT_CHOICE",
                    )
    if manager.phase == "CARD_REWARD":
        actions = manager.get_available_actions()
        for pick_action, skip_action, item_key in (
            ("pick_potion", "skip_potion", "potion_id"),
            ("pick_relic_reward", "skip_relic", "relic_id"),
        ):
            pick = next((item for item in actions if item.get("action") == pick_action), None)
            if pick is None:
                continue
            item_id = str(pick[item_key])
            source_id = f"reward:{item_key}:{item_id}"
            result["reward_item_type"] = item_key
            result["options"] = [{
                "entity_id": source_id,
                "id": item_id,
                item_key: item_id,
                "action": pick_action,
                "index": 0,
            }]
            result["candidates"] = [{
                "candidate_id": f"reward:{pick_action}:{item_id}",
                "action_type": "CHOOSE",
                "source_id": source_id,
                "payload": {"action": pick_action},
                "features": {
                    "model_source_content": item_id,
                    "model_source_zone": "candidate",
                },
            }]
            if any(item.get("action") == skip_action for item in actions):
                result["candidates"].append({
                    "candidate_id": f"reward:{skip_action}:{item_id}",
                    "action_type": "SKIP",
                    "payload": {"action": skip_action},
                    "features": {"model_source_zone": "candidate"},
                })
            break
    result["candidates"] = [
        candidate.to_dict()
        for candidate in build_action_candidates(result)
    ]
    return attach_v2_envelope(result)
