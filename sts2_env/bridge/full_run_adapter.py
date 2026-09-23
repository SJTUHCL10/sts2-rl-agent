"""Adapt live Bridge states to the full-run Gymnasium policy interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.bridge.protocol import BridgeAction, BridgeStateType
from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.gym_env.run_env import (
    OBS_ACT_FLOOR_SCALE,
    OBS_ASCENSION_SCALE,
    OBS_CURRENT_ACT_SCALE,
    OBS_DECK_SIZE_SCALE,
    OBS_GOLD_SCALE,
    OBS_MAX_POTION_SLOTS_SCALE,
    OBS_RELIC_COUNT_SCALE,
    OBS_TOTAL_FLOOR_SCALE,
    RUN_OBS_SIZE,
    TOTAL_ACTIONS,
    _BOSS_RELIC_START,
    _CARD_RWD_EXTRA_START,
    _CARD_RWD_START,
    _COMBAT_SIZE,
    _EVENT_START,
    _MAP_START,
    _PHASE_INDEX,
    _REST_START,
    _SHOP_START,
    _TREASURE_START,
)
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE


_TYPE_TO_PHASE = {
    BridgeStateType.MAP_SELECT: "MAP_CHOICE",
    BridgeStateType.COMBAT_ACTION: "COMBAT",
    BridgeStateType.CARD_REWARD: "CARD_REWARD",
    BridgeStateType.REWARD_SCREEN: "CARD_REWARD",
    BridgeStateType.CARD_BUNDLE: "CARD_REWARD",
    BridgeStateType.CARD_SELECT: "CARD_REWARD",
    BridgeStateType.BOSS_RELIC: "BOSS_RELIC",
    BridgeStateType.SHOP: "SHOP",
    BridgeStateType.REST_SITE: "REST_SITE",
    BridgeStateType.EVENT: "EVENT",
    BridgeStateType.CRYSTAL_SPHERE: "EVENT",
    BridgeStateType.TREASURE: "TREASURE",
}


class FullRunStateAdapter:
    """Stateful adapter for policies trained on ``STS2RunEnv``.

    Current Bridge messages expose exact actionable choices but only a subset
    of run metadata on each screen.  The adapter retains the most recently
    observed run values so the 20 run-level features remain stable between
    screens.  Missing values have conservative starter-run defaults.
    """

    def __init__(self, extra_choice_slots: int = 0) -> None:
        if extra_choice_slots < 0:
            raise ValueError("extra_choice_slots must be nonnegative")
        self.extra_choice_slots = extra_choice_slots
        self.total_actions = TOTAL_ACTIONS + extra_choice_slots
        self.combat = StateAdapter()
        self.act = 0
        self.total_floor = 0
        self.act_floor = 0
        self.hp = 80
        self.max_hp = 80
        self.gold = 99
        self.deck_size = 10
        self.relic_count = 1
        self.num_potions = 0
        self.max_potion_slots = 3
        self.ascension = 0
        self.room_type = ""

    def encode_observation(self, state: dict[str, Any]) -> np.ndarray:
        self._update_run_context(state)
        obs = np.zeros(RUN_OBS_SIZE, dtype=np.float32)
        if state.get("type") == BridgeStateType.COMBAT_ACTION or "combat_state" in state:
            obs[:COMBAT_OBS_SIZE] = self.combat.encode_observation(state)

        idx = COMBAT_OBS_SIZE
        obs[idx] = self.act / OBS_CURRENT_ACT_SCALE
        obs[idx + 1] = self.total_floor / OBS_TOTAL_FLOOR_SCALE
        obs[idx + 2] = self.act_floor / OBS_ACT_FLOOR_SCALE
        obs[idx + 3] = self.hp / max(self.max_hp, 1)
        obs[idx + 4] = self.gold / OBS_GOLD_SCALE
        obs[idx + 5] = self.deck_size / OBS_DECK_SIZE_SCALE
        obs[idx + 6] = self.relic_count / OBS_RELIC_COUNT_SCALE
        obs[idx + 7] = self.num_potions / max(self.max_potion_slots, 1)
        obs[idx + 8] = self.max_potion_slots / OBS_MAX_POTION_SLOTS_SCALE
        phase = _TYPE_TO_PHASE.get(str(state.get("type", "")), "MAP_CHOICE")
        obs[idx + 9 + _PHASE_INDEX[phase]] = 1.0
        obs[idx + 17] = self.ascension / OBS_ASCENSION_SCALE
        room = self.room_type.lower()
        obs[idx + 18] = 1.0 if room == "elite" else 0.0
        obs[idx + 19] = 1.0 if room == "boss" else 0.0
        np.clip(obs, -1.0, 10.0, out=obs)
        return obs

    def compute_action_mask(self, state: dict[str, Any]) -> np.ndarray:
        mask = np.zeros(self.total_actions, dtype=np.int8)
        state_type = state.get("type")
        if state_type == BridgeStateType.COMBAT_ACTION:
            mask[:_COMBAT_SIZE] = self.combat.compute_action_mask(state)
        elif state_type == BridgeStateType.MAP_SELECT:
            self._mask_sequence(mask, _MAP_START, state.get("nodes", []), 5)
        elif state_type == BridgeStateType.CARD_SELECT:
            cards = list(state.get("cards", []))
            if int(state.get("min_select", 0) or 0) == 0:
                mask[0] = 1
            for i in range(min(len(cards), _COMBAT_SIZE - 1)):
                mask[1 + i] = 1
        elif state_type == BridgeStateType.REWARD_SCREEN:
            options = self._enabled(state.get("options", []))
            picks = [o for o in options if str(o.get("action", "")).lower() != "proceed"]
            for i in range(min(len(picks), 3)):
                mask[_CARD_RWD_START + i] = 1
            if any(str(o.get("action", "")).lower() == "proceed" for o in options):
                mask[_CARD_RWD_START + 3] = 1
        elif state_type in {BridgeStateType.CARD_REWARD, BridgeStateType.CARD_BUNDLE}:
            items = self._choice_items(state)
            for i in range(min(len(items), 6)):
                mask[_CARD_RWD_START + i if i < 3 else _CARD_RWD_EXTRA_START + i - 3] = 1
            if state.get("can_skip"):
                mask[_CARD_RWD_START + 3] = 1
        elif state_type == BridgeStateType.BOSS_RELIC:
            self._mask_sequence(mask, _BOSS_RELIC_START, self._choice_items(state), 3)
        elif state_type == BridgeStateType.SHOP:
            options = self._enabled(state.get("options", []))
            mask[_SHOP_START] = 1
            buys = [o for o in options if str(o.get("action", "")).lower() != "leave_shop"]
            for i in range(min(len(buys), 9)):
                mask[_SHOP_START + 1 + i] = 1
        elif state_type == BridgeStateType.REST_SITE:
            self._mask_sequence(mask, _REST_START, self._choice_items(state), 5)
        elif state_type in {BridgeStateType.EVENT, BridgeStateType.CRYSTAL_SPHERE}:
            self._mask_sequence(mask, _EVENT_START, self._event_choice_items(state), 4)
        elif state_type == BridgeStateType.TREASURE:
            mask[_TREASURE_START] = 1
        for index, option in enumerate(
            self._extra_choice_items(state)[:self.extra_choice_slots]
        ):
            if option.get("enabled", True):
                mask[TOTAL_ACTIONS + index] = 1
        if not mask.any():
            mask[0] = 1
        return mask

    def decode_action(self, action: int, state: dict[str, Any]) -> dict[str, Any]:
        if action >= TOTAL_ACTIONS:
            return self._choose(
                self._extra_choice_items(state), action - TOTAL_ACTIONS,
            )
        state_type = state.get("type")
        if state_type == BridgeStateType.COMBAT_ACTION:
            decoded = self.combat.decode_action(action, state)
            if decoded["type"] == "END_TURN":
                return {"action": BridgeAction.END_TURN}
            if decoded.get("out_of_hand"):
                return {
                    "action": BridgeAction.POTION,
                    "slot": decoded["potion_slot"],
                    "target_index": decoded.get("target_index", -1),
                }
            return {
                "action": BridgeAction.PLAY,
                "card_index": decoded["card_index"],
                "target_index": decoded.get("target_index", -1),
            }
        if state_type == BridgeStateType.MAP_SELECT:
            local = action - _MAP_START
            nodes = list(state.get("nodes", []))
            if 0 <= local < len(nodes):
                self.room_type = str(nodes[local].get("type", ""))
            return self._choose(nodes, local)
        if state_type == BridgeStateType.CARD_SELECT:
            if action == 0:
                return {"action": BridgeAction.SKIP}
            return self._choose(state.get("cards", []), action - 1)
        if state_type == BridgeStateType.REWARD_SCREEN:
            options = self._enabled(state.get("options", []))
            if action == _CARD_RWD_START + 3:
                proceeds = [o for o in options if str(o.get("action", "")).lower() == "proceed"]
                return self._choose(proceeds or options, 0)
            picks = [o for o in options if str(o.get("action", "")).lower() != "proceed"]
            return self._choose(picks, action - _CARD_RWD_START)
        if state_type in {BridgeStateType.CARD_REWARD, BridgeStateType.CARD_BUNDLE}:
            local = action - _CARD_RWD_START if action < _CARD_RWD_EXTRA_START else 3 + action - _CARD_RWD_EXTRA_START
            items = self._choice_items(state)
            if action == _CARD_RWD_START + 3:
                return {"action": BridgeAction.SKIP}
            return self._choose(items, local)
        if state_type == BridgeStateType.BOSS_RELIC:
            return self._choose(self._choice_items(state), action - _BOSS_RELIC_START)
        if state_type == BridgeStateType.SHOP:
            options = self._enabled(state.get("options", []))
            if action == _SHOP_START:
                leaves = [o for o in options if str(o.get("action", "")).lower() == "leave_shop"]
                return self._choose(leaves or options, 0)
            buys = [o for o in options if str(o.get("action", "")).lower() != "leave_shop"]
            return self._choose(buys, action - _SHOP_START - 1)
        if state_type == BridgeStateType.REST_SITE:
            return self._choose(self._choice_items(state), action - _REST_START)
        if state_type in {BridgeStateType.EVENT, BridgeStateType.CRYSTAL_SPHERE}:
            return self._choose(self._event_choice_items(state), action - _EVENT_START)
        if state_type == BridgeStateType.TREASURE:
            return self._choose(self._choice_items(state), 0)
        return {"action": BridgeAction.SKIP}

    def _update_run_context(self, state: dict[str, Any]) -> None:
        run = state.get("run_state", {}) if isinstance(state.get("run_state"), dict) else {}
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        if not player and isinstance(run.get("player"), dict):
            player = run["player"]
        if not player and isinstance(run.get("players"), list) and run["players"]:
            player = run["players"][0]
        self.act = max(0, int(state.get("act", run.get("act", self.act + 1))) - 1)
        self.total_floor = int(state.get("floor", run.get("floor", self.total_floor)) or self.total_floor)
        self.act_floor = int(state.get("act_floor", run.get("act_floor", self.total_floor)) or self.act_floor)
        self.hp = int(player.get("hp", state.get("hp", self.hp)) or self.hp)
        self.max_hp = int(player.get("max_hp", state.get("max_hp", self.max_hp)) or self.max_hp)
        self.gold = int(player.get("gold", run.get("gold", state.get("gold", self.gold))) or 0)
        deck = run.get("deck", run.get("cards"))
        self.deck_size = len(deck) if isinstance(deck, list) else int(state.get("deck_size", self.deck_size) or 0)
        relics = run.get("relics")
        self.relic_count = len(relics) if isinstance(relics, list) else int(state.get("relic_count", self.relic_count) or 0)
        potions = state.get("potions", run.get("potions"))
        if isinstance(potions, list):
            self.num_potions = sum(1 for potion in potions if potion)
        self.max_potion_slots = int(player.get(
            "max_potion_slots",
            run.get("max_potion_slots", state.get("max_potion_slots", self.max_potion_slots)),
        ) or 1)
        self.ascension = int(run.get("ascension", state.get("ascension", self.ascension)) or 0)

    @staticmethod
    def _enabled(items: Any) -> list[dict[str, Any]]:
        return [item for item in list(items or []) if bool(item.get("enabled", True))]

    def _choice_items(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("options", "cards", "bundles", "relics"):
            if state.get(key):
                return self._enabled(state[key])
        return []

    def _event_choice_items(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        choices = self._choice_items(state)
        if state.get("type") != BridgeStateType.CRYSTAL_SPHERE:
            return choices
        minigame = state.get("minigame")
        if not isinstance(minigame, dict):
            return choices
        finished = bool(minigame.get("finished"))
        remaining = minigame.get("divinations_remaining")
        if not finished and remaining != 0:
            return choices
        proceeds = [
            option
            for option in choices
            if str(option.get("action", "")).lower() == "proceed"
        ]
        return proceeds or choices

    def _extra_choice_items(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Additional choices share one tail of the entity action layout."""
        state_type = state.get("type")
        if state_type == BridgeStateType.MAP_SELECT:
            return list(state.get("nodes", []))[5:]
        if state_type == BridgeStateType.CARD_SELECT:
            return self._choice_items(state)[_COMBAT_SIZE - 1:]
        if state_type == BridgeStateType.REWARD_SCREEN:
            options = self._enabled(state.get("options", []))
            return [
                option for option in options
                if str(option.get("action", "")).lower() != "proceed"
            ][3:]
        if state_type in {BridgeStateType.CARD_REWARD, BridgeStateType.CARD_BUNDLE}:
            return self._choice_items(state)[6:]
        if state_type == BridgeStateType.BOSS_RELIC:
            return self._choice_items(state)[3:]
        if state_type == BridgeStateType.SHOP:
            options = self._enabled(state.get("options", []))
            return [
                option for option in options
                if str(option.get("action", "")).lower() != "leave_shop"
            ][9:]
        if state_type == BridgeStateType.REST_SITE:
            return self._choice_items(state)[5:]
        if state_type in {BridgeStateType.EVENT, BridgeStateType.CRYSTAL_SPHERE}:
            return self._event_choice_items(state)[4:]
        return []

    @staticmethod
    def _mask_sequence(mask: np.ndarray, start: int, items: Any, limit: int) -> None:
        for i, item in enumerate(list(items or [])[:limit]):
            if bool(item.get("enabled", True)):
                mask[start + i] = 1

    @staticmethod
    def _choose(items: Any, local: int) -> dict[str, Any]:
        choices = list(items or [])
        if not choices:
            return {"action": BridgeAction.SKIP}
        local = max(0, min(local, len(choices) - 1))
        return {
            "action": BridgeAction.CHOOSE,
            "index": int(choices[local].get("index", local)),
        }
