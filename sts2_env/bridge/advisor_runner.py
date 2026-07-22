"""Run a trained full-run policy as a read-only in-game advisor.

The companion STS2AdvisorMod sends actionable states on port 9003. This
runner returns text recommendations only; the mod never executes the decoded
command and the player remains responsible for every click and key press.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from sts2_env.bridge.agent_runner import load_model
from sts2_env.bridge.client import STS2GameClient
from sts2_env.bridge.full_run_adapter import FullRunStateAdapter
from sts2_env.bridge.protocol import BridgeAction, BridgeStateType

logger = logging.getLogger(__name__)

DEFAULT_ADVISOR_PORT = 9003
DEFAULT_MODEL_PATH = Path("output/full_run_v0109_smoke_masked_20260721/final_model.zip")


def run_advisor(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = DEFAULT_ADVISOR_PORT,
    deterministic: bool = True,
) -> None:
    """Connect to the passive mod and serve recommendations indefinitely."""
    model = load_model(model_path)
    observation_shape = getattr(model.observation_space, "shape", ())
    action_count = getattr(model.action_space, "n", 0)
    if observation_shape != (151,) or action_count != 157:
        raise ValueError(
            "Advisor MVP requires the full-run 151x157 policy; got "
            f"{observation_shape}x{action_count}."
        )

    adapter = FullRunStateAdapter()
    logger.info("Connecting to STS2 Advisor Mod at %s:%d", host, port)
    # A player may spend arbitrarily long reading cards or planning a turn.
    # Unlike the automatic bridge, idle time is not a connection failure.
    with STS2GameClient(host=host, port=port, timeout=None) as client:
        logger.info("Connected. Recommendations are display-only.")
        while True:
            state = client.receive_state()
            observation = adapter.encode_observation(state)
            mask = adapter.compute_action_mask(state)
            action, _states = model.predict(
                observation,
                action_masks=mask,
                deterministic=deterministic,
            )
            action_index = int(action)
            command = adapter.decode_action(action_index, state)
            summary, details = describe_recommendation(state, command, action_index)
            response = {
                "type": "advice",
                "request_id": state.get("request_id"),
                "fingerprint": state.get("fingerprint"),
                "decision_type": state.get("type", ""),
                "summary": summary,
                "details": details,
                "policy_action": action_index,
                "command": command,
            }
            client.send_action(response)
            logger.info("%s -> %s", state.get("type", "?"), summary)


def describe_recommendation(
    state: dict[str, Any], command: dict[str, Any], action_index: int
) -> tuple[str, str]:
    """Convert a decoded policy command into concise player-facing text."""
    decision_type = state.get("type")
    action = command.get("action")
    details = f"策略动作 #{action_index}（这是动作偏好，不是胜率）"

    if decision_type == BridgeStateType.COMBAT_ACTION:
        if action == BridgeAction.END_TURN:
            return "结束回合", details
        if action == BridgeAction.POTION:
            slot = int(command.get("slot", -1))
            potions = list(state.get("potions", []))
            potion = next((p for p in potions if int(p.get("slot", -2)) == slot), {})
            name = potion.get("id", f"槽位 {slot + 1}")
            target = _enemy_name(state, int(command.get("target_index", -1)))
            suffix = f"，目标 {target}" if target else ""
            return f"使用药水槽 #{slot + 1} {name}{suffix}", details
        if action == BridgeAction.PLAY:
            card_index = int(command.get("card_index", -1))
            hand = list(state.get("hand", []))
            card = hand[card_index] if 0 <= card_index < len(hand) else {}
            name = _card_name(card, card_index)
            target = _enemy_name(state, int(command.get("target_index", -1)))
            suffix = f" → {target}" if target else ""
            return f"打出手牌 #{card_index + 1} {name}{suffix}", details

    if action == BridgeAction.SKIP:
        return "跳过", details

    if action == BridgeAction.CHOOSE:
        index = int(command.get("index", 0))
        if decision_type == BridgeStateType.MAP_SELECT:
            node = _item_by_index(state.get("nodes", []), index)
            room = node.get("type", f"路线 {index + 1}")
            coord = (
                f"（第 {int(node['row']) + 1} 行，第 {int(node['col']) + 1} 列）"
                if "row" in node and "col" in node
                else ""
            )
            return f"选择 {room} {coord}".strip(), details
        if decision_type == BridgeStateType.CARD_REWARD:
            card = _item_by_index(state.get("cards", []), index)
            return f"选择卡牌 #{index + 1} {_card_name(card, index)}", details
        item = _item_by_index(
            state.get("options", state.get("cards", state.get("relics", []))), index
        )
        label = item.get("label", item.get("id", item.get("type", f"选项 {index + 1}")))
        return f"选择 {label}", details

    return "暂时无法解释该建议", details


def _item_by_index(items: Any, index: int) -> dict[str, Any]:
    values = list(items or [])
    for fallback, item in enumerate(values):
        if int(item.get("index", fallback)) == index:
            return item
    return values[index] if 0 <= index < len(values) else {}


def _enemy_name(state: dict[str, Any], index: int) -> str:
    enemies = list(state.get("enemies", []))
    if 0 <= index < len(enemies):
        enemy = enemies[index]
        name = str(enemy.get("id", f"敌人 {index + 1}"))
        hp = enemy.get("hp")
        max_hp = enemy.get("max_hp")
        intent = enemy.get("intent")
        facts = []
        if hp is not None and max_hp is not None:
            facts.append(f"HP {hp}/{max_hp}")
        if intent:
            facts.append(str(intent))
        suffix = f"（{', '.join(facts)}）" if facts else ""
        return f"#{index + 1} {name}{suffix}"
    return ""


def _card_name(card: dict[str, Any], fallback_index: int) -> str:
    name = str(card.get("id", f"第 {fallback_index + 1} 张牌"))
    enchantment = card.get("enchantment")
    if isinstance(enchantment, dict) and enchantment.get("id"):
        amount = int(enchantment.get("amount", 0) or 0)
        amount_text = f" +{amount}" if amount else ""
        name += f" [{enchantment['id']}{amount_text}]"
    affliction = card.get("affliction")
    if isinstance(affliction, dict) and affliction.get("id"):
        amount = int(affliction.get("amount", 0) or 0)
        amount_text = f" +{amount}" if amount else ""
        name += f" [{affliction['id']}{amount_text}]"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Display-only STS2 full-run policy advisor")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_ADVISOR_PORT)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_advisor(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
