"""Generate ``docs/CARDS_REFERENCE.md`` from spire-codex card JSON.

The runtime card factory consumes this file, so it must be refreshed whenever
new cards are introduced.  This script deliberately emits only normalized,
machine-consumable fields rather than localized descriptions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


POOL_ORDER = (
    "ironclad",
    "silent",
    "defect",
    "necrobinder",
    "regent",
    "colorless",
    "event",
    "token",
    "status",
    "curse",
    "quest",
)


def _display_pool(pool: str) -> str:
    return pool.replace("_", " ").title()


def _format_cost(card: dict[str, Any]) -> str:
    if card.get("is_x_cost"):
        cost = "X"
    elif card["cost"] == -1:
        cost = "Unplayable"
    else:
        cost = str(card["cost"])
    if card.get("star_cost") is not None:
        cost += f" | StarCost: {card['star_cost']}"
    return cost


def _format_vars(card: dict[str, Any]) -> str:
    values = card.get("vars") or {}
    return "{" + ", ".join(
        f"{''.join(part.title() for part in key.split('_'))}: {value}"
        for key, value in values.items()
    ) + "}"


def _upgrade_field(card: dict[str, Any], key: str) -> str:
    normalized = key.replace("_", "").lower()
    alias = {
        "calculationbase": "calc_base",
        "calculationextra": "calc_extra",
    }.get(normalized)
    if alias is None and normalized.endswith("power"):
        alias = normalized.removesuffix("power")
    if alias is not None:
        for var_name in (card.get("vars") or {}):
            if var_name.replace("_", "").lower() == alias.replace("_", ""):
                return "".join(part.title() for part in var_name.split("_"))
    for var_name in (card.get("vars") or {}):
        if var_name.replace("_", "").lower() == normalized:
            return "".join(part.title() for part in var_name.split("_"))
    return "".join(part.title() for part in key.split("_"))


def _format_upgrade(card: dict[str, Any]) -> str:
    upgrade = card.get("upgrade")
    if not upgrade:
        return "Cannot be upgraded"
    changes: list[str] = []
    for key, value in upgrade.items():
        if key == "cost":
            changes.append(f"Cost{int(value) - int(card['cost']):+d}")
        elif key.startswith("add_") and value:
            changes.append(f"Add {_upgrade_field(card, key[4:])}")
        elif key.startswith("remove_") and value:
            changes.append(f"Remove {_upgrade_field(card, key[7:])}")
        elif key == "description_changed":
            changes.append("Description changed")
        elif isinstance(value, str) and value[:1] in {"+", "-"}:
            changes.append(f"{_upgrade_field(card, key)}{value}")
    return "; ".join(changes) or "Description changed"


def generate(cards: list[dict[str, Any]]) -> str:
    cards = sorted(cards, key=lambda card: (POOL_ORDER.index(card["color"]), card["id"]))
    counts = Counter(card["color"] for card in cards)
    lines = [
        "# Slay the Spire 2 - Complete Card Reference",
        "",
        f"**Total cards parsed: {len(cards)}**",
        "",
        "Generated from spire-codex JSON. This file is also consumed by the Python card factory.",
        "",
    ]
    for pool in POOL_ORDER:
        pool_cards = [card for card in cards if card["color"] == pool]
        if not pool_cards:
            continue
        lines.extend((f"## {_display_pool(pool)}", "", f"*{counts[pool]} cards*", ""))
        for card in pool_cards:
            keywords = ", ".join(card.get("keywords") or ()) or "None"
            tags = ", ".join(card.get("tags") or ()) or "None"
            lines.extend(
                (
                    f"### {card['name']}",
                    f"- **ID:** {card['id']}",
                    f"- **Color:** {card['color']}",
                    f"- **Cost:** {_format_cost(card)}",
                    f"- **Type:** {card['type']}",
                    f"- **Rarity:** {card['rarity']}",
                    f"- **Target:** {card['target']}",
                    f"- **Keywords:** {keywords}",
                    f"- **Tags:** {tags}",
                    f"- **Vars:** {_format_vars(card)}",
                    f"- **Upgrade:** {_format_upgrade(card)}",
                    "",
                )
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cards_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/CARDS_REFERENCE.md"))
    args = parser.parse_args()
    cards = json.loads(args.cards_json.read_text(encoding="utf-8"))
    # spire-codex also exposes calculated and alias vars for presentation.
    # CardInstance stores the literal CanonicalVars parsed from C# instead.
    from sts2_env.cards.reference_static_metadata import reference_dynamic_vars_by_card_id
    from sts2_env.core.enums import CardId

    canonical_vars = reference_dynamic_vars_by_card_id()
    for card in cards:
        card_id = CardId.__members__.get(card["id"])
        if card_id is None:
            card_id = CardId.__members__.get(f"{card['id']}_CARD")
        if card_id is None:
            card_id = CardId.__members__.get(f"{card['id']}_STATUS")
        if card_id is not None:
            card["vars"] = canonical_vars.get(card_id, {})
    args.output.write_text(generate(cards), encoding="utf-8")
    print(f"Wrote {len(cards)} cards to {args.output}")


if __name__ == "__main__":
    main()
