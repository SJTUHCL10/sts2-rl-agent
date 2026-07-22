"""Read-only spire-codex coverage and version-drift reporter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sts2_env.potions  # noqa: F401
import sts2_env.powers  # noqa: F401
from audit_card_dynamic_vars import collect_card_dynamic_var_mismatches
from audit_card_static_metadata import collect_static_metadata_mismatches
from sts2_env.cards.factory import _factory_registry, _reference_definition
from sts2_env.cards.registry import _CARD_EFFECTS
from sts2_env.core.enums import CardId, PowerId
from sts2_env.potions.base import all_potion_models
from sts2_env.relics.base import RelicId
from sts2_env.relics.registry import RELIC_REGISTRY, load_all_relics


def _load(data_dir: Path, name: str) -> list[dict]:
    return json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))


def _enum_name(value: str) -> str:
    if value == value.upper() and re.fullmatch(r"[A-Z0-9_]+", value):
        return value
    return re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"[^A-Za-z0-9]", "", value)).upper()


def _card_id(value: str) -> CardId | None:
    for candidate in (value, f"{value}_CARD", f"{value}_STATUS"):
        if candidate in CardId.__members__:
            return CardId[candidate]
    return None


def _power_id(value: str) -> PowerId | None:
    name = _enum_name(value)
    for candidate in (name, name.removesuffix("_POWER"), f"{name}_POWER"):
        if candidate in PowerId.__members__:
            return PowerId[candidate]
    return None


def _relic_id(value: str) -> RelicId | None:
    name = _enum_name(value)
    return RelicId.__members__.get(name)


def _coverage(ids: list[str], resolver, implemented: set) -> tuple[int, list[str]]:
    missing = [value for value in ids if (resolved := resolver(value)) is None or resolved not in implemented]
    return len(ids) - len(missing), missing


def generate_report(data_dir: Path, version: str) -> str:
    cards = _load(data_dir, "cards")
    relics = _load(data_dir, "relics")
    potions = _load(data_dir, "potions")
    powers = _load(data_dir, "powers")
    monsters = _load(data_dir, "monsters")
    encounters = _load(data_dir, "encounters")

    card_ids = [_card_id(card["id"]) for card in cards]
    metadata_implemented = set(_factory_registry()) | {
        card_id for card_id in CardId if _reference_definition(card_id) is not None
    }
    card_metadata_count = sum(card_id in metadata_implemented for card_id in card_ids if card_id is not None)
    card_behavior_missing = [
        card["id"] for card, card_id in zip(cards, card_ids)
        if card_id is None or card_id not in _CARD_EFFECTS
    ]

    load_all_relics()
    relic_count, relic_missing = _coverage(
        [relic["id"] for relic in relics], _relic_id, set(RELIC_REGISTRY)
    )
    potion_models = {_enum_name(model.potion_id) for model in all_potion_models()}
    potion_missing = [potion["id"] for potion in potions if _enum_name(potion["id"]) not in potion_models]
    power_count, power_missing = _coverage(
        [power["id"] for power in powers], _power_id, set(PowerId)
    )
    static_mismatches = collect_static_metadata_mismatches()
    dynamic_mismatches = collect_card_dynamic_var_mismatches()

    def names(values: list[str], limit: int = 40) -> str:
        if not values:
            return "None"
        suffix = f" … (+{len(values) - limit})" if len(values) > limit else ""
        return ", ".join(values[:limit]) + suffix

    return "\n".join(
        (
            f"# STS2 {version} spire-codex drift report",
            "",
            f"Source: `{data_dir}`",
            "",
            "## Coverage",
            "",
            "| Surface | Current data | Python coverage | Missing |",
            "|---|---:|---:|---:|",
            f"| Cards: metadata | {len(cards)} | {card_metadata_count} | {len(cards) - card_metadata_count} |",
            f"| Cards: registered play behavior | {len(cards)} | {len(cards) - len(card_behavior_missing)} | {len(card_behavior_missing)} |",
            f"| Relics | {len(relics)} | {relic_count} | {len(relic_missing)} |",
            f"| Potions | {len(potions)} | {len(potions) - len(potion_missing)} | {len(potion_missing)} |",
            f"| Power IDs | {len(powers)} | {power_count} | {len(power_missing)} |",
            f"| Monsters (data inventory) | {len(monsters)} | n/a | n/a |",
            f"| Encounters (data inventory) | {len(encounters)} | n/a | n/a |",
            "",
            "## Reference audits",
            "",
            f"- Card static metadata mismatches: {len(static_mismatches)}",
            f"- Card dynamic variable mismatches: {len(dynamic_mismatches)}",
            "",
            "## Missing IDs / behavior registrations",
            "",
            f"- Card behavior: {names(card_behavior_missing)}",
            f"- Relics: {names(relic_missing)}",
            f"- Potions: {names(potion_missing)}",
            f"- Powers: {names(power_missing)}",
            "",
            "## Manual parity work still required",
            "",
            "- Multiplayer-only targeting, ownership transfer, and cross-player card movement need bridge replay validation.",
            "- Monster move effects and encounter composition are not inferred from JSON; use decompiled sources and scenario tests.",
            "- Event option effects and run-level quest completion remain manual parity surfaces.",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--version", default="unknown")
    parser.add_argument("--output", type=Path, default=Path("reports/codex_drift.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_report(args.data_dir, args.version), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
