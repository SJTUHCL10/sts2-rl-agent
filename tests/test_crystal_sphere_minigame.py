"""0.109.0 decompiled-parity tests for the Crystal Sphere minigame."""

from __future__ import annotations

from pathlib import Path

from sts2_env.agent_v2.candidates import build_action_candidates
from sts2_env.core.rng import Rng
from sts2_env.run.crystal_sphere import (
    CrystalSphereItemType,
    CrystalSphereMinigame,
)


def test_seeded_population_matches_dotnet_rng_golden_layout() -> None:
    rng = Rng(109)
    game = CrystalSphereMinigame(rng, 3)

    assert rng.counter == 15
    assert game.placed_all_items is True
    assert [(item.item_type, item.rarity, item.position) for item in game.items] == [
        (CrystalSphereItemType.RELIC, None, (4, 6)),
        (CrystalSphereItemType.POTION, "Common", (8, 4)),
        (CrystalSphereItemType.POTION, "Common", (10, 3)),
        (CrystalSphereItemType.POTION, "Rare", (6, 2)),
        (CrystalSphereItemType.CARD_REWARD, "Common", (2, 5)),
        (CrystalSphereItemType.CARD_REWARD, "Uncommon", (5, 4)),
        (CrystalSphereItemType.CARD_REWARD, "Rare", (4, 1)),
        (CrystalSphereItemType.CURSE, None, (0, 5)),
        (CrystalSphereItemType.GOLD, None, (4, 3)),
        (CrystalSphereItemType.GOLD, None, (4, 4)),
        (CrystalSphereItemType.GOLD, None, (6, 1)),
        (CrystalSphereItemType.GOLD, None, (5, 10)),
        (CrystalSphereItemType.GOLD, None, (5, 3)),
        (CrystalSphereItemType.GOLD, None, (1, 3)),
        (CrystalSphereItemType.GOLD, None, (2, 1)),
    ]

    initially_clear = {
        (x, y)
        for x in range(11)
        for y in range(11)
        if not game.cells[x][y].hidden
    }
    assert len(initially_clear) == 24
    assert all(
        (x, y) not in initially_clear
        for item in game.items
        if item.position is not None
        for x in range(item.position[0], item.position[0] + item.width)
        for y in range(item.position[1], item.position[1] + item.height)
    )


def test_big_tool_order_reveal_and_hidden_information_boundary() -> None:
    game = CrystalSphereMinigame(Rng(109), 3)
    assert game.affected_cells(5, 5) == [
        (4, 5), (6, 5),
        (5, 4), (5, 6),
        (4, 4), (4, 6), (6, 4), (6, 6),
        (5, 5),
    ]
    before = game.public_snapshot()
    assert before["revealed_items"] == []
    assert all(cell["revealed_item_id"] is None for cell in before["cells"])

    # The 1x1 gold at (4, 4) is fully uncovered by this click.
    revealed = game.click(5, 5)
    assert game.divination_count == 2
    assert revealed
    snapshot = game.public_snapshot()
    assert snapshot["revealed_items"]
    assert all(
        cell["revealed_item_id"] is None
        for cell in snapshot["cells"]
        if cell["hidden"]
    )


def test_crystal_candidates_are_coordinate_stable_across_reordering() -> None:
    state = {
        "type": "crystal_sphere",
        "options": [
            {
                "index": 7,
                "action": "divine_cell",
                "x": 3,
                "y": 4,
                "entity_id": "crystal-cell:3:4",
                "affected_cell_ids": ["crystal-cell:3:4"],
                "enabled": True,
            },
            {"index": 8, "action": "proceed", "enabled": True},
        ],
    }
    candidates = build_action_candidates(state)

    assert [candidate.candidate_id for candidate in candidates] == [
        "crystal_sphere:divine:3:4",
        "crystal_sphere:proceed",
    ]
    assert candidates[0].features["affected_cell_ids"] == ["crystal-cell:3:4"]


def test_python_logic_is_pinned_to_decompiled_crystal_sphere_contract() -> None:
    root = Path(__file__).parents[1] / "decompiled"
    minigame = (
        root
        / "MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent"
        / "CrystalSphereMinigame.cs"
    ).read_text(encoding="utf-8")
    event = (
        root / "MegaCrit.Sts2.Core.Models.Events" / "CrystalSphere.cs"
    ).read_text(encoding="utf-8")

    for source_fragment in (
        "private const int _defaultWidth = 11;",
        "new CrystalSphereRelic()",
        "new CrystalSpherePotion(PotionRarity.Common)",
        "new CrystalSpherePotion(PotionRarity.Rare)",
        "new CrystalSphereCardReward(CardRarity.Common",
        "new CrystalSphereCardReward(CardRarity.Uncommon",
        "new CrystalSphereCardReward(CardRarity.Rare",
        "new CrystalSphereCurse()",
        "new CrystalSphereGold(isBig: false)",
        "new CrystalSphereGold(isBig: true)",
        "GetHorizontalCells(x, y).Concat(GetVerticalCells(x, y))",
    ):
        assert source_fragment in minigame
    assert "new CrystalSphereMinigame(base.Owner, base.Rng, 3)" in event
    assert "new CrystalSphereMinigame(base.Owner, base.Rng, 6)" in event


def test_live_serializer_exposes_public_grid_without_hidden_item_leakage() -> None:
    source = (
        Path(__file__).parents[1]
        / "bridge_mod"
        / "RlCrystalSphereScreenHandler.cs"
    ).read_text(encoding="utf-8")

    assert '"minigame"' in source
    assert '"crystal_cells"' in source
    assert '"divinations_remaining"' in source
    assert '"affected_cell_ids"' in source
    assert 'GetField(\n            "_revealed"' in source
    assert "cell.Item.ToSerializable()" not in source
