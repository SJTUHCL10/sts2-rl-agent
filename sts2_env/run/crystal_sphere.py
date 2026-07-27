"""Decompiled-backed Crystal Sphere 11x11 minigame simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sts2_env.core.rng import Rng


GRID_SIZE = 11


class CrystalSphereItemType(str, Enum):
    RELIC = "Relic"
    POTION = "Potion"
    CARD_REWARD = "CardReward"
    CURSE = "Curse"
    GOLD = "Gold"


@dataclass(slots=True)
class CrystalSphereCell:
    x: int
    y: int
    hidden: bool = True
    item_index: int | None = None


@dataclass(slots=True)
class CrystalSphereItem:
    item_type: CrystalSphereItemType
    width: int
    height: int
    rarity: str | None = None
    is_big_gold: bool = False
    position: tuple[int, int] | None = None
    revealed: bool = False

    @property
    def is_good(self) -> bool:
        return self.item_type is not CrystalSphereItemType.CURSE

    @property
    def entity_id(self) -> str:
        if self.position is None:
            return "crystal-item:unplaced"
        x, y = self.position
        return f"crystal-item:{x}:{y}:{self.item_type.value}"

    def public_state(self) -> dict:
        state = {
            "entity_id": self.entity_id,
            "item_type": self.item_type.value,
            "width": self.width,
            "height": self.height,
            "revealed": self.revealed,
            "is_good": self.is_good,
        }
        if self.rarity is not None:
            state["rarity"] = self.rarity
        if self.item_type is CrystalSphereItemType.GOLD:
            state["amount"] = 30 if self.is_big_gold else 10
        return state


def _item_population() -> list[CrystalSphereItem]:
    # Exact order from CrystalSphereMinigame.PopulateItems().
    return [
        CrystalSphereItem(CrystalSphereItemType.RELIC, 4, 4),
        CrystalSphereItem(CrystalSphereItemType.POTION, 1, 3, rarity="Common"),
        CrystalSphereItem(CrystalSphereItemType.POTION, 1, 3, rarity="Common"),
        CrystalSphereItem(CrystalSphereItemType.POTION, 2, 2, rarity="Rare"),
        CrystalSphereItem(CrystalSphereItemType.CARD_REWARD, 2, 2, rarity="Common"),
        CrystalSphereItem(CrystalSphereItemType.CARD_REWARD, 2, 2, rarity="Uncommon"),
        CrystalSphereItem(CrystalSphereItemType.CARD_REWARD, 2, 2, rarity="Rare"),
        CrystalSphereItem(CrystalSphereItemType.CURSE, 2, 2),
        *[
            CrystalSphereItem(
                CrystalSphereItemType.GOLD,
                1,
                1,
                is_big_gold=False,
            )
            for _ in range(5)
        ],
        *[
            CrystalSphereItem(
                CrystalSphereItemType.GOLD,
                2,
                1,
                is_big_gold=True,
            )
            for _ in range(2)
        ],
    ]


class CrystalSphereMinigame:
    """Exact grid, placement, fog, and reveal-order state machine."""

    def __init__(self, rng: Rng, divination_count: int):
        self.rng = rng
        self.divination_count = divination_count
        self.tool = "Big"
        self.cells = [
            [CrystalSphereCell(x, y) for y in range(GRID_SIZE)]
            for x in range(GRID_SIZE)
        ]
        self.items: list[CrystalSphereItem] = []
        self.revealed_indices: list[int] = []
        self.placed_all_items = False
        self._clear_initial_corners()
        attempts = 0
        while not self.placed_all_items and attempts < 10:
            self.placed_all_items = self._populate_items()
            attempts += 1

    @property
    def is_finished(self) -> bool:
        return self.divination_count == 0

    def _horizontal(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (nx, y)
            for nx in (x - 1, x + 1)
            if 0 <= nx < GRID_SIZE
        ]

    def _vertical(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x, ny)
            for ny in (y - 1, y + 1)
            if 0 <= ny < GRID_SIZE
        ]

    def _diagonal(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (nx, ny)
            for nx in (x - 1, x + 1)
            for ny in (y - 1, y + 1)
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
        ]

    def affected_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        if self.tool != "Big":
            return [(x, y)]
        return [
            *self._horizontal(x, y),
            *self._vertical(x, y),
            *self._diagonal(x, y),
            (x, y),
        ]

    def _clear_initial_corners(self) -> None:
        coords = [
            (0, 0),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
            (0, GRID_SIZE - 1),
        ]
        for _ in range(2):
            coords = [
                *coords,
                *[
                    neighbor
                    for coord in coords
                    for neighbor in self._horizontal(*coord)
                ],
                *[
                    neighbor
                    for coord in coords
                    for neighbor in self._vertical(*coord)
                ],
            ]
        for x, y in coords:
            self.cells[x][y].hidden = False

    def _can_place(self, item: CrystalSphereItem, x: int, y: int) -> bool:
        for dx in range(item.width):
            for dy in range(item.height):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    return False
                cell = self.cells[nx][ny]
                if not cell.hidden or cell.item_index is not None:
                    return False
        return True

    def _populate_items(self) -> bool:
        success = True
        for item in _item_population():
            # The game uses `flag = flag && item.PlaceItem(...)`; C# therefore
            # stops calling PlaceItem after the first failure, while still
            # appending the remaining unplaced item objects.
            if not success:
                self.items.append(item)
                continue
            candidates = [
                (x, y)
                for x in range(GRID_SIZE)
                for y in range(GRID_SIZE)
                if self._can_place(item, x, y)
            ]
            if not candidates:
                success = False
                self.items.append(item)
                continue
            x, y = self.rng.choice(candidates)
            item.position = (x, y)
            self.items.append(item)
            item_index = len(self.items) - 1
            for dx in range(item.width):
                for dy in range(item.height):
                    self.cells[x + dx][y + dy].item_index = item_index
        return success

    def click(self, x: int, y: int) -> list[int]:
        if self.is_finished:
            raise ValueError("Crystal Sphere is already finished")
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError(f"Invalid Crystal Sphere coordinate ({x}, {y})")
        if not self.cells[x][y].hidden:
            raise ValueError(f"Crystal Sphere cell ({x}, {y}) is already clear")
        self.divination_count -= 1
        revealed_now: list[int] = []
        for cell_x, cell_y in self.affected_cells(x, y):
            cell = self.cells[cell_x][cell_y]
            if not cell.hidden:
                continue
            cell.hidden = False
            item_index = cell.item_index
            if item_index is None:
                continue
            item = self.items[item_index]
            if item.revealed or item.position is None:
                continue
            item_x, item_y = item.position
            fully_clear = all(
                not self.cells[item_x + dx][item_y + dy].hidden
                for dx in range(item.width)
                for dy in range(item.height)
            )
            if fully_clear:
                item.revealed = True
                self.revealed_indices.append(item_index)
                revealed_now.append(item_index)
        return revealed_now

    def hidden_coordinates(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if self.cells[x][y].hidden
        ]

    def public_snapshot(self) -> dict:
        cells = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                cell = self.cells[x][y]
                item = (
                    self.items[cell.item_index]
                    if cell.item_index is not None
                    else None
                )
                cells.append({
                    "entity_id": f"crystal-cell:{x}:{y}",
                    "x": x,
                    "y": y,
                    "hidden": cell.hidden,
                    "clickable": cell.hidden and not self.is_finished,
                    # Never reveal hidden item identity to the policy.
                    "revealed_item_id": (
                        item.entity_id
                        if item is not None and item.revealed
                        else None
                    ),
                })
        return {
            "type": "crystal_sphere",
            "grid_width": GRID_SIZE,
            "grid_height": GRID_SIZE,
            "divinations_remaining": self.divination_count,
            "tool": self.tool,
            "finished": self.is_finished,
            "placed_all_items": self.placed_all_items,
            "cells": cells,
            "revealed_items": [
                self.items[index].public_state()
                for index in self.revealed_indices
            ],
        }
