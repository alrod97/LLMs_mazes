"""Core data model for the procedural maze generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class CellType(IntEnum):
    FLOOR = 0
    WALL = 1
    TRAP = 2
    START = 3
    GOAL = 4


@dataclass
class MazeConfig:
    rows: int
    cols: int
    wall_density: float = 0.25          # fraction of non-start/goal cells to attempt as walls
    trap_count: int = 0                 # number of trap cells to place
    start_pos: Optional[tuple[int, int]] = None  # (row, col); None = auto-pick
    goal_pos: Optional[tuple[int, int]] = None   # (row, col); None = auto-pick
    force_unreachable: bool = False     # guarantee no path from start to goal
    seed: Optional[int] = 42
    border_walls: bool = False          # wrap grid in a wall border


@dataclass
class MazeGrid:
    rows: int
    cols: int
    grid: list[list[CellType]]
    start: tuple[int, int]             # (row, col)
    goal: tuple[int, int]              # (row, col)
    has_border: bool = False

    def is_walkable(self, row: int, col: int) -> bool:
        """Return True if the cell at (row, col) can be walked on."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        ct = self.grid[row][col]
        return ct in (CellType.FLOOR, CellType.START, CellType.GOAL)
