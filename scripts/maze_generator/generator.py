"""Procedural maze generator.

Algorithm:
  1. Start with an all-FLOOR grid.
  2. Pick START and GOAL positions.  When border_walls=True the positions are
     chosen on the second-innermost ring so they survive the border pass.
  3. Incrementally attempt to place walls on non-border, non-start/goal cells.
     Each candidate wall is only placed if it does not break reachability
     (unless force_unreachable=True).
  4. Place TRAP cells on remaining FLOOR cells.
  5. If force_unreachable, verify the maze is actually unreachable and add
     more walls if still reachable.
  6. If border_walls, overwrite the outermost ring with WALL cells (start/goal
     are already interior so they are never overwritten).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

from .maze_model import CellType, MazeConfig, MazeGrid
from .solver import verify_reachability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(rows: int, cols: int, fill: CellType = CellType.FLOOR) -> list[list[CellType]]:
    return [[fill] * cols for _ in range(rows)]


def generate_varied_start_goal(
    rows: int,
    cols: int,
    rng: random.Random,
    border_offset: int = 0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pick start and goal on opposite edges with a decent Manhattan distance.

    When border_offset > 0 the positions are kept at least border_offset cells
    away from the grid boundary (used when border_walls=True so that the border
    wall pass never overwrites start/goal).

    Strategy:
      - Randomly choose a primary axis (horizontal or vertical).
      - Place start on one edge side, goal on the opposite edge side.
      - Require minimum Manhattan distance of (rows + cols) // 3.
    """
    o = border_offset  # shorthand
    min_dist = (rows + cols) // 3

    # Inner bounds
    r_lo, r_hi = o, rows - 1 - o
    c_lo, c_hi = o, cols - 1 - o

    if r_lo > r_hi or c_lo > c_hi:
        # Grid is too small for border offset; fall back to no offset
        r_lo, r_hi = 0, rows - 1
        c_lo, c_hi = 0, cols - 1

    for _ in range(400):
        axis = rng.randint(0, 1)  # 0 = top/bottom rows, 1 = left/right cols
        if axis == 0:
            start_r = r_lo
            goal_r = r_hi
            start_c = rng.randint(c_lo, c_hi)
            goal_c = rng.randint(c_lo, c_hi)
            start = (start_r, start_c)
            goal = (goal_r, goal_c)
        else:
            start_c = c_lo
            goal_c = c_hi
            start_r = rng.randint(r_lo, r_hi)
            goal_r = rng.randint(r_lo, r_hi)
            start = (start_r, start_c)
            goal = (goal_r, goal_c)

        dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if dist >= min_dist and start != goal:
            return start, goal

    # Fallback: inner top-left / bottom-right
    return (r_lo, c_lo), (r_hi, c_hi)


def _apply_border_walls(grid: list[list[CellType]], rows: int, cols: int) -> None:
    """Overwrite the outermost ring of cells with WALL."""
    for c in range(cols):
        grid[0][c] = CellType.WALL
        grid[rows - 1][c] = CellType.WALL
    for r in range(1, rows - 1):
        grid[r][0] = CellType.WALL
        grid[r][cols - 1] = CellType.WALL


# ---------------------------------------------------------------------------
# Force-block helper (used when force_unreachable=True but BFS still finds a path)
# ---------------------------------------------------------------------------

def _force_block_path(
    grid: list[list[CellType]],
    maze: MazeGrid,
    rng: random.Random,
) -> None:
    """Aggressively wall off the shortest path to make the maze unreachable."""
    rows, cols = maze.rows, maze.cols
    start, goal = maze.start, maze.goal
    _MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs_path_nodes() -> Optional[list[tuple[int, int]]]:
        visited: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            node = q.popleft()
            if node == goal:
                path: list[tuple[int, int]] = []
                cur: Optional[tuple[int, int]] = goal
                while cur is not None:
                    path.append(cur)
                    cur = visited[cur]
                return list(reversed(path))
            r, c = node
            for dr, dc in _MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    ct = grid[nr][nc]
                    if ct in (CellType.FLOOR, CellType.TRAP, CellType.GOAL):
                        visited[(nr, nc)] = node
                        q.append((nr, nc))
        return None

    for _ in range(30):
        path = bfs_path_nodes()
        if path is None:
            break
        interior = [p for p in path if p != start and p != goal]
        if not interior:
            break
        # Wall off the middle cell for guaranteed progress
        mid = interior[len(interior) // 2]
        grid[mid[0]][mid[1]] = CellType.WALL


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_maze(config: MazeConfig) -> MazeGrid:
    rng = random.Random(config.seed)

    # --- 1. Grid initialisation ---
    grid = _make_grid(config.rows, config.cols, CellType.FLOOR)

    # When border_walls=True, start/goal must stay off the outermost ring so
    # the border pass never overwrites them.
    border_offset = 1 if config.border_walls else 0

    # --- 2. Pick start and goal ---
    if config.start_pos is not None and config.goal_pos is not None:
        start = config.start_pos
        goal = config.goal_pos
    else:
        start, goal = generate_varied_start_goal(
            config.rows, config.cols, rng, border_offset=border_offset
        )

    grid[start[0]][start[1]] = CellType.START
    grid[goal[0]][goal[1]] = CellType.GOAL

    # --- 3. Build candidate wall positions ---
    # Exclude start, goal, and (when border_walls) the outermost ring.
    # When border_walls=True, pre-stamp the border as WALL cells NOW so that
    # reachability checks during wall placement already account for them.
    # This prevents the border pass from later breaking a path that looked
    # valid before the border existed.
    excluded: set[tuple[int, int]] = {start, goal}
    if config.border_walls:
        for c in range(config.cols):
            excluded.add((0, c))
            excluded.add((config.rows - 1, c))
        for r in range(config.rows):
            excluded.add((r, 0))
            excluded.add((r, config.cols - 1))
        # Pre-apply border walls so reachability checks see the final topology.
        _apply_border_walls(grid, config.rows, config.cols)

    candidates: list[tuple[int, int]] = [
        (r, c)
        for r in range(config.rows)
        for c in range(config.cols)
        if (r, c) not in excluded
    ]
    rng.shuffle(candidates)

    num_walls_target = int(len(candidates) * config.wall_density)

    maze_obj = MazeGrid(
        rows=config.rows,
        cols=config.cols,
        grid=grid,
        start=start,
        goal=goal,
        has_border=config.border_walls,
    )

    walls_placed = 0

    for pos in candidates:
        if walls_placed >= num_walls_target:
            break
        r, c = pos
        grid[r][c] = CellType.WALL
        if config.force_unreachable:
            walls_placed += 1
        else:
            if verify_reachability(maze_obj):
                walls_placed += 1
            else:
                grid[r][c] = CellType.FLOOR  # reachability broken — undo

    # --- 4. Place traps on remaining FLOOR cells (not excluded) ---
    # When force_unreachable is False, each trap placement is checked for
    # reachability and reverted if it would break the path.
    floor_cells = [
        (r, c)
        for r in range(config.rows)
        for c in range(config.cols)
        if grid[r][c] == CellType.FLOOR and (r, c) not in excluded
    ]
    rng.shuffle(floor_cells)
    traps_to_place = min(config.trap_count, len(floor_cells))
    traps_placed = 0
    for pos in floor_cells:
        if traps_placed >= traps_to_place:
            break
        r, c = pos
        grid[r][c] = CellType.TRAP
        if config.force_unreachable:
            traps_placed += 1
        else:
            if verify_reachability(maze_obj):
                traps_placed += 1
            else:
                grid[r][c] = CellType.FLOOR  # reachability broken — undo

    # --- 5. Force unreachable (extra verification / enforcement) ---
    if config.force_unreachable:
        if verify_reachability(maze_obj):
            _force_block_path(grid, maze_obj, rng)

    # --- 6. Apply border walls (idempotent — already applied above if border_walls) ---
    if config.border_walls:
        # Re-apply to handle any cells that may have been reverted during wall
        # placement.  This is safe since start/goal are interior cells.
        _apply_border_walls(grid, config.rows, config.cols)

    return MazeGrid(
        rows=config.rows,
        cols=config.cols,
        grid=grid,
        start=start,
        goal=goal,
        has_border=config.border_walls,
    )
