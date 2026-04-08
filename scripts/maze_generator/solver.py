"""BFS solver for maze grids.

Direction encoding (matches benchmark convention):
  U = row - 1  (up)
  D = row + 1  (down)
  L = col - 1  (left)
  R = col + 1  (right)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from .maze_model import MazeGrid

# (delta_row, delta_col, direction_char)
_MOVES = [(-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")]

_PATH_CAP = 50  # max number of shortest paths to return


def verify_reachability(maze: MazeGrid) -> bool:
    """Return True if goal is reachable from start using BFS."""
    start = maze.start
    goal = maze.goal
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc, _ in _MOVES:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and maze.is_walkable(nr, nc):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def bfs_solve(maze: MazeGrid) -> tuple[bool, list[str]]:
    """Find all shortest paths from start to goal.

    Returns:
        (reachable, paths) where paths is a list of direction strings like "RRDDLL".
        At most _PATH_CAP paths are returned to avoid combinatorial explosion.
    """
    start = maze.start
    goal = maze.goal

    # Multi-parent BFS: track all parents that lead to a node at the minimum depth.
    # parents[node] = list of (parent_node, direction_char) that reach node optimally.
    parents: dict[tuple[int, int], list[tuple[tuple[int, int], str]]] = {start: []}
    dist: dict[tuple[int, int], int] = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])
    goal_reached = False

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            goal_reached = True
            # Don't break — allow other nodes at the same depth to finish,
            # but we don't need to expand past goal.
            continue
        # If we already found goal, stop expanding nodes deeper than goal's depth.
        if goal_reached and dist[(r, c)] >= dist[goal]:
            continue
        for dr, dc, direction in _MOVES:
            nr, nc = r + dr, c + dc
            if not maze.is_walkable(nr, nc):
                continue
            new_dist = dist[(r, c)] + 1
            if (nr, nc) not in dist:
                dist[(nr, nc)] = new_dist
                parents[(nr, nc)] = [((r, c), direction)]
                queue.append((nr, nc))
            elif dist[(nr, nc)] == new_dist:
                # Same depth: another optimal parent.
                parents[(nr, nc)].append(((r, c), direction))

    if not goal_reached:
        return False, []

    # Reconstruct all shortest paths by backtracking from goal via parents.
    paths: list[str] = []
    # Stack entries: (current_node, accumulated_path_reversed)
    stack: list[tuple[tuple[int, int], list[str]]] = [(goal, [])]
    while stack and len(paths) < _PATH_CAP:
        node, path_rev = stack.pop()
        if node == start:
            paths.append("".join(reversed(path_rev)))
            continue
        for parent_node, direction in parents[node]:
            stack.append((parent_node, path_rev + [direction]))

    return True, paths
