"""Ground-truth annotation builder and writer.

Produces a dict compatible with
  benchmark/ground_truth/maze_annotations.json

Schema per maze:
  {
    "maze_name": {
      "reachable": true | false,
      "accepted_shortest_paths": ["RRDDLL", ...]
    }
  }

Validation rules:
  - If reachable=False, accepted_shortest_paths must be empty.
  - If reachable=True, accepted_shortest_paths must be non-empty and all
    paths must have identical length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .maze_model import MazeGrid


def build_annotations(
    mazes: list[tuple[str, MazeGrid, bool, list[str]]],
) -> dict[str, dict]:
    """Build the annotation dict from a list of maze results.

    Args:
        mazes: List of (maze_name, maze_grid, reachable, paths) tuples.
               *paths* is the list of accepted shortest-path strings (may be
               empty when reachable=False).

    Returns:
        Dict mapping maze_name -> {"reachable": bool, "accepted_shortest_paths": [...]}.

    Raises:
        ValueError: If validation fails for any maze entry.
    """
    annotations: dict[str, dict] = {}

    for maze_name, maze_grid, reachable, paths in mazes:
        _validate(maze_name, reachable, paths)
        annotations[maze_name] = {
            "reachable": reachable,
            "accepted_shortest_paths": paths,
        }

    return annotations


def write_annotations(annotations: dict, output_path: Path) -> None:
    """Serialise *annotations* to a JSON file at *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2)
        fh.write("\n")


def _validate(maze_name: str, reachable: bool, paths: list[str]) -> None:
    if not reachable:
        if paths:
            raise ValueError(
                f"[{maze_name}] reachable=False but paths is non-empty: {paths}"
            )
        return

    # reachable=True checks
    if not paths:
        raise ValueError(
            f"[{maze_name}] reachable=True but no paths were provided."
        )

    lengths = {len(p) for p in paths}
    if len(lengths) > 1:
        raise ValueError(
            f"[{maze_name}] Paths have inconsistent lengths: {lengths}. "
            f"All accepted shortest paths must be the same length."
        )

    # Sanity-check that paths use only valid direction characters.
    valid_chars = set("UDLR")
    for path in paths:
        bad = set(path) - valid_chars
        if bad:
            raise ValueError(
                f"[{maze_name}] Path contains invalid characters {bad}: '{path}'"
            )
