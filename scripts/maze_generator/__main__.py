"""CLI entry point: generate the five example mazes and write outputs.

Usage (from the repo root):
    python -m scripts.maze_generator

Outputs:
    generated_mazes/<name>.png          — 1024x1024 maze images
    generated_mazes/maze_annotations.json  — ground-truth annotation file
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Allow running as `python -m scripts.maze_generator` from the repo root as
# well as `python __main__.py` from inside this directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.maze_generator.annotation_writer import build_annotations, write_annotations
from scripts.maze_generator.generator import generate_maze
from scripts.maze_generator.maze_model import MazeConfig
from scripts.maze_generator.renderer import render_maze
from scripts.maze_generator.solver import bfs_solve

# ---------------------------------------------------------------------------
# Maze specifications
# ---------------------------------------------------------------------------

@dataclass
class MazeSpec:
    name: str
    rows: int
    cols: int
    wall_density: float
    trap_count: int
    force_unreachable: bool
    palette: str
    border_walls: bool
    seed: int
    start_pos: tuple[int, int] | None = None
    goal_pos: tuple[int, int] | None = None


_SPECS: list[MazeSpec] = [
    # --- GROUP A: Diagnostic (001-008) ---
    MazeSpec(
        name="gen_maze_001",
        rows=5, cols=5,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=1001,
        start_pos=(0, 2), goal_pos=(4, 2),
    ),
    MazeSpec(
        name="gen_maze_002",
        rows=5, cols=5,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=1002,
        start_pos=(2, 0), goal_pos=(2, 4),
    ),
    MazeSpec(
        name="gen_maze_003",
        rows=5, cols=5,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=1003,
        start_pos=(0, 0), goal_pos=(4, 4),
    ),
    MazeSpec(
        name="gen_maze_004",
        rows=5, cols=5,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=1004,
        start_pos=(4, 0), goal_pos=(0, 4),
    ),
    MazeSpec(
        name="gen_maze_005",
        rows=7, cols=7,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=1005,
        start_pos=(0, 3), goal_pos=(6, 3),
    ),
    MazeSpec(
        name="gen_maze_006",
        rows=7, cols=7,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=1006,
        start_pos=(3, 0), goal_pos=(3, 6),
    ),
    MazeSpec(
        name="gen_maze_007",
        rows=7, cols=7,
        wall_density=0.05, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=1007,
        start_pos=(0, 3), goal_pos=(6, 3),
    ),
    MazeSpec(
        name="gen_maze_008",
        rows=7, cols=7,
        wall_density=0.00, trap_count=2,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=1008,
        start_pos=(3, 0), goal_pos=(3, 6),
    ),
    # --- GROUP B: Grid Scale (009-023) ---
    MazeSpec(
        name="gen_maze_009",
        rows=5, cols=5,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=2001,
    ),
    MazeSpec(
        name="gen_maze_010",
        rows=5, cols=5,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=2002,
    ),
    MazeSpec(
        name="gen_maze_011",
        rows=6, cols=6,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=2003,
    ),
    MazeSpec(
        name="gen_maze_012",
        rows=7, cols=7,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=2004,
    ),
    MazeSpec(
        name="gen_maze_013",
        rows=7, cols=7,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=2005,
    ),
    MazeSpec(
        name="gen_maze_014",
        rows=8, cols=8,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=2006,
    ),
    MazeSpec(
        name="gen_maze_015",
        rows=8, cols=8,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=2007,
    ),
    MazeSpec(
        name="gen_maze_016",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=2008,
    ),
    MazeSpec(
        name="gen_maze_017",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=2009,
    ),
    MazeSpec(
        name="gen_maze_018",
        rows=10, cols=10,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=2010,
    ),
    MazeSpec(
        name="gen_maze_019",
        rows=10, cols=10,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=2011,
    ),
    MazeSpec(
        name="gen_maze_020",
        rows=11, cols=11,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=2012,
    ),
    MazeSpec(
        name="gen_maze_021",
        rows=12, cols=12,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=2013,
    ),
    MazeSpec(
        name="gen_maze_022",
        rows=13, cols=13,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=2014,
    ),
    MazeSpec(
        name="gen_maze_023",
        rows=13, cols=13,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=2015,
    ),
    # --- GROUP C: Wall Density (024-038) ---
    MazeSpec(
        name="gen_maze_024",
        rows=9, cols=9,
        wall_density=0.00, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=3001,
    ),
    MazeSpec(
        name="gen_maze_025",
        rows=9, cols=9,
        wall_density=0.05, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=3002,
    ),
    MazeSpec(
        name="gen_maze_026",
        rows=9, cols=9,
        wall_density=0.10, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=3003,
    ),
    MazeSpec(
        name="gen_maze_027",
        rows=9, cols=9,
        wall_density=0.15, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=3004,
    ),
    MazeSpec(
        name="gen_maze_028",
        rows=9, cols=9,
        wall_density=0.20, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=3005,
    ),
    MazeSpec(
        name="gen_maze_029",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=3006,
    ),
    MazeSpec(
        name="gen_maze_030",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=3007,
    ),
    MazeSpec(
        name="gen_maze_031",
        rows=9, cols=9,
        wall_density=0.30, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=3008,
    ),
    MazeSpec(
        name="gen_maze_032",
        rows=9, cols=9,
        wall_density=0.30, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=3009,
    ),
    MazeSpec(
        name="gen_maze_033",
        rows=9, cols=9,
        wall_density=0.35, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=3010,
    ),
    MazeSpec(
        name="gen_maze_034",
        rows=9, cols=9,
        wall_density=0.35, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=3011,
    ),
    MazeSpec(
        name="gen_maze_035",
        rows=9, cols=9,
        wall_density=0.40, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=3012,
    ),
    MazeSpec(
        name="gen_maze_036",
        rows=9, cols=9,
        wall_density=0.40, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=3013,
    ),
    MazeSpec(
        name="gen_maze_037",
        rows=9, cols=9,
        wall_density=0.45, trap_count=0,
        force_unreachable=True,
        palette="forest",
        border_walls=False,
        seed=3014,
    ),
    MazeSpec(
        name="gen_maze_038",
        rows=9, cols=9,
        wall_density=0.45, trap_count=0,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=3015,
    ),
    # --- GROUP D: Trap Ablation (039-050) ---
    MazeSpec(
        name="gen_maze_039",
        rows=7, cols=7,
        wall_density=0.20, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=4001,
    ),
    MazeSpec(
        name="gen_maze_040",
        rows=7, cols=7,
        wall_density=0.20, trap_count=3,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=4001,
    ),
    MazeSpec(
        name="gen_maze_041",
        rows=7, cols=7,
        wall_density=0.20, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=4002,
    ),
    MazeSpec(
        name="gen_maze_042",
        rows=7, cols=7,
        wall_density=0.20, trap_count=3,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=4002,
    ),
    MazeSpec(
        name="gen_maze_043",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=4003,
    ),
    MazeSpec(
        name="gen_maze_044",
        rows=9, cols=9,
        wall_density=0.25, trap_count=4,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=4003,
    ),
    MazeSpec(
        name="gen_maze_045",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=4004,
    ),
    MazeSpec(
        name="gen_maze_046",
        rows=9, cols=9,
        wall_density=0.25, trap_count=4,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=4004,
    ),
    MazeSpec(
        name="gen_maze_047",
        rows=11, cols=11,
        wall_density=0.30, trap_count=0,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=4005,
    ),
    MazeSpec(
        name="gen_maze_048",
        rows=11, cols=11,
        wall_density=0.30, trap_count=5,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=4005,
    ),
    MazeSpec(
        name="gen_maze_049",
        rows=11, cols=11,
        wall_density=0.30, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=4006,
    ),
    MazeSpec(
        name="gen_maze_050",
        rows=11, cols=11,
        wall_density=0.30, trap_count=6,
        force_unreachable=True,
        palette="meadow",
        border_walls=False,
        seed=4006,
    ),
    # --- GROUP E: Unreachable Detection (051-064) ---
    MazeSpec(
        name="gen_maze_051",
        rows=5, cols=5,
        wall_density=0.30, trap_count=0,
        force_unreachable=True,
        palette="forest",
        border_walls=False,
        seed=5001,
    ),
    MazeSpec(
        name="gen_maze_052",
        rows=5, cols=5,
        wall_density=0.40, trap_count=0,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=5002,
    ),
    MazeSpec(
        name="gen_maze_053",
        rows=7, cols=7,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=5003,
    ),
    MazeSpec(
        name="gen_maze_054",
        rows=7, cols=7,
        wall_density=0.35, trap_count=0,
        force_unreachable=True,
        palette="meadow",
        border_walls=False,
        seed=5004,
    ),
    MazeSpec(
        name="gen_maze_055",
        rows=7, cols=7,
        wall_density=0.25, trap_count=3,
        force_unreachable=True,
        palette="forest",
        border_walls=False,
        seed=5005,
    ),
    MazeSpec(
        name="gen_maze_056",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=5006,
    ),
    MazeSpec(
        name="gen_maze_057",
        rows=9, cols=9,
        wall_density=0.35, trap_count=0,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=5007,
    ),
    MazeSpec(
        name="gen_maze_058",
        rows=9, cols=9,
        wall_density=0.30, trap_count=4,
        force_unreachable=True,
        palette="meadow",
        border_walls=False,
        seed=5008,
    ),
    MazeSpec(
        name="gen_maze_059",
        rows=9, cols=9,
        wall_density=0.35, trap_count=0,
        force_unreachable=True,
        palette="forest",
        border_walls=True,
        seed=5009,
    ),
    MazeSpec(
        name="gen_maze_060",
        rows=11, cols=11,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=5010,
    ),
    MazeSpec(
        name="gen_maze_061",
        rows=11, cols=11,
        wall_density=0.35, trap_count=0,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=5011,
    ),
    MazeSpec(
        name="gen_maze_062",
        rows=11, cols=11,
        wall_density=0.30, trap_count=5,
        force_unreachable=True,
        palette="meadow",
        border_walls=False,
        seed=5012,
    ),
    MazeSpec(
        name="gen_maze_063",
        rows=13, cols=13,
        wall_density=0.30, trap_count=0,
        force_unreachable=True,
        palette="forest",
        border_walls=False,
        seed=5013,
    ),
    MazeSpec(
        name="gen_maze_064",
        rows=13, cols=13,
        wall_density=0.35, trap_count=4,
        force_unreachable=True,
        palette="desert",
        border_walls=True,
        seed=5014,
    ),
    # --- GROUP F: Border Walls Ablation (065-074) ---
    MazeSpec(
        name="gen_maze_065",
        rows=7, cols=7,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=6001,
    ),
    MazeSpec(
        name="gen_maze_066",
        rows=7, cols=7,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=True,
        seed=6001,
    ),
    MazeSpec(
        name="gen_maze_067",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=6002,
    ),
    MazeSpec(
        name="gen_maze_068",
        rows=9, cols=9,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=6002,
    ),
    MazeSpec(
        name="gen_maze_069",
        rows=9, cols=9,
        wall_density=0.30, trap_count=3,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=6003,
    ),
    MazeSpec(
        name="gen_maze_070",
        rows=9, cols=9,
        wall_density=0.30, trap_count=3,
        force_unreachable=False,
        palette="dungeon",
        border_walls=True,
        seed=6003,
    ),
    MazeSpec(
        name="gen_maze_071",
        rows=11, cols=11,
        wall_density=0.30, trap_count=0,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=6004,
    ),
    MazeSpec(
        name="gen_maze_072",
        rows=11, cols=11,
        wall_density=0.30, trap_count=0,
        force_unreachable=True,
        palette="meadow",
        border_walls=True,
        seed=6004,
    ),
    MazeSpec(
        name="gen_maze_073",
        rows=13, cols=13,
        wall_density=0.25, trap_count=0,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=6005,
    ),
    MazeSpec(
        name="gen_maze_074",
        rows=13, cols=13,
        wall_density=0.25, trap_count=0,
        force_unreachable=True,
        palette="forest",
        border_walls=True,
        seed=6005,
    ),
    # --- GROUP G: Combined Hard (075-090) ---
    MazeSpec(
        name="gen_maze_075",
        rows=9, cols=9,
        wall_density=0.35, trap_count=3,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=7001,
    ),
    MazeSpec(
        name="gen_maze_076",
        rows=9, cols=9,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=7002,
    ),
    MazeSpec(
        name="gen_maze_077",
        rows=9, cols=9,
        wall_density=0.40, trap_count=2,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=7003,
    ),
    MazeSpec(
        name="gen_maze_078",
        rows=9, cols=9,
        wall_density=0.40, trap_count=3,
        force_unreachable=False,
        palette="meadow",
        border_walls=True,
        seed=7004,
    ),
    MazeSpec(
        name="gen_maze_079",
        rows=10, cols=10,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=7005,
    ),
    MazeSpec(
        name="gen_maze_080",
        rows=10, cols=10,
        wall_density=0.35, trap_count=5,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=7006,
    ),
    MazeSpec(
        name="gen_maze_081",
        rows=10, cols=10,
        wall_density=0.40, trap_count=3,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=7007,
    ),
    MazeSpec(
        name="gen_maze_082",
        rows=10, cols=10,
        wall_density=0.40, trap_count=4,
        force_unreachable=True,
        palette="meadow",
        border_walls=True,
        seed=7008,
    ),
    MazeSpec(
        name="gen_maze_083",
        rows=11, cols=11,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=7009,
    ),
    MazeSpec(
        name="gen_maze_084",
        rows=11, cols=11,
        wall_density=0.35, trap_count=5,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=7010,
    ),
    MazeSpec(
        name="gen_maze_085",
        rows=11, cols=11,
        wall_density=0.40, trap_count=3,
        force_unreachable=True,
        palette="dungeon",
        border_walls=False,
        seed=7011,
    ),
    MazeSpec(
        name="gen_maze_086",
        rows=12, cols=12,
        wall_density=0.35, trap_count=5,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=7012,
    ),
    MazeSpec(
        name="gen_maze_087",
        rows=12, cols=12,
        wall_density=0.40, trap_count=4,
        force_unreachable=True,
        palette="forest",
        border_walls=True,
        seed=7013,
    ),
    MazeSpec(
        name="gen_maze_088",
        rows=13, cols=13,
        wall_density=0.35, trap_count=5,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=7014,
    ),
    MazeSpec(
        name="gen_maze_089",
        rows=13, cols=13,
        wall_density=0.40, trap_count=6,
        force_unreachable=True,
        palette="dungeon",
        border_walls=True,
        seed=7015,
    ),
    MazeSpec(
        name="gen_maze_090",
        rows=13, cols=13,
        wall_density=0.45, trap_count=8,
        force_unreachable=True,
        palette="meadow",
        border_walls=False,
        seed=7016,
    ),
    # --- GROUP H: Palette Stress (091-100) ---
    MazeSpec(
        name="gen_maze_091",
        rows=8, cols=8,
        wall_density=0.25, trap_count=2,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=8001,
    ),
    MazeSpec(
        name="gen_maze_092",
        rows=8, cols=8,
        wall_density=0.25, trap_count=2,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=8001,
    ),
    MazeSpec(
        name="gen_maze_093",
        rows=8, cols=8,
        wall_density=0.25, trap_count=2,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=8001,
    ),
    MazeSpec(
        name="gen_maze_094",
        rows=8, cols=8,
        wall_density=0.25, trap_count=2,
        force_unreachable=False,
        palette="meadow",
        border_walls=False,
        seed=8001,
    ),
    MazeSpec(
        name="gen_maze_095",
        rows=10, cols=10,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="forest",
        border_walls=True,
        seed=8002,
    ),
    MazeSpec(
        name="gen_maze_096",
        rows=10, cols=10,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=8002,
    ),
    MazeSpec(
        name="gen_maze_097",
        rows=10, cols=10,
        wall_density=0.35, trap_count=4,
        force_unreachable=True,
        palette="dungeon",
        border_walls=True,
        seed=8002,
    ),
    MazeSpec(
        name="gen_maze_098",
        rows=10, cols=10,
        wall_density=0.35, trap_count=4,
        force_unreachable=False,
        palette="meadow",
        border_walls=True,
        seed=8002,
    ),
    MazeSpec(
        name="gen_maze_099",
        rows=7, cols=7,
        wall_density=0.15, trap_count=1,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=8003,
    ),
    MazeSpec(
        name="gen_maze_100",
        rows=11, cols=11,
        wall_density=0.30, trap_count=3,
        force_unreachable=False,
        palette="desert",
        border_walls=False,
        seed=8004,
    ),
    # --- GROUP X: Ultra Hard (101-110) ---
    MazeSpec(  # 20x20, dense walls, heavy traps, border
        name="gen_maze_101",
        rows=20, cols=20,
        wall_density=0.45, trap_count=12,
        force_unreachable=False,
        palette="dungeon",
        border_walls=True,
        seed=9001,
    ),
    MazeSpec(  # 20x20, very dense, traps, no border
        name="gen_maze_102",
        rows=20, cols=20,
        wall_density=0.50, trap_count=10,
        force_unreachable=False,
        palette="forest",
        border_walls=False,
        seed=9002,
    ),
    MazeSpec(  # 20x20, max density, max traps, border
        name="gen_maze_103",
        rows=20, cols=20,
        wall_density=0.55, trap_count=15,
        force_unreachable=False,
        palette="desert",
        border_walls=True,
        seed=9003,
    ),
    MazeSpec(  # 20x20, unreachable — looks almost solvable
        name="gen_maze_104",
        rows=20, cols=20,
        wall_density=0.40, trap_count=10,
        force_unreachable=True,
        palette="meadow",
        border_walls=True,
        seed=9004,
    ),
    MazeSpec(  # 20x20, extreme traps (20), moderate walls
        name="gen_maze_105",
        rows=20, cols=20,
        wall_density=0.35, trap_count=20,
        force_unreachable=False,
        palette="dungeon",
        border_walls=False,
        seed=9005,
    ),
    MazeSpec(  # 20x20, ultra dense corridor maze
        name="gen_maze_106",
        rows=20, cols=20,
        wall_density=0.55, trap_count=8,
        force_unreachable=False,
        palette="forest",
        border_walls=True,
        seed=9006,
    ),
    MazeSpec(  # 20x20, unreachable, heavy traps
        name="gen_maze_107",
        rows=20, cols=20,
        wall_density=0.45, trap_count=15,
        force_unreachable=True,
        palette="desert",
        border_walls=False,
        seed=9007,
    ),
    MazeSpec(  # 20x20, moderate walls but massive trap field
        name="gen_maze_108",
        rows=20, cols=20,
        wall_density=0.30, trap_count=25,
        force_unreachable=False,
        palette="meadow",
        border_walls=True,
        seed=9008,
    ),
    MazeSpec(  # 20x20, near-maximum everything
        name="gen_maze_109",
        rows=20, cols=20,
        wall_density=0.50, trap_count=18,
        force_unreachable=False,
        palette="dungeon",
        border_walls=True,
        seed=9009,
    ),
    MazeSpec(  # 20x20, unreachable, maximum chaos
        name="gen_maze_110",
        rows=20, cols=20,
        wall_density=0.50, trap_count=20,
        force_unreachable=True,
        palette="forest",
        border_walls=True,
        seed=9010,
    ),
]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

_OUTPUT_DIR = _REPO_ROOT / "generated_mazes"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    maze_entries: list[tuple[str, object, bool, list[str]]] = []

    # Header
    col_w = [12, 8, 12, 8, 8, 10, 8, 8]
    headers = ["Name", "Grid", "Reachable", "Paths", "PathLen", "WallPct", "Traps", "Time(s)"]
    _print_row(headers, col_w)
    _print_divider(col_w)

    for spec in _SPECS:
        t0 = time.perf_counter()

        config = MazeConfig(
            rows=spec.rows,
            cols=spec.cols,
            wall_density=spec.wall_density,
            trap_count=spec.trap_count,
            force_unreachable=spec.force_unreachable,
            border_walls=spec.border_walls,
            seed=spec.seed,
            start_pos=spec.start_pos,
            goal_pos=spec.goal_pos,
        )

        maze = generate_maze(config)
        reachable, paths = bfs_solve(maze)

        img_path = _OUTPUT_DIR / f"{spec.name}.png"
        render_maze(maze, img_path, image_size=1024, palette=spec.palette)

        elapsed = time.perf_counter() - t0

        # Count actual walls for reporting
        wall_count = sum(
            1
            for r in range(maze.rows)
            for c in range(maze.cols)
            if maze.grid[r][c].value == 1  # CellType.WALL
        )
        total_cells = maze.rows * maze.cols
        wall_pct = 100.0 * wall_count / total_cells

        path_len = len(paths[0]) if paths else 0
        grid_str = f"{spec.rows}x{spec.cols}"

        _print_row(
            [
                spec.name,
                grid_str,
                "Yes" if reachable else "No",
                str(len(paths)),
                str(path_len) if reachable else "-",
                f"{wall_pct:.1f}%",
                str(spec.trap_count),
                f"{elapsed:.2f}",
            ],
            col_w,
        )

        maze_entries.append((spec.name, maze, reachable, paths))

    _print_divider(col_w)

    # Write annotations
    annotations = build_annotations(maze_entries)  # type: ignore[arg-type]
    ann_path = _OUTPUT_DIR / "maze_annotations.json"
    write_annotations(annotations, ann_path)

    print(f"\nImages   -> {_OUTPUT_DIR}/")
    print(f"Annotations -> {ann_path}")
    return 0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_row(cells: list[str], widths: list[int]) -> None:
    parts = [cell.ljust(w) for cell, w in zip(cells, widths)]
    print("  " + "  ".join(parts))


def _print_divider(widths: list[int]) -> None:
    parts = ["-" * w for w in widths]
    print("  " + "  ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
