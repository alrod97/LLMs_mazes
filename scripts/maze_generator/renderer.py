"""Full image renderer for MazeGrid objects.

Renders a 1024x1024 PNG with:
  - Dark background
  - Centred grid of sprite tiles
  - Outer border frame around the grid
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from .maze_model import MazeGrid
from .sprites import generate_sprite_sheet, _PALETTES

# Background fill colour
_BG_COLOUR = (18, 18, 22)

# Border frame colour & thickness (drawn just outside the grid rect)
_FRAME_COLOUR = (200, 200, 210)
_FRAME_THICKNESS = 3


def render_maze(
    maze: MazeGrid,
    output_path: Path,
    image_size: int = 1024,
    palette: str = "forest",
    palette_dict: Optional[dict] = None,
) -> None:
    """Render *maze* to a PNG file at *output_path*.

    The canvas is always *image_size* x *image_size* pixels.  The tile size is
    derived from the smaller of (image_size / rows) and (image_size / cols) so
    that the grid fits with a margin.

    Args:
        maze: The MazeGrid to render.
        output_path: Destination PNG file path.
        image_size: Canvas side length in pixels (default 1024).
        palette: Sprite palette name passed to generate_sprite_sheet().
                 Ignored when *palette_dict* is provided.
        palette_dict: Optional palette dictionary with the same 22-key schema
                      as the built-in palettes.  When provided, *palette* is
                      not consulted and no name-lookup validation is performed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Leave a margin so the grid is not flush with the canvas edge.
    margin_ratio = 0.06
    usable = int(image_size * (1 - 2 * margin_ratio))

    tile_size = min(usable // maze.cols, usable // maze.rows)
    tile_size = max(tile_size, 4)  # at least 4 px per tile

    sprites = generate_sprite_sheet(tile_size, palette, seed=None, palette_dict=palette_dict)

    grid_w = tile_size * maze.cols
    grid_h = tile_size * maze.rows

    # Centre the grid on the canvas.
    origin_x = (image_size - grid_w) // 2
    origin_y = (image_size - grid_h) // 2

    # Build canvas.
    canvas = Image.new("RGB", (image_size, image_size), _BG_COLOUR)

    # Paste tiles.
    for r in range(maze.rows):
        for c in range(maze.cols):
            cell_type = maze.grid[r][c]
            sprite = sprites[cell_type]
            x = origin_x + c * tile_size
            y = origin_y + r * tile_size
            canvas.paste(sprite, (x, y))

    # Draw outer border frame.
    draw = ImageDraw.Draw(canvas)
    frame_x0 = origin_x - _FRAME_THICKNESS
    frame_y0 = origin_y - _FRAME_THICKNESS
    frame_x1 = origin_x + grid_w + _FRAME_THICKNESS - 1
    frame_y1 = origin_y + grid_h + _FRAME_THICKNESS - 1
    for i in range(_FRAME_THICKNESS):
        draw.rectangle(
            [frame_x0 + i, frame_y0 + i, frame_x1 - i, frame_y1 - i],
            outline=_frame_colour_for_palette(palette),
        )

    canvas.save(str(output_path), "PNG")


def _frame_colour_for_palette(palette: str) -> tuple[int, int, int]:
    """Return a frame colour that complements the given palette."""
    frame_map: dict[str, tuple[int, int, int]] = {
        "forest":  (180, 220, 160),
        "desert":  (220, 185, 120),
        "dungeon": (100, 100, 130),
        "meadow":  (160, 210, 130),
    }
    return frame_map.get(palette, _FRAME_COLOUR)
