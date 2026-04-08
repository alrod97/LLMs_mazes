"""Programmatic pixel-art sprite generation using PIL.

Each tile is a square image of `tile_size` pixels.  Four palettes are
supported: forest, desert, dungeon, meadow.

Colour conventions per palette
--------------------------------
forest  : green grass, dark-grey stone, bright-red traps
desert  : sandy tan, warm brown stone, orange traps
dungeon : near-black slate, charcoal stone, blood-red traps
meadow  : light sage, mossy stone, yellow-orange traps
"""

from __future__ import annotations

import random
from typing import Optional

from PIL import Image, ImageDraw

from .maze_model import CellType

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------

Colour = tuple[int, int, int]

_PALETTES: dict[str, dict[str, Colour]] = {
    "forest": {
        "floor_base":    (72, 140,  60),
        "floor_dot":     (55, 115,  45),
        "floor_bright":  (90, 165,  75),
        "floor_border":  (45,  95,  35),
        "wall_base":     (70,  70,  75),
        "wall_dark":     (45,  45,  50),
        "wall_light":    (95,  95, 100),
        "wall_crack":    (30,  30,  35),
        "trap_base":     (200,  40,  30),
        "trap_flame1":   (240, 120,  30),
        "trap_flame2":   (255, 200,  50),
        "trap_spike":    (160,  20,  10),
        "start_body":    (240, 220, 180),
        "start_shirt":   ( 60, 100, 180),
        "start_sword":   (200, 200, 220),
        "goal_chest":    (120,  70,  20),
        "goal_lid":      (150,  90,  30),
        "goal_clasp":    (220, 180,  30),
        "goal_glow":     (255, 230, 100),
        "label_fg":      (255, 255, 255),
        "label_bg":      ( 20,  60,  10),
        "bg":            ( 25,  40,  20),
    },
    "desert": {
        "floor_base":    (210, 180, 120),
        "floor_dot":     (185, 155, 100),
        "floor_bright":  (230, 205, 150),
        "floor_border":  (170, 135,  85),
        "wall_base":     (140, 100,  65),
        "wall_dark":     (110,  75,  45),
        "wall_light":    (175, 135,  90),
        "wall_crack":    ( 90,  55,  30),
        "trap_base":     (215,  95,  20),
        "trap_flame1":   (240, 155,  40),
        "trap_flame2":   (255, 215,  60),
        "trap_spike":    (180,  60,  10),
        "start_body":    (240, 220, 185),
        "start_shirt":   (180, 120,  40),
        "start_sword":   (215, 205, 175),
        "goal_chest":    (130,  80,  25),
        "goal_lid":      (165, 105,  40),
        "goal_clasp":    (230, 190,  40),
        "goal_glow":     (255, 235, 120),
        "label_fg":      (255, 255, 255),
        "label_bg":      (100,  65,  20),
        "bg":            ( 80,  55,  25),
    },
    "dungeon": {
        "floor_base":    ( 40,  42,  48),
        "floor_dot":     ( 30,  32,  38),
        "floor_bright":  ( 55,  58,  65),
        "floor_border":  ( 22,  24,  30),
        "wall_base":     ( 28,  28,  32),
        "wall_dark":     ( 15,  15,  18),
        "wall_light":    ( 50,  50,  58),
        "wall_crack":    (  8,   8,  10),
        "trap_base":     (160,  20,  20),
        "trap_flame1":   (200,  60,  30),
        "trap_flame2":   (240, 120,  40),
        "trap_spike":    (120,  10,  10),
        "start_body":    (200, 180, 145),
        "start_shirt":   ( 80,  40,  80),
        "start_sword":   (180, 180, 200),
        "goal_chest":    ( 90,  55,  15),
        "goal_lid":      (115,  70,  22),
        "goal_clasp":    (200, 160,  20),
        "goal_glow":     (255, 210,  80),
        "label_fg":      (220, 220, 255),
        "label_bg":      ( 10,  10,  20),
        "bg":            (  8,   8,  14),
    },
    "meadow": {
        "floor_base":    (130, 185, 100),
        "floor_dot":     (105, 160,  80),
        "floor_bright":  (155, 210, 125),
        "floor_border":  ( 90, 140,  65),
        "wall_base":     ( 90, 105,  70),
        "wall_dark":     ( 65,  80,  50),
        "wall_light":    (120, 140,  95),
        "wall_crack":    ( 50,  60,  35),
        "trap_base":     (215, 120,  20),
        "trap_flame1":   (240, 175,  50),
        "trap_flame2":   (255, 230,  80),
        "trap_spike":    (170,  80,  10),
        "start_body":    (240, 220, 180),
        "start_shirt":   ( 80, 160,  80),
        "start_sword":   (200, 210, 195),
        "goal_chest":    (110,  70,  20),
        "goal_lid":      (140,  90,  32),
        "goal_clasp":    (225, 185,  35),
        "goal_glow":     (255, 235, 110),
        "label_fg":      (255, 255, 255),
        "label_bg":      ( 40,  75,  20),
        "bg":            ( 30,  60,  15),
    },
}


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

def _fill(img: Image.Image, colour: Colour) -> None:
    img.paste(colour, [0, 0, img.width, img.height])


def _pixel(draw: ImageDraw.ImageDraw, x: int, y: int, colour: Colour) -> None:
    draw.point((x, y), fill=colour)


def _rect(
    draw: ImageDraw.ImageDraw,
    x0: int, y0: int, x1: int, y1: int,
    colour: Colour,
    outline: Optional[Colour] = None,
) -> None:
    draw.rectangle([x0, y0, x1, y1], fill=colour, outline=outline)


def _triangle(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[int, int]],
    colour: Colour,
) -> None:
    draw.polygon(pts, fill=colour)


# ---------------------------------------------------------------------------
# Tile generators
# ---------------------------------------------------------------------------

def _floor_tile(size: int, p: dict[str, Colour], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), p["floor_base"])
    draw = ImageDraw.Draw(img)

    # Subtle brightness variation: a few brighter patches
    patch_count = max(2, size // 8)
    for _ in range(patch_count):
        px = rng.randint(0, size - 1)
        py = rng.randint(0, size - 1)
        bright = rng.random() < 0.5
        col = p["floor_bright"] if bright else p["floor_dot"]
        draw.point((px, py), fill=col)

    # Random grass-texture dots
    dot_count = max(4, size // 4)
    for _ in range(dot_count):
        px = rng.randint(1, size - 2)
        py = rng.randint(1, size - 2)
        col = p["floor_dot"] if rng.random() < 0.6 else p["floor_bright"]
        _pixel(draw, px, py, col)

    # Faint 1-pixel border
    border_col = p["floor_border"]
    for x in range(size):
        _pixel(draw, x, 0, border_col)
        _pixel(draw, x, size - 1, border_col)
    for y in range(size):
        _pixel(draw, 0, y, border_col)
        _pixel(draw, size - 1, y, border_col)

    return img


def _wall_tile(size: int, p: dict[str, Colour], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), p["wall_base"])
    draw = ImageDraw.Draw(img)

    # Draw a brick-line pattern
    # Horizontal mortar lines at 1/3 and 2/3 height
    mortar_h = max(1, size // 12)
    third = size // 3
    two_third = 2 * size // 3

    for y_line in (third, two_third):
        _rect(draw, 0, y_line, size - 1, y_line + mortar_h - 1, p["wall_dark"])

    # Vertical mortar lines — offset per row band
    half = size // 2
    for (y0, y1, x_off) in [
        (0, third, 0),
        (third + mortar_h, two_third, half),
        (two_third + mortar_h, size - 1, size // 4),
    ]:
        x_v = x_off % size
        if 0 < x_v < size - 1:
            _rect(draw, x_v, y0, x_v + mortar_h - 1, y1, p["wall_dark"])

    # Highlight top-left edges of each brick band for a 3-D look
    highlight_col = p["wall_light"]
    draw.line([(0, 0), (size - 1, 0)], fill=highlight_col)
    draw.line([(0, 0), (0, size - 1)], fill=highlight_col)

    # Subtle random cracks
    num_cracks = max(1, size // 20)
    for _ in range(num_cracks):
        cx = rng.randint(2, size - 3)
        cy = rng.randint(2, size - 3)
        length = rng.randint(2, max(3, size // 8))
        dx = rng.choice([-1, 0, 1])
        dy = rng.choice([-1, 0, 1])
        for step in range(length):
            px = cx + dx * step
            py = cy + dy * step
            if 0 <= px < size and 0 <= py < size:
                _pixel(draw, px, py, p["wall_crack"])

    return img


def _trap_tile(size: int, p: dict[str, Colour], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), p["trap_base"])
    draw = ImageDraw.Draw(img)

    # Hazard floor with a slightly darker centre
    centre_size = size * 2 // 3
    margin = (size - centre_size) // 2
    _rect(draw, margin, margin, margin + centre_size, margin + centre_size,
          _darken(p["trap_base"], 30))

    # Draw 3 spike triangles pointing upward, evenly spaced
    num_spikes = 3
    spike_w = max(4, size // (num_spikes + 1))
    spike_h = size * 2 // 3
    base_y = size - 2
    spacing = size // num_spikes

    for i in range(num_spikes):
        cx = spacing // 2 + i * spacing
        tip_y = base_y - spike_h + rng.randint(0, max(1, size // 10))
        # Main spike body (darker colour for depth)
        _triangle(draw, [
            (cx, tip_y),
            (cx - spike_w // 2, base_y),
            (cx + spike_w // 2, base_y),
        ], p["trap_spike"])
        # Highlight left face
        _triangle(draw, [
            (cx, tip_y),
            (cx - spike_w // 2, base_y),
            (cx - spike_w // 4, base_y - spike_h // 3),
        ], p["trap_flame1"])
        # Tiny tip highlight
        draw.point((cx, tip_y + 1), fill=p["trap_flame2"])

    # Flame glow dots near base
    for _ in range(max(2, size // 6)):
        fx = rng.randint(1, size - 2)
        fy = rng.randint(max(1, size * 2 // 3), size - 2)
        col = p["trap_flame1"] if rng.random() < 0.6 else p["trap_flame2"]
        _pixel(draw, fx, fy, col)

    # Thin border
    border_col = _darken(p["trap_base"], 50)
    for x in range(size):
        _pixel(draw, x, 0, border_col)
        _pixel(draw, x, size - 1, border_col)
    for y in range(size):
        _pixel(draw, 0, y, border_col)
        _pixel(draw, size - 1, y, border_col)

    return img


def _start_tile(size: int, p: dict[str, Colour], rng: random.Random) -> Image.Image:
    """Floor tile + a small adventurer sprite + "S" label."""
    img = _floor_tile(size, p, rng)
    draw = ImageDraw.Draw(img)

    # Character sprite proportions (scaled to tile_size)
    scale = size / 32.0  # design at 32 px

    def s(v: float) -> int:
        return max(1, int(v * scale))

    cx = size // 2

    # Legs (two small rectangles)
    leg_top = s(22)
    leg_h = s(6)
    leg_w = s(3)
    gap = s(1)
    _rect(draw, cx - leg_w - gap, leg_top, cx - gap, leg_top + leg_h, p["floor_border"])
    _rect(draw, cx + gap, leg_top, cx + leg_w + gap, leg_top + leg_h, p["floor_border"])

    # Body (torso rectangle)
    body_top = s(13)
    body_h = s(9)
    body_w = s(7)
    _rect(draw, cx - body_w, body_top, cx + body_w, body_top + body_h, p["start_shirt"])

    # Head (circle approximated as a rounded rectangle)
    head_r = s(5)
    head_cy = s(8)
    _rect(draw, cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r, p["start_body"])
    # Eyes
    eye_y = head_cy - s(1)
    _pixel(draw, cx - s(2), eye_y, (30, 20, 10))
    _pixel(draw, cx + s(2), eye_y, (30, 20, 10))

    # Sword (diagonal line from right side of body)
    sword_x0 = cx + body_w + s(1)
    sword_y0 = body_top - s(2)
    sword_x1 = sword_x0 + s(6)
    sword_y1 = sword_y0 - s(6)
    draw.line([(sword_x0, sword_y0), (sword_x1, sword_y1)], fill=p["start_sword"], width=max(1, s(1)))
    # Crossguard
    cg_cx = (sword_x0 + sword_x1) // 2
    cg_cy = (sword_y0 + sword_y1) // 2
    draw.line([(cg_cx - s(2), cg_cy + s(2)), (cg_cx + s(2), cg_cy - s(2))],
              fill=p["start_sword"], width=max(1, s(1)))

    # "S" label badge in bottom-right corner
    badge_size = max(6, size // 5)
    bx = size - badge_size - 1
    by = size - badge_size - 1
    _rect(draw, bx, by, size - 2, size - 2, p["label_bg"])
    _draw_letter(draw, "S", bx + badge_size // 2, by + badge_size // 2, badge_size, p["label_fg"])

    return img


def _goal_tile(size: int, p: dict[str, Colour], rng: random.Random) -> Image.Image:
    """Floor tile + treasure chest sprite + "G" label."""
    img = _floor_tile(size, p, rng)
    draw = ImageDraw.Draw(img)

    scale = size / 32.0

    def s(v: float) -> int:
        return max(1, int(v * scale))

    cx = size // 2
    chest_w = s(14)
    chest_h = s(9)
    lid_h = s(4)
    chest_y = size // 2 - chest_h // 2 + s(2)  # slightly below centre

    # Glow behind chest
    glow_r = chest_w + s(3)
    glow_col = _blend(p["goal_glow"], p["floor_base"], 0.35)
    draw.ellipse(
        [cx - glow_r, chest_y - s(2), cx + glow_r, chest_y + chest_h + lid_h + s(2)],
        fill=glow_col,
    )

    # Chest body
    _rect(draw, cx - chest_w, chest_y + lid_h, cx + chest_w, chest_y + chest_h + lid_h,
          p["goal_chest"], outline=_darken(p["goal_chest"], 30))

    # Chest lid (slightly wider for perspective look)
    lid_extra = s(1)
    _rect(draw, cx - chest_w - lid_extra, chest_y,
          cx + chest_w + lid_extra, chest_y + lid_h,
          p["goal_lid"], outline=_darken(p["goal_lid"], 25))

    # Gold clasp in centre
    clasp_w = s(4)
    clasp_h = s(5)
    clasp_x = cx - clasp_w // 2
    clasp_y = chest_y + lid_h - clasp_h // 2
    _rect(draw, clasp_x, clasp_y, clasp_x + clasp_w, clasp_y + clasp_h,
          p["goal_clasp"], outline=_darken(p["goal_clasp"], 40))
    # Clasp keyhole dot
    _pixel(draw, cx, clasp_y + clasp_h // 2, _darken(p["goal_clasp"], 80))

    # Corner rivets on chest body
    rivet_col = _darken(p["goal_clasp"], 20)
    for rx, ry in [
        (cx - chest_w + s(1), chest_y + lid_h + s(1)),
        (cx + chest_w - s(1), chest_y + lid_h + s(1)),
        (cx - chest_w + s(1), chest_y + chest_h + lid_h - s(1)),
        (cx + chest_w - s(1), chest_y + chest_h + lid_h - s(1)),
    ]:
        _pixel(draw, rx, ry, rivet_col)

    # Sparkle dots above chest
    for _ in range(max(2, size // 12)):
        sx = rng.randint(cx - chest_w, cx + chest_w)
        sy = rng.randint(max(1, chest_y - s(6)), chest_y - 1)
        col = p["goal_glow"] if rng.random() < 0.5 else p["goal_clasp"]
        _pixel(draw, sx, sy, col)

    # "G" label badge in bottom-right corner
    badge_size = max(6, size // 5)
    bx = size - badge_size - 1
    by = size - badge_size - 1
    _rect(draw, bx, by, size - 2, size - 2, p["label_bg"])
    _draw_letter(draw, "G", bx + badge_size // 2, by + badge_size // 2, badge_size, p["label_fg"])

    return img


# ---------------------------------------------------------------------------
# Letter drawing (pixel-art style, no font dependency)
# ---------------------------------------------------------------------------

# 5x7 pixel-art bitmaps for S and G.
# Each row is a bitmask (MSB = leftmost pixel).
_GLYPHS: dict[str, list[int]] = {
    "S": [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
}
_GLYPH_W = 5
_GLYPH_H = 7


def _draw_letter(
    draw: ImageDraw.ImageDraw,
    letter: str,
    cx: int,
    cy: int,
    badge_size: int,
    colour: Colour,
) -> None:
    """Draw a pixel-art letter centred at (cx, cy), scaled to fit badge_size."""
    bitmap = _GLYPHS.get(letter)
    if bitmap is None:
        return
    # Scale each pixel
    scale = max(1, badge_size // max(_GLYPH_W, _GLYPH_H) - 1)
    glyph_px_w = _GLYPH_W * scale
    glyph_px_h = _GLYPH_H * scale
    ox = cx - glyph_px_w // 2
    oy = cy - glyph_px_h // 2
    for row_idx, row_bits in enumerate(bitmap):
        for col_idx in range(_GLYPH_W):
            if row_bits & (1 << (_GLYPH_W - 1 - col_idx)):
                px = ox + col_idx * scale
                py = oy + row_idx * scale
                draw.rectangle([px, py, px + scale - 1, py + scale - 1], fill=colour)


# ---------------------------------------------------------------------------
# Colour math utilities
# ---------------------------------------------------------------------------

def _darken(c: Colour, amount: int) -> Colour:
    return (max(0, c[0] - amount), max(0, c[1] - amount), max(0, c[2] - amount))


def _lighten(c: Colour, amount: int) -> Colour:
    return (min(255, c[0] + amount), min(255, c[1] + amount), min(255, c[2] + amount))


def _blend(c1: Colour, c2: Colour, t: float) -> Colour:
    """Linear blend: result = c1 * t + c2 * (1 - t)."""
    return (
        int(c1[0] * t + c2[0] * (1 - t)),
        int(c1[1] * t + c2[1] * (1 - t)),
        int(c1[2] * t + c2[2] * (1 - t)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sprite_sheet(
    tile_size: int,
    palette: str = "forest",
    seed: Optional[int] = 42,
    palette_dict: Optional[dict] = None,
) -> dict[CellType, Image.Image]:
    """Return a dict mapping each CellType to a PIL Image of size tile_size x tile_size.

    Args:
        tile_size: Side length in pixels for each tile.
        palette: Named palette to look up from the built-in registry.  Ignored
                 when *palette_dict* is provided.
        seed: RNG seed for procedural tile variation.
        palette_dict: Optional palette dictionary with the same 22-key schema
                      as the built-in palettes.  When provided, *palette* is
                      not consulted and no name-lookup validation is performed.
    """
    if palette_dict is not None:
        p = palette_dict
    else:
        if palette not in _PALETTES:
            raise ValueError(f"Unknown palette '{palette}'. Choose from: {list(_PALETTES)}")
        p = _PALETTES[palette]
    rng = random.Random(seed)

    return {
        CellType.FLOOR: _floor_tile(tile_size, p, rng),
        CellType.WALL:  _wall_tile(tile_size, p, rng),
        CellType.TRAP:  _trap_tile(tile_size, p, rng),
        CellType.START: _start_tile(tile_size, p, rng),
        CellType.GOAL:  _goal_tile(tile_size, p, rng),
    }
