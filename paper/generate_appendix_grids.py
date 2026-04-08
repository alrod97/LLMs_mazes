"""Generate appendix pages: one grid per group showing all mazes."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MAZES = ROOT / "generated_mazes"
OUT = Path(__file__).resolve().parent

GROUPS = [
    ("A: Diagnostic", [f"gen_maze_{i:03d}" for i in range(1, 9)]),
    ("B: Grid Scale", [f"gen_maze_{i:03d}" for i in range(9, 24)]),
    ("C: Wall Density", [f"gen_maze_{i:03d}" for i in range(24, 39)]),
    ("D: Trap Ablation", [f"gen_maze_{i:03d}" for i in range(39, 51)]),
    ("E: Unreachable", [f"gen_maze_{i:03d}" for i in range(51, 65)]),
    ("F: Border Walls", [f"gen_maze_{i:03d}" for i in range(65, 75)]),
    ("G: Combined Hard", [f"gen_maze_{i:03d}" for i in range(75, 91)]),
    ("H: Palette Stress", [f"gen_maze_{i:03d}" for i in range(91, 101)]),
    ("X: Ultra-Hard", [f"gen_maze_{i:03d}" for i in range(101, 111)]),
]


def try_font(size, bold=False):
    path = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    try:
        return ImageFont.truetype(path, size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def build_group_grid(group_name, maze_names, max_cols=5):
    n = len(maze_names)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    cell = 280
    pad = 10
    label_h = 36
    title_h = 60
    outer = 20

    w = outer * 2 + cols * cell + (cols - 1) * pad
    h = outer + title_h + rows * (cell + label_h) + (rows - 1) * pad + outer

    canvas = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = try_font(28, bold=True)
    label_font = try_font(16)

    # Title
    draw.text((outer, outer), group_name, fill="#222222", font=title_font)

    for idx, name in enumerate(maze_names):
        r, c = divmod(idx, cols)
        x = outer + c * (cell + pad)
        y = outer + title_h + r * (cell + label_h + pad)

        img_path = MAZES / f"{name}.png"
        if img_path.exists():
            maze = Image.open(img_path).convert("RGB")
            maze = maze.resize((cell, cell), Image.Resampling.LANCZOS)
            draw.rectangle(
                (x - 1, y - 1, x + cell + 1, y + cell + 1),
                outline="#dddddd", width=1,
            )
            canvas.paste(maze, (x, y))

        draw.text((x + 2, y + cell + 2), name, fill="#555555", font=label_font)

    return canvas


def main():
    all_pages = []
    for group_name, maze_names in GROUPS:
        page = build_group_grid(group_name, maze_names)
        all_pages.append((group_name, page))

    # Save individual group images
    for group_name, page in all_pages:
        safe = group_name.split(":")[0].strip().lower()
        out_path = OUT / f"appendix_group_{safe}.png"
        page.save(out_path, optimize=True)
        print(f"Saved {out_path} ({page.size[0]}x{page.size[1]})")


if __name__ == "__main__":
    main()
