"""Generate a compact 1x5 maze strip for the title area."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MAZES = ROOT / "generated_mazes"
OUT = Path(__file__).resolve().parent

EXAMPLES = [
    ("gen_maze_001", "A: Diagnostic"),
    ("gen_maze_033", "C: Density 35%"),
    ("gen_maze_044", "D: Traps"),
    ("gen_maze_088", "G: Hard 13×13"),
    ("gen_maze_101", "X: Ultra 20×20"),
]

def try_font(size, bold=False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(path, size=size)
    except:
        return ImageFont.load_default()

def build():
    cell = 320
    pad = 10
    label_h = 40
    outer = 10
    cols = len(EXAMPLES)

    w = outer * 2 + cols * cell + (cols - 1) * pad
    h = outer + cell + label_h + outer

    canvas = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    label_font = try_font(20, bold=True)

    for idx, (name, label) in enumerate(EXAMPLES):
        x = outer + idx * (cell + pad)
        y = outer
        img_path = MAZES / f"{name}.png"
        if img_path.exists():
            maze = Image.open(img_path).convert("RGB")
            maze = maze.resize((cell, cell), Image.Resampling.LANCZOS)
            draw.rectangle((x-1, y-1, x+cell+1, y+cell+1), outline="#cccccc", width=1)
            canvas.paste(maze, (x, y))
        draw.text((x + cell//2, y + cell + 4), label, fill="#444444", font=label_font, anchor="mt")

    out_path = OUT / "fig_maze_strip.png"
    canvas.save(out_path, optimize=True)
    print(f"Saved {out_path} ({canvas.size[0]}x{canvas.size[1]})")

if __name__ == "__main__":
    build()
