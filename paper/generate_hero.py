"""Generate hero figure: maze grid (top) + scatter plot (bottom) stacked vertically."""

from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parent


def build():
    top = Image.open(OUT / "fig_maze_grid.png").convert("RGB")
    bottom = Image.open(OUT / "fig_efficiency_scatter.png").convert("RGB")

    # Unify width so both panels line up
    target_w = 1600

    top_ratio = target_w / top.width
    top = top.resize(
        (target_w, int(top.height * top_ratio)),
        Image.Resampling.LANCZOS,
    )

    bottom_ratio = target_w / bottom.width
    bottom = bottom.resize(
        (target_w, int(bottom.height * bottom_ratio)),
        Image.Resampling.LANCZOS,
    )

    gap = 40
    total_h = top.height + gap + bottom.height
    canvas = Image.new("RGB", (target_w, total_h), "#ffffff")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height + gap))

    out_path = OUT / "fig_hero.png"
    canvas.save(out_path, optimize=True)
    print(f"Saved {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build()
