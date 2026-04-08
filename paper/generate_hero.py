"""Generate hero figure: maze grid (left) + scatter plot (right) side by side."""

from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parent


def build():
    left = Image.open(OUT / "fig_maze_grid.png").convert("RGB")
    right = Image.open(OUT / "fig_efficiency_scatter.png").convert("RGB")

    # Both panels same height, compact
    target_h = 580

    # Scale maze grid to target height
    left_ratio = target_h / left.height
    left = left.resize(
        (int(left.width * left_ratio), target_h),
        Image.Resampling.LANCZOS,
    )

    # Scale scatter to target height
    right_ratio = target_h / right.height
    right = right.resize(
        (int(right.width * right_ratio), target_h),
        Image.Resampling.LANCZOS,
    )

    gap = 30
    total_w = left.width + gap + right.width
    canvas = Image.new("RGB", (total_w, target_h), "#ffffff")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))

    out_path = OUT / "fig_hero.png"
    canvas.save(out_path, optimize=True)
    print(f"Saved {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build()
