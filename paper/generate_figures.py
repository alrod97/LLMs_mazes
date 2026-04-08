"""Generate paper figures: maze grid + efficiency scatter plot."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAZES = ROOT / "generated_mazes"
OUT = Path(__file__).resolve().parent

# ── Figure 1: 2x5 maze grid ──────────────────────────────────────

EXAMPLES = [
    ("gen_maze_001", "A: Diagnostic\n5×5, empty"),
    ("gen_maze_014", "B: Grid Scale\n8×8, 25% walls"),
    ("gen_maze_033", "C: Wall Density\n9×9, 35% walls"),
    ("gen_maze_044", "D: Trap Ablation\n9×9, 4 traps"),
    ("gen_maze_057", "E: Unreachable\n9×9, blocked"),
    ("gen_maze_068", "F: Border Walls\n9×9, bordered"),
    ("gen_maze_088", "G: Combined Hard\n13×13, 5 traps"),
    ("gen_maze_093", "H: Palette (dungeon)\n8×8, 2 traps"),
    ("gen_maze_101", "X: Ultra-Hard\n20×20, 12 traps"),
    ("gen_maze_109", "X: Ultra-Hard\n20×20, 18 traps"),
]


def try_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def build_maze_grid():
    cols, rows = 5, 2
    cell_img = 360
    pad = 12
    label_h = 56
    outer_pad = 30

    total_w = outer_pad * 2 + cols * cell_img + (cols - 1) * pad
    total_h = outer_pad * 2 + rows * (cell_img + label_h) + (rows - 1) * pad

    canvas = Image.new("RGB", (total_w, total_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    label_font = try_font(18, bold=True)
    sub_font = try_font(15)

    for idx, (name, label) in enumerate(EXAMPLES):
        r, c = divmod(idx, cols)
        x = outer_pad + c * (cell_img + pad)
        y = outer_pad + r * (cell_img + label_h + pad)

        img_path = MAZES / f"{name}.png"
        if not img_path.exists():
            continue
        maze = Image.open(img_path).convert("RGB")
        maze = maze.resize((cell_img, cell_img), Image.Resampling.LANCZOS)

        draw.rounded_rectangle(
            (x - 2, y - 2, x + cell_img + 2, y + cell_img + 2),
            radius=6, outline="#cccccc", width=2,
        )
        canvas.paste(maze, (x, y))

        lines = label.split("\n")
        draw.text((x + 4, y + cell_img + 4), lines[0], fill="#222222", font=label_font)
        if len(lines) > 1:
            draw.text((x + 4, y + cell_img + 24), lines[1], fill="#666666", font=sub_font)

    out_path = OUT / "fig_maze_grid.png"
    canvas.save(out_path, optimize=True)
    print(f"Saved {out_path} ({canvas.size[0]}×{canvas.size[1]})")


# ── Figure 2: Scatter plot ────────────────────────────────────────

MODELS = [
    # label,             display_name,         solved%, tok_K, provider,   reasoning
    ("GPT-5.4 (medium)", "GPT-5.4",            91, 265,  "OpenAI",    "medium"),
    ("GPT-5.4 (low)",    "GPT-5.4",            85, 145,  "OpenAI",    "low"),
    ("GPT-5.4 (none)",   "GPT-5.4",            12,   9,  "OpenAI",    "none"),
    ("Gemini 3.1 Pro",   "Gemini 3.1 Pro",     79, 731,  "Google",    "hidden"),
    ("Gemini 3 Flash",   "Gemini 3 Flash",     53, 804,  "Google",    "hidden"),
    ("Qwen 3.5 Flash",   "Qwen 3.5 Flash",    15, 238,  "Qwen",      "low"),
    ("Qwen 3.5 Plus",    "Qwen 3.5 Plus",     11, 240,  "Qwen",      "low"),
    ("Sonnet 4.6 (none)","Sonnet 4.6",          6,  86,  "Anthropic", "none"),
    ("Opus 4.6 (low)",   "Opus 4.6",            4, 114,  "Anthropic", "low"),
    ("Opus 4.6 (none)",  "Opus 4.6",            4,  91,  "Anthropic", "none"),
    ("Haiku 4.5 (low)",  "Haiku 4.5",           3,  91,  "Anthropic", "low"),
    ("Sonnet 4.6 (low)", "Sonnet 4.6",          2,  61,  "Anthropic", "low"),
    ("Haiku 4.5 (none)", "Haiku 4.5",           2,  51,  "Anthropic", "none"),
]

PROV_COLOR = {
    "OpenAI":    "#2563eb",
    "Google":    "#ea580c",
    "Qwen":      "#059669",
    "Anthropic": "#dc2626",
}

REASON_SIZE = {
    "none":   55,
    "low":    160,
    "medium": 360,
    "hidden": 360,   # Gemini's hidden thinking ≈ medium budget
}

# Manual label offsets  (x_mult, y_offset)
# x_mult multiplies the x coordinate (log space), y_offset adds to y
LABEL_OFFSETS = {
    "GPT-5.4 (medium)": (-0.42, 2),
    "GPT-5.4 (low)":    (-0.35, -6),
    "GPT-5.4 (none)":   (0.2, -7),
    "Gemini 3.1 Pro":   (-0.38, 3),
    "Gemini 3 Flash":   (-0.35, 4),
    "Qwen 3.5 Flash":   (0.12, 3.5),
    "Qwen 3.5 Plus":    (0.12, -7),
}


def build_scatter():
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, linestyle="-", alpha=0.25, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    ax.set_xscale("log")
    ax.set_xlim(5, 1200)
    ax.set_ylim(-4, 104)

    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x)}K")
    )
    ax.minorticks_off()
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{int(y)}%")
    )

    ax.set_xlabel("Total Tokens Consumed (thinking + output)",
                  fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Exact Solve Rate",
                  fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Reasoning Token Efficiency Across Frontier Models",
                 fontsize=13, fontweight="bold", pad=12)

    txt_fx = [pe.withStroke(linewidth=3, foreground="white")]

    # ── Pink failure zone ──
    ax.axhspan(-4, 20, color="#fee2e2", alpha=0.35, zorder=1)
    ax.text(900, 10, "no-reasoning failure zone",
            fontsize=7.5, color="#b91c1c", alpha=0.6,
            ha="right", fontstyle="italic", path_effects=txt_fx)

    # ── Plot points ──
    for full_label, disp, solved, tok_k, prov, reason in MODELS:
        ax.scatter(
            tok_k, solved,
            s=REASON_SIZE[reason],
            c=PROV_COLOR[prov],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            alpha=0.92,
        )

    # ── GPT-5.4 scaling line ──
    gpt_pts = [(9, 12), (145, 85), (265, 91)]
    ax.plot([p[0] for p in gpt_pts], [p[1] for p in gpt_pts],
            color="#2563eb", alpha=0.35, linewidth=2,
            linestyle="--", zorder=3)
    ax.annotate("reasoning effort\nscaling  →",
                xy=(38, 52), fontsize=7.5, fontstyle="italic",
                color="#2563eb", alpha=0.7, path_effects=txt_fx)

    # ── Labels for non-Anthropic models ──
    for full_label, disp, solved, tok_k, prov, reason in MODELS:
        if prov == "Anthropic":
            continue
        if full_label not in LABEL_OFFSETS:
            continue
        xm, yo = LABEL_OFFSETS[full_label]
        # In log space, multiply x by 10^xm for offset
        tx = tok_k * (10 ** xm)
        ty = solved + yo
        txt = disp
        # For GPT-5.4 add reasoning level
        if disp == "GPT-5.4":
            reason_lbl = {"none": "(none)", "low": "(low)", "medium": "(med.)"}
            txt = f"{disp} {reason_lbl.get(reason, '')}"
        ax.annotate(txt, (tok_k, solved), xytext=(tx, ty),
                    fontsize=7.5, color="#222222",
                    path_effects=txt_fx, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                    lw=0.6, shrinkA=4, shrinkB=2)
                    if abs(xm) > 0.3 else None)

    # ── Claude cluster bracket ──
    # Draw a single label for all Anthropic models
    claude_x = [m[3] for m in MODELS if m[4] == "Anthropic"]
    claude_y = [m[2] for m in MODELS if m[4] == "Anthropic"]
    cx, cy = np.mean(claude_x), np.mean(claude_y)
    ax.annotate("Claude family\n(Opus, Sonnet, Haiku)\n2–6% regardless\nof config",
                xy=(cx, cy),
                xytext=(20, 32),
                fontsize=7, color="#b91c1c",
                fontstyle="italic",
                path_effects=txt_fx,
                zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="#dc2626",
                                lw=1, shrinkA=2, shrinkB=8))

    # ── Legend ──
    # Provider legend (color)
    prov_handles = []
    for prov, color in PROV_COLOR.items():
        prov_handles.append(
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor=color, markersize=9,
                          markeredgecolor="white", markeredgewidth=0.8,
                          label=prov)
        )
    leg1 = ax.legend(handles=prov_handles, loc="upper left",
                     title="Provider", title_fontsize=8,
                     fontsize=7.5, framealpha=0.9,
                     edgecolor="#dddddd", borderpad=0.6,
                     bbox_to_anchor=(0.0, 1.0))
    leg1.get_frame().set_linewidth(0.6)
    ax.add_artist(leg1)

    # Size legend (reasoning level)
    size_handles = []
    for lbl, sz in [("None", 55), ("Low", 160), ("Med. / Hidden", 360)]:
        size_handles.append(
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor="#888888",
                          markersize=np.sqrt(sz) * 0.55,
                          markeredgecolor="white", markeredgewidth=0.8,
                          label=lbl)
        )
    leg2 = ax.legend(handles=size_handles, loc="upper left",
                     title="Reasoning", title_fontsize=8,
                     fontsize=7.5, framealpha=0.9,
                     edgecolor="#dddddd", borderpad=0.6,
                     bbox_to_anchor=(0.17, 1.0))
    leg2.get_frame().set_linewidth(0.6)
    ax.add_artist(leg2)

    # Re-add first legend (add_artist removes it from auto-legend)
    ax.add_artist(leg1)

    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#555555", labelsize=9)

    fig.tight_layout()
    out_path = OUT / "fig_efficiency_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_maze_grid()
    build_scatter()
