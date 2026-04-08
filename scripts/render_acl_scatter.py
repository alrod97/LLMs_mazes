from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT_PATH = ASSETS / "acl_workshop_reasoning_token_efficiency_scatter.png"

FIG_WIDTH = 3.3
FIG_HEIGHT = 5.1
DPI = 300

PROVIDER_STYLES = {
    "OpenAI": {"color": "#2563eb"},
    "Google": {"color": "#ea580c"},
    "Qwen": {"color": "#059669"},
    "Anthropic": {"color": "#dc2626"},
}

EFFORT_SIZES = {
    "none": 38,
    "default": 70,
    "medium": 108,
}

DATA = [
    {"id": 1, "label": "GPT-5.4 (medium)", "solved": 91, "tokens_k": 265, "provider": "OpenAI"},
    {"id": 2, "label": "GPT-5.4 (low)", "solved": 85, "tokens_k": 145, "provider": "OpenAI"},
    {"id": 3, "label": "GPT-5.4 (none)", "solved": 12, "tokens_k": 9, "provider": "OpenAI"},
    {"id": 4, "label": "Gemini 3.1 Pro", "solved": 79, "tokens_k": 731, "provider": "Google"},
    {"id": 5, "label": "Gemini 3 Flash", "solved": 53, "tokens_k": 804, "provider": "Google"},
    {"id": 6, "label": "Qwen 3.5 Flash", "solved": 15, "tokens_k": 238, "provider": "Qwen"},
    {"id": 7, "label": "Qwen 3.5 Plus", "solved": 11, "tokens_k": 240, "provider": "Qwen"},
    {"id": 8, "label": "Sonnet 4.6 (none)", "solved": 6, "tokens_k": 86, "provider": "Anthropic"},
    {"id": 9, "label": "Opus 4.6 (low)", "solved": 4, "tokens_k": 114, "provider": "Anthropic"},
    {"id": 10, "label": "Opus 4.6 (none)", "solved": 4, "tokens_k": 91, "provider": "Anthropic"},
    {"id": 11, "label": "Haiku 4.5 (low)", "solved": 3, "tokens_k": 91, "provider": "Anthropic"},
    {"id": 12, "label": "Sonnet 4.6 (low)", "solved": 2, "tokens_k": 61, "provider": "Anthropic"},
    {"id": 13, "label": "Haiku 4.5 (none)", "solved": 2, "tokens_k": 51, "provider": "Anthropic"},
]

POINT_ID_OFFSETS = {
    1: (8, 1),
    2: (-7, -6),
    3: (8, 10),
    4: (-9, 0),
    5: (-8, 0),
    6: (8, 8),
    7: (8, -8),
    8: (0, 10),
    9: (10, 8),
    10: (0, -10),
    11: (-10, 7),
    12: (-12, 0),
    13: (-10, -9),
}


def fmt_tokens(value: float, _: float) -> str:
    if value >= 1000:
        return f"{value/1000:.1f}M"
    return f"{int(value)}K"


def effort_tier(label: str) -> str:
    lowered = label.lower()
    if "(none)" in lowered:
        return "none"
    if "(medium)" in lowered or "(high)" in lowered:
        return "medium"
    return "default"


def build_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.7,
        }
    )

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4.5, 1.55], hspace=0.1)
    ax = fig.add_subplot(gs[0])
    key_ax = fig.add_subplot(gs[1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key_ax.set_facecolor("white")

    ax.axhspan(0, 20, color="#fca5a5", alpha=0.18, zorder=0)
    ax.text(
        0.02,
        6.2,
        "no-reasoning failure zone",
        transform=ax.get_yaxis_transform(),
        color="#b91c1c",
        fontsize=5.7,
        fontstyle="italic",
        ha="left",
        va="center",
    )

    for provider, style in PROVIDER_STYLES.items():
        subset = [point for point in DATA if point["provider"] == provider]
        for point in subset:
            ax.scatter(
                [point["tokens_k"]],
                [point["solved"]],
                s=EFFORT_SIZES[effort_tier(point["label"])],
                marker="o",
                c=style["color"],
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )

    gpt_curve = [point for point in DATA if point["label"].startswith("GPT-5.4")]
    gpt_curve.sort(key=lambda point: point["tokens_k"])
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    ax.plot(
        [point["tokens_k"] for point in gpt_curve],
        [point["solved"] for point in gpt_curve],
        color=PROVIDER_STYLES["OpenAI"]["color"],
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        alpha=0.95,
        zorder=2,
    )
    ax.text(
        385,
        89.0,
        "reasoning effort scaling",
        color=PROVIDER_STYLES["OpenAI"]["color"],
        fontsize=5.4,
        fontstyle="italic",
        ha="left",
        va="bottom",
        path_effects=halo,
    )

    for point in DATA:
        dx, dy = POINT_ID_OFFSETS[point["id"]]
        ax.annotate(
            str(point["id"]),
            (point["tokens_k"], point["solved"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color="#111827",
            zorder=4,
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": "white",
                "alpha": 0.96,
            },
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=5.8,
            label=provider,
        )
        for provider, style in PROVIDER_STYLES.items()
    ]
    provider_legend = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.897),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d1d5db",
        borderpad=0.35,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.35,
        fontsize=5.0,
    )
    provider_legend.set_title("Provider", prop={"size": 5.0, "weight": "bold"})

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6b7280",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=4.8,
            label="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6b7280",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=6.8,
            label="low / default",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6b7280",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.6,
            label="medium+",
        ),
    ]
    size_legend = fig.legend(
        handles=size_handles,
        loc="upper left",
        bbox_to_anchor=(0.16, 0.897),
        ncol=3,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d1d5db",
        borderpad=0.35,
        handletextpad=0.45,
        columnspacing=0.8,
        labelspacing=0.35,
        fontsize=4.9,
    )
    size_legend.set_title("Reasoning budget", prop={"size": 5.0, "weight": "bold"})

    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.axis("off")
    key_ax.text(
        0.0,
        0.98,
        "Model key",
        ha="left",
        va="top",
        fontsize=5.8,
        fontweight="bold",
        color="#111827",
    )

    left_column = DATA[:7]
    right_column = DATA[7:]
    x_positions = [0.02, 0.52]
    columns = [left_column, right_column]
    row_step = 0.12
    start_y = 0.82

    for x0, items in zip(x_positions, columns):
        for idx, point in enumerate(items):
            y = start_y - idx * row_step
            style = PROVIDER_STYLES[point["provider"]]
            key_ax.scatter(
                [x0],
                [y],
                s=EFFORT_SIZES[effort_tier(point["label"])],
                marker="o",
                c=style["color"],
                edgecolors="white",
                linewidths=0.8,
                clip_on=False,
            )
            key_ax.text(
                x0 + 0.045,
                y,
                f"{point['id']}. {point['label']}",
                ha="left",
                va="center",
                fontsize=4.9,
                color="#111827",
            )

    ax.set_xscale("log")
    ax.set_xlim(8, 900)
    ax.set_ylim(0, 100)
    ax.set_xticks([10, 20, 50, 100, 200, 500, 800])
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlabel("Total Tokens Consumed (thinking + output)", labelpad=2)
    ax.set_ylabel("Exact Solve Rate (%)", labelpad=4)

    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.65)
    ax.tick_params(axis="both", which="both", length=2.2, width=0.6, color="#6b7280")
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#9ca3af")

    fig.text(
        0.125,
        0.974,
        "Reasoning Token Efficiency Across Frontier Multimodal Models",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.125,
        0.944,
        "Each point = one model config on 100 mazes. Lower-right = more efficient.",
        ha="left",
        va="top",
        fontsize=5.9,
        color="#4b5563",
    )

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.06, top=0.82)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build_plot()
    print(OUT_PATH)
