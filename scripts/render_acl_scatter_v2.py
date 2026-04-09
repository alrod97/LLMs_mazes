"""Improved reasoning token efficiency scatter plot for the paper/blog."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT_PATH = ASSETS / "acl_workshop_reasoning_token_efficiency_scatter_v2.png"

# ── colours ──────────────────────────────────────────────────────────
PROVIDER_COLORS = {
    "OpenAI":    "#2563eb",
    "Google":    "#ea580c",
    "Qwen":      "#059669",
    "Anthropic": "#dc2626",
}

# ── reasoning-effort → bubble area ──────────────────────────────────
EFFORT_AREA = {
    "none":    30,
    "low":     65,
    "default": 65,
    "medium":  120,
    "high":    160,
}

# ── data ─────────────────────────────────────────────────────────────
DATA = [
    {"id":  1, "label": "GPT-5.4",         "effort": "medium",  "solved": 91, "tokens_k": 265, "provider": "OpenAI"},
    {"id":  2, "label": "GPT-5.4",         "effort": "low",     "solved": 85, "tokens_k": 145, "provider": "OpenAI"},
    {"id":  3, "label": "GPT-5.4",         "effort": "none",    "solved": 12, "tokens_k":   9, "provider": "OpenAI"},
    {"id":  4, "label": "Gemini 3.1 Pro",  "effort": "default", "solved": 79, "tokens_k": 731, "provider": "Google"},
    {"id":  5, "label": "Gemini 3 Flash",  "effort": "default", "solved": 53, "tokens_k": 804, "provider": "Google"},
    {"id":  6, "label": "Qwen 3.5 Flash",  "effort": "default", "solved": 15, "tokens_k": 238, "provider": "Qwen"},
    {"id":  7, "label": "Qwen 3.5 Plus",   "effort": "default", "solved": 11, "tokens_k": 240, "provider": "Qwen"},
    {"id":  8, "label": "Sonnet 4.6",      "effort": "none",    "solved":  6, "tokens_k":  86, "provider": "Anthropic"},
    {"id":  9, "label": "Opus 4.6",        "effort": "low",     "solved":  4, "tokens_k": 114, "provider": "Anthropic"},
    {"id": 10, "label": "Opus 4.6",        "effort": "none",    "solved":  4, "tokens_k":  91, "provider": "Anthropic"},
    {"id": 11, "label": "Haiku 4.5",       "effort": "low",     "solved":  3, "tokens_k":  91, "provider": "Anthropic"},
    {"id": 12, "label": "Sonnet 4.6",      "effort": "low",     "solved":  2, "tokens_k":  61, "provider": "Anthropic"},
    {"id": 13, "label": "Haiku 4.5",       "effort": "none",    "solved":  2, "tokens_k":  51, "provider": "Anthropic"},
]

# ── numbered-label offsets (dx, dy in points) ───────────────────────
OFFSETS = {
    1:  ( 12,  -4),
    2:  (-14,  -4),
    3:  ( 10,   8),
    4:  (-14,   6),
    5:  (-14,  -8),
    6:  ( 12,   6),
    7:  ( 12,  -8),
    8:  ( 12,   8),
    9:  ( 12,   8),
    10: ( 12,  -4),
    11: (-14,   8),
    12: (-14,   4),
    13: (-14,  -6),
}


def fmt_tokens(value: float, _: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.0f}M"
    return f"{int(value)}K"


def build_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # ── no-reasoning zone (subtle) ──────────────────────────────────
    ax.axhline(y=15, color="#d4d4d4", linewidth=0.8, linestyle="--", zorder=1)
    ax.fill_between([1, 2000], 0, 15, color="#fef2f2", alpha=0.5, zorder=0)
    ax.text(
        0.98, 7.5,
        "no-reasoning baseline (< 15%)",
        transform=ax.get_yaxis_transform(),
        color="#a3a3a3", fontsize=6.5, fontstyle="italic",
        ha="right", va="center",
    )

    # ── GPT-5.4 scaling curve ───────────────────────────────────────
    gpt = sorted(
        [p for p in DATA if p["label"] == "GPT-5.4"],
        key=lambda p: p["tokens_k"],
    )
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    ax.plot(
        [p["tokens_k"] for p in gpt],
        [p["solved"] for p in gpt],
        color=PROVIDER_COLORS["OpenAI"],
        linestyle="--", linewidth=1.4, alpha=0.6, zorder=2,
        path_effects=halo,
    )
    # curve label
    ax.text(
        310, 92,
        "GPT-5.4 effort\nscaling curve",
        color=PROVIDER_COLORS["OpenAI"], fontsize=6, fontstyle="italic",
        ha="left", va="bottom", path_effects=halo, linespacing=1.2,
    )

    # ── Pareto frontier shading ─────────────────────────────────────
    # points on the Pareto front: GPT-5.4 none → low → medium
    pareto_x = [p["tokens_k"] for p in gpt]
    pareto_y = [p["solved"] for p in gpt]
    # extend to axes for shading
    ax.fill_betweenx(
        [0, pareto_y[0], pareto_y[1], pareto_y[2], 100],
        [pareto_x[0], pareto_x[0], pareto_x[1], pareto_x[2], pareto_x[2]],
        [2000, 2000, 2000, 2000, 2000],
        color=PROVIDER_COLORS["OpenAI"], alpha=0.03, zorder=0,
    )

    # ── scatter all points ──────────────────────────────────────────
    for point in DATA:
        color = PROVIDER_COLORS[point["provider"]]
        area = EFFORT_AREA[point["effort"]]
        ax.scatter(
            point["tokens_k"], point["solved"],
            s=area, marker="o",
            c=color, edgecolors="white", linewidths=0.9,
            zorder=4,
        )

    # ── numbered ID labels ──────────────────────────────────────────
    for point in DATA:
        dx, dy = OFFSETS[point["id"]]
        ax.annotate(
            str(point["id"]),
            (point["tokens_k"], point["solved"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=6, fontweight="bold", color="#374151",
            ha="center", va="center", zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white", edgecolor="#d1d5db",
                linewidth=0.5, alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#9ca3af",
                linewidth=0.5,
                shrinkA=0, shrinkB=3,
            ),
        )

    # (Claude cluster annotation removed)

    # ── axes ────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(6, 1100)
    ax.set_ylim(-2, 102)
    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))

    ax.set_xlabel("Total Tokens Consumed (thinking + output)", labelpad=6)
    ax.set_ylabel("Exact Solve Rate", labelpad=6)

    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", which="both", length=3, width=0.6, color="#9ca3af")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#d1d5db")

    # ── title + subtitle ────────────────────────────────────────────
    ax.set_title(
        "Reasoning Token Efficiency Across Frontier Models",
        fontsize=11, fontweight="bold", color="#111827",
        pad=28, loc="center",
    )
    fig.text(
        0.535, 0.92,
        "Each point = one model configuration evaluated on 110 mazes.  Bubble size = reasoning effort.  Top-left = Pareto-optimal.",
        fontsize=7, color="#6b7280", ha="center", va="top",
    )

    # ── provider legend ─────────────────────────────────────────────
    provider_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=c, markeredgecolor="white",
               markeredgewidth=0.8, markersize=7, label=name)
        for name, c in PROVIDER_COLORS.items()
    ]
    leg1 = ax.legend(
        handles=provider_handles, loc="upper left",
        frameon=True, framealpha=0.95, facecolor="white",
        edgecolor="#e5e7eb", fontsize=6.5,
        borderpad=0.4, handletextpad=0.4, labelspacing=0.35,
        title="Provider", title_fontproperties={"size": 7, "weight": "bold"},
    )
    ax.add_artist(leg1)

    # ── effort legend ───────────────────────────────────────────────
    effort_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#9ca3af", markeredgecolor="white",
               markeredgewidth=0.8, markersize=sz, label=name)
        for name, sz in [("none", 4), ("low / default", 5.8), ("medium+", 7.8)]
    ]
    leg2 = ax.legend(
        handles=effort_handles, loc="upper center",
        bbox_to_anchor=(0.38, 1.0),
        frameon=True, framealpha=0.95, facecolor="white",
        edgecolor="#e5e7eb", fontsize=6.5, ncol=3,
        borderpad=0.4, handletextpad=0.3, columnspacing=1.0,
        title="Reasoning effort", title_fontproperties={"size": 7, "weight": "bold"},
    )
    ax.add_artist(leg1)  # re-add so both show

    # ── model key (below plot) ──────────────────────────────────────
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.30, top=0.90)

    key_lines = []
    for p in DATA:
        eff = p["effort"]
        eff_str = f" ({eff})" if eff != "default" else ""
        key_lines.append(f"{p['id']:>2}. {p['label']}{eff_str}")

    col1 = "\n".join(key_lines[:5])
    col2 = "\n".join(key_lines[5:9])
    col3 = "\n".join(key_lines[9:])

    kw = dict(fontsize=8, fontfamily="monospace", color="#374151",
              va="top", linespacing=1.45)
    fig.text(0.10,  0.14, col1, **kw)
    fig.text(0.40, 0.14, col2, **kw)
    fig.text(0.72, 0.14, col3, **kw)
    fig.text(0.10, 0.16, "Model key", fontsize=9, fontweight="bold",
             color="#374151", va="bottom")

    # thin separator line between plot and key
    fig.patches.append(plt.Rectangle(
        (0.08, 0.195), 0.87, 0.0015,
        transform=fig.transFigure, facecolor="#e5e7eb", zorder=10,
    ))
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build_plot()
