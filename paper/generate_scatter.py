"""Scatter plot: color=provider, size=model tier, all circles."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# label, display, solved%, tok_K, provider, tier(1-3), reasoning
MODELS = [
    ("GPT-5.4 (medium)",       "GPT-5.4 (med.)",       91, 265, "OpenAI",    3, "medium"),
    ("GPT-5.4 (low)",          "GPT-5.4 (low)",        85, 145, "OpenAI",    3, "low"),
    ("GPT-5.4 (none)",         "GPT-5.4 (none)",       12,   9, "OpenAI",    3, "none"),
    ("GPT-5.4-mini (medium)",  "mini (med.)",           51, 503, "OpenAI",    2, "medium"),
    ("GPT-5.4-mini (low)",     "mini (low)",            49, 152, "OpenAI",    2, "low"),
    ("GPT-5.4-mini (none)",    "mini (none)",            8,  11, "OpenAI",    2, "none"),
    ("Gemini 3.1 Pro",         "Gemini 3.1 Pro",       79, 731, "Google",    3, "hidden"),
    ("Gemini 3 Flash",         "Gemini 3 Flash",       53, 804, "Google",    1, "hidden"),
    ("Qwen 3.5 Plus",          "Qwen 3.5 Plus",        11, 240, "Qwen",      2, "low"),
    ("Qwen 3.5 Flash",         "Qwen 3.5 Flash",       15, 238, "Qwen",      1, "low"),
    ("Opus 4.6 (low)",         None,                    4, 114, "Anthropic", 3, "low"),
    ("Opus 4.6 (none)",        None,                    4,  91, "Anthropic", 3, "none"),
    ("Sonnet 4.6 (none)",      None,                    6,  86, "Anthropic", 2, "none"),
    ("Sonnet 4.6 (low)",       None,                    2,  61, "Anthropic", 2, "low"),
    ("Haiku 4.5 (low)",        None,                    3,  91, "Anthropic", 1, "low"),
    ("Haiku 4.5 (none)",       None,                    2,  51, "Anthropic", 1, "none"),
]

PROV_COLOR = {
    "OpenAI":    "#2563eb",
    "Google":    "#ea580c",
    "Qwen":      "#059669",
    "Anthropic": "#dc2626",
}

TIER_SIZE = {1: 70, 2: 170, 3: 350}

# (dx in log10, dy, ha, va) — carefully tuned to avoid ALL overlaps
LABEL_POS = {
    "GPT-5.4 (medium)":       ( 0.12,   3, "left",  "bottom"),
    "GPT-5.4 (low)":          (-0.35,   0, "right", "center"),
    "GPT-5.4 (none)":         ( 0.15,   4, "left",  "bottom"),
    "GPT-5.4-mini (medium)":  ( 0.12,  -4, "left",  "top"),
    "GPT-5.4-mini (low)":     ( 0.12,   4, "left",  "bottom"),
    "GPT-5.4-mini (none)":    ( 0.15,  -4, "left",  "top"),
    "Gemini 3.1 Pro":         ( 0.08,   4, "left",  "bottom"),
    "Gemini 3 Flash":         (-0.30,   4, "right", "bottom"),
    "Qwen 3.5 Flash":         ( 0.10,   3, "left",  "bottom"),
    "Qwen 3.5 Plus":          ( 0.10,  -5, "left",  "top"),
}


def build():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_facecolor("#fafaf8")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, ls="-", alpha=0.18, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    ax.set_xscale("log")
    ax.set_xlim(5, 1200)
    ax.set_ylim(-5, 105)
    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x)}K"))
    ax.minorticks_off()
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{int(y)}%"))

    ax.set_xlabel("Total Tokens Consumed (thinking + output)",
                  fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Exact Solve Rate",
                  fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Reasoning Token Efficiency Across Frontier Models",
                 fontsize=13, fontweight="bold", pad=12)

    txt_fx = [pe.withStroke(linewidth=3, foreground="white")]

    # ── Failure zone — very subtle ──
    ax.axhspan(-5, 18, color="#fee2e2", alpha=0.22, zorder=1)
    ax.text(7, 15.5, "no-reasoning failure zone",
            fontsize=6.5, color="#b91c1c", alpha=0.45,
            fontstyle="italic", path_effects=txt_fx)

    # ── Plot all points ──
    for full, disp, solved, tok, prov, tier, reason in MODELS:
        ax.scatter(tok, solved, s=TIER_SIZE[tier], c=PROV_COLOR[prov],
                   marker="o", edgecolors="white", linewidths=1.0,
                   zorder=5, alpha=0.88)

    # ── GPT-5.4 scaling line ──
    gpts = [(9, 12), (145, 85), (265, 91)]
    ax.plot([p[0] for p in gpts], [p[1] for p in gpts],
            color="#2563eb", alpha=0.25, lw=2, ls="--", zorder=3)

    # ── GPT-5.4-mini scaling line ──
    mini = [(11, 8), (152, 49), (503, 51)]
    ax.plot([p[0] for p in mini], [p[1] for p in mini],
            color="#2563eb", alpha=0.15, lw=1.5, ls=":", zorder=3)

    # ── Labels for non-Anthropic models ──
    for full, disp, solved, tok, prov, tier, reason in MODELS:
        if disp is None:  # Anthropic — handled separately
            continue
        if full not in LABEL_POS:
            continue
        dx, dy, ha, va = LABEL_POS[full]
        tx = tok * (10 ** dx)
        ty = solved + dy
        ax.annotate(
            disp, (tok, solved), xytext=(tx, ty),
            fontsize=7, color="#333333", ha=ha, va=va,
            path_effects=txt_fx, zorder=6,
            arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                            lw=0.4, shrinkA=3, shrinkB=1),
        )

    # ── Claude cluster label — simple text above the red dots ──
    ax.text(82, 9, "Claude (Haiku / Sonnet / Opus)",
            fontsize=7, color="#b91c1c", fontstyle="italic",
            ha="center", va="bottom", path_effects=txt_fx, zorder=6)

    # ── Legend: Provider ──
    prov_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                      markersize=8, markeredgecolor="white",
                      markeredgewidth=0.5, label=p, linestyle="None")
        for p, c in PROV_COLOR.items()
    ]
    leg1 = ax.legend(handles=prov_handles, loc="upper left",
                     title="Provider", title_fontsize=8,
                     fontsize=7.5, framealpha=0.92,
                     edgecolor="#dddddd", borderpad=0.4,
                     bbox_to_anchor=(0.0, 1.0))
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)

    # ── Legend: Model size ──
    size_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#888",
                      markersize=np.sqrt(TIER_SIZE[t]) * 0.40,
                      markeredgecolor="white", markeredgewidth=0.5,
                      label=lbl, linestyle="None")
        for lbl, t in [("Small (Flash/Haiku)", 1),
                        ("Mid (Plus/Sonnet/Mini)", 2),
                        ("Large (Pro/Opus/5.4)", 3)]
    ]
    leg2 = ax.legend(handles=size_handles, loc="upper left",
                     title="Model size", title_fontsize=8,
                     fontsize=6.5, framealpha=0.92,
                     edgecolor="#dddddd", borderpad=0.4,
                     bbox_to_anchor=(0.15, 1.0))
    leg2.get_frame().set_linewidth(0.5)
    ax.add_artist(leg2)
    ax.add_artist(leg1)

    for sp in ax.spines.values():
        sp.set_color("#aaaaaa")
        sp.set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=9)

    fig.tight_layout()
    out = OUT / "fig_efficiency_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
