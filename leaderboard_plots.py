"""Render leaderboard dot plots from docs/leaderboard data.

Data is transcribed from https://bergson.readthedocs.io/en/latest/leaderboard.html
(main table and per-token section). Writes figures/lds_leaderboard.png and
figures/qld_leaderboard.png in the style of the announcement-thread plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

FAMILIES = {
    "magic": ("#2a78d6", "MAGIC"),
    "influence": ("#eb6834", "Influence function"),
    "projected": ("#1baf7a", "Projected influence function"),
    "rawgrad": ("#eda100", "Gradients"),
    "nongrad": ("#4a3aa7", "Non-gradient baseline"),
}

# (label, family, LDS (lo, hi), QLD (lo, hi), per-token QLD (lo, hi) or None)
ROWS = [
    ("MAGIC (fixed seed)", "magic",
     (0.931, 0.925, 0.936), (0.100, 0.090, 0.112), (1.375, 1.348, 1.403)),
    ("MAGIC (across seeds)", "magic",
     (0.829, 0.815, 0.840), (0.098, 0.087, 0.110), None),
    ("EKFAC + CG", "influence",
     (0.685, 0.664, 0.702), (0.077, 0.066, 0.089), (1.014, 0.986, 1.043)),
    ("EKFAC + ASTRA", "influence",
     (0.629, 0.607, 0.647), (0.071, 0.060, 0.084), None),
    ("Eigenvalue-corrected Shampoo", "influence",
     (0.517, 0.491, 0.539), (0.071, 0.060, 0.082), (0.922, 0.887, 0.957)),
    ("SOURCE", "influence",
     (0.473, 0.446, 0.498), (0.071, 0.060, 0.084), (0.908, 0.877, 0.942)),
    ("EKFAC", "influence",
     (0.454, 0.426, 0.479), (0.070, 0.058, 0.082), (0.863, 0.833, 0.894)),
    ("KFAC", "influence",
     (0.420, 0.391, 0.446), (0.067, 0.056, 0.080), (0.786, 0.758, 0.815)),
    ("TrackStar (no optimizer)", "projected",
     (0.276, 0.252, 0.299), (0.045, 0.036, 0.055), (0.492, 0.467, 0.517)),
    ("TrackStar (Adam)", "projected",
     (0.242, 0.216, 0.267), (0.043, 0.034, 0.052), None),
    ("BM25", "nongrad",
     (0.220, 0.185, 0.252), (0.062, 0.048, 0.076), (0.677, 0.650, 0.704)),
    ("Activation similarity (act. cov. preconditioner)", "nongrad",
     (0.157, 0.120, 0.192), (0.055, 0.043, 0.069), (0.073, 0.056, 0.092)),
    ("Gradient cosine similarity", "rawgrad",
     (0.156, 0.131, 0.181), (0.021, 0.016, 0.027), (0.217, 0.206, 0.229)),
    ("Gradient dot product", "rawgrad",
     (0.156, 0.130, 0.180), (0.019, 0.015, 0.024), None),
    ("TRAK (8-model ensemble)", "projected",
     (0.138, 0.111, 0.165), (0.032, 0.024, 0.040), (0.119, 0.112, 0.125)),
    ("Qwen3-Embedding-8B semantic search", "nongrad",
     (0.132, 0.093, 0.169), (0.049, 0.038, 0.061), (0.008, 0.006, 0.010)),
    ("Projected gradient cosine similarity", "rawgrad",
     (0.132, 0.103, 0.159), (0.016, 0.012, 0.020), None),
    ("Jina v5 semantic search", "nongrad",
     (0.124, 0.087, 0.160), (0.046, 0.035, 0.059), None),
    ("Activation similarity", "nongrad",
     (0.110, 0.070, 0.149), (0.049, 0.037, 0.063), (0.028, 0.018, 0.039)),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def family_legend(ax):
    handles = [Patch(facecolor=c, edgecolor="none") for c, _ in FAMILIES.values()]
    labels = [name for _, name in FAMILIES.values()]
    return ax.legend(
        handles, labels,
        loc="lower left", bbox_to_anchor=(-0.52, 1.01), ncol=5,
        frameon=False, fontsize=11.5, handlelength=1.0, handleheight=1.0,
        columnspacing=1.3, handletextpad=0.5, borderaxespad=0,
    )


def style_axes(ax, n_rows):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.tick_params(axis="both", length=0)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_yticks(range(n_rows))


def draw_point(ax, x, lo, hi, y, color, label_at=None, open_marker=False):
    ax.errorbar(
        x, y, xerr=[[x - lo], [hi - x]], fmt="none",
        ecolor=color, elinewidth=1.6, capsize=3, capthick=1.6, zorder=3,
    )
    face = SURFACE if open_marker else color
    ax.plot(
        x, y, "o", markersize=7.5, markerfacecolor=face,
        markeredgecolor=color, markeredgewidth=1.6, zorder=4,
    )
    if label_at is not None:
        ax.annotate(
            f"{x:.3f}", (label_at, y), xytext=(6, 0),
            textcoords="offset points", va="center", ha="left",
            fontsize=11.5, color=INK, zorder=5,
        )


def lds_plot(out: Path):
    rows = sorted(ROWS, key=lambda r: r[2][0], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8.6))
    style_axes(ax, len(rows))
    for y, (label, family, (lds, lo, hi), _, _) in enumerate(rows):
        draw_point(ax, lds, lo, hi, y, FAMILIES[family][0], label_at=hi)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("LDS (95% CI)", fontsize=13, color=INK, labelpad=8)
    family_legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def qld_plot(out: Path):
    rows = sorted(ROWS, key=lambda r: r[3][0], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 9.2))
    style_axes(ax, len(rows))
    for y, (label, family, _, (qld, lo, hi), token) in enumerate(rows):
        color = FAMILIES[family][0]
        y_seq = y - 0.18 if token else y
        draw_point(ax, qld, lo, hi, y_seq, color, label_at=hi)
        if token:
            tok, tlo, thi = token
            draw_point(ax, tok, tlo, thi, y + 0.18, color, label_at=thi,
                       open_marker=True)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.set_xscale("log")
    ax.set_xlim(0.004, 2.6)
    ticks = [0.01, 0.03, 0.1, 0.3, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=12)
    ax.minorticks_off()
    ax.set_xlabel("Proponent QLD (95% CI, log scale)", fontsize=13,
                  color=INK, labelpad=8)
    series_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7.5,
               markerfacecolor=MUTED, markeredgecolor=MUTED),
        Line2D([], [], marker="o", linestyle="none", markersize=7.5,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=1.6),
    ]
    families = family_legend(ax)
    ax.legend(
        series_handles, ["Per-sequence QLD", "Per-token QLD"],
        loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False,
        fontsize=11.5, handletextpad=0.5, borderaxespad=0, labelcolor=INK,
    )
    ax.add_artist(families)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25,
                bbox_extra_artists=(families,))
    plt.close(fig)


if __name__ == "__main__":
    figures = Path(__file__).resolve().parent / "figures"
    figures.mkdir(exist_ok=True)
    lds_plot(figures / "lds_leaderboard.png")
    qld_plot(figures / "qld_leaderboard.png")
    print(f"Wrote {figures}/lds_leaderboard.png and {figures}/qld_leaderboard.png")
