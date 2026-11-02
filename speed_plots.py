"""Render the leaderboard's speed figures and tables from speed/estimates.json.

The estimates are bergson-estimate's predictions for each method's leaderboard
recipe on one 8x A40 node, at a ladder of dataset sizes and numbers of queries
(see speed/README.md). Each stage's time is fitted with a cost model

    t = fixed + per document x dataset size + per query x queries
        + per pair x dataset size x queries

whose terms are non-negative. Writes figures/speed_*.png and the tables in
index.html.
"""

import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.optimize import nnls

from leaderboard_plots import FAMILIES, GRID, INK, MUTED, SURFACE

HERE = Path(__file__).resolve().parent
REFERENCE = (4608, 50)
"""The leaderboard's training chunks and queries."""

# method: (label, family, LDS)
METHODS = {
    "magic": ("MAGIC", "magic", 0.931),
    "unrolled_gauss_newton": ("Unrolled Gauss-Newton", "magic", 0.8185),
    "ekfac_cg": ("EKFAC + CG", "influence", 0.685),
    "ekfac_astra": ("EKFAC + ASTRA", "influence", 0.605),
    "shampoo_astra": ("Eigenvalue-corrected Shampoo + ASTRA", "influence", 0.625),
    # Its one real run took about four times the estimate
    "lissa": ("LiSSA (underestimated)", "influence", 0.572),
    "shampoo": ("Eigenvalue-corrected Shampoo", "influence", 0.517),
    "source_adam": ("SOURCE (Adam)", "influence", 0.473),
    "ekfac": ("EKFAC", "influence", 0.454),
    "kfac": ("KFAC", "influence", 0.420),
    "trackstar_p64": ("TrackStar (no optimizer)", "projected", 0.276),
    "trackstar_adam_p64": ("TrackStar (Adam)", "projected", 0.242),
    "gradient_cosine": ("Gradient cosine similarity", "rawgrad", 0.156),
    "gradient_dot": ("Gradient dot product", "rawgrad", 0.156),
    "trak": ("TRAK (8-model ensemble)", "projected", 0.138),
    "gradient_cosine_projected": ("Projected gradient cosine similarity", "rawgrad", 0.132),
    "kfac_p64": ("KFAC (projection 64)", "influence", 0.244),
}

# Real runs of the same recipes on the leaderboard data, in seconds
# Real runs of the recipes on all eight GPUs, in seconds
MEASURED = {
    "ekfac": 521,
    "lissa": 25.4 * 3600,
    "unrolled_gauss_newton": 79 * 60,
}

# Stages grouped by what they cost, in pipeline order, each group one hue. Every
# method's stack passes the palette validator's adjacent checks; MAGIC's green
# beside red sits in the band that needs the surface gaps between bands
GROUPS = [
    ("Startup and overhead", "#b9b7ae", ["Startup and overhead"]),
    ("Model training", "#2a78d6", ["Model training", "Training replay", "Checkpoint saves"]),
    ("Covariance fit", "#eb6834", ["Covariance fit"]),
    ("Eigenvalue correction", "#1baf7a", ["Eigenvalue correction"]),
    ("Eigendecomposition and inversion", "#6250d6", ["Eigendecomposition", "Gram inversion"]),
    ("Query gradients", "#e87ba4", ["Query gradients", "Query preconditioning"]),
    (
        "Per-query computation",
        "#008300",
        ["ASTRA refinement", "LiSSA iterations", "Backward through training"],
    ),
    ("Trainer-state copies and I/O", "#e34948", ["Trainer-state copies", "Checkpoint I/O"]),
    ("Training gradients and scoring", "#eda100", ["Training gradients and scoring", "Training-set probabilities"]),
]
GROUP_OF = {stage: name for name, _, stages in GROUPS for stage in stages}
COLOR = {name: color for name, color, _ in GROUPS}

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def features(n, q):
    n, q = np.asarray(n, float), np.asarray(q, float)
    return np.stack([np.ones_like(n), n, q, n * q], axis=-1)


class CostModel:
    """Per-stage-group cost models of one method, fitted to its estimates."""

    def __init__(self, rows: list[dict]):
        self.points = [(r["dataset_size"], r["queries"]) for r in rows]
        groups = {name for r in rows for s in r["stages"] for name in [GROUP_OF.get(s, s)]}
        self.terms = {}
        x = features(*zip(*self.points))
        # Each term relative to the reference, so the fit weighs them alike
        scale = features(*REFERENCE)
        self.observed = {}
        for group in groups:
            y = np.array([
                sum(v for s, v in r["stages"].items() if GROUP_OF.get(s, s) == group)
                for r in rows
            ])
            # Relative error at every point counts alike
            w = 1 / np.maximum(y, 1.0)
            coef, _ = nnls((x / scale) * w[:, None], y * w)
            self.terms[group] = coef / scale
            self.observed[group] = y
        self.peak = {(r["dataset_size"], r["queries"]): r["peak_gib"] for r in rows}
        self.estimated = {(r["dataset_size"], r["queries"]): r["stages"] for r in rows}

    def stages(self, n, q) -> dict[str, np.ndarray]:
        return {g: features(n, q) @ c for g, c in self.terms.items()}

    def total(self, n, q) -> np.ndarray:
        return sum(self.stages(n, q).values())

    def fit_error(self) -> float:
        """Largest relative error of the model's totals at the estimated points."""
        n, q = map(np.array, zip(*self.points))
        observed = sum(self.observed.values())
        return float(np.max(np.abs(self.total(n, q) / observed - 1)))


# Methods that run on the checkpoints of a training run that already exists; their
# runs also train the model, which their times leave out
ON_CHECKPOINTS = {"magic", "unrolled_gauss_newton"}
TRAINING = "Model training"


def load() -> dict[str, CostModel]:
    data = json.load(open(HERE / "speed" / "estimates.json"))
    models = {m: CostModel(data[m]) for m in METHODS if m in data}
    for m in ON_CHECKPOINTS & set(models):
        model = models[m]
        model.training = model.terms.pop(TRAINING, np.zeros(4))
        model.observed.pop(TRAINING, None)
        for stages in model.estimated.values():
            for s in [s for s in stages if GROUP_OF.get(s, s) == TRAINING]:
                del stages[s]
    return models


def training_hours(models, n):
    """Hours to train the attributed model on ``n`` documents, from MAGIC's run."""
    return hours(features(n, 0) @ models["magic"].training)


def hours(seconds):
    return np.asarray(seconds) / 3600


def style(ax, grid_y=True):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="both" if grid_y else "x", color=GRID, linewidth=1)
    ax.tick_params(length=0, labelsize=11)


def family_handles():
    used = [f for f in FAMILIES if any(METHODS[m][1] == f for m in METHODS)]
    return [Patch(facecolor=FAMILIES[f][0], edgecolor="none") for f in used], [
        FAMILIES[f][1] for f in used
    ]


PLAIN = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")


def plain_log(ax, x=True, y=False):
    """Log scale with ticks written as plain numbers."""
    for on, axis, scale in ((x, ax.xaxis, ax.set_xscale), (y, ax.yaxis, ax.set_yscale)):
        if on:
            scale("log")
            axis.set_major_formatter(PLAIN)


def dodge(fig, ax, texts, gap=2.0, rounds=200):
    """Shift overlapping labels apart vertically, keeping each beside its point."""
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    shift = [0.0] * len(texts)
    for _ in range(rounds):
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if bi.x1 < bj.x0 or bj.x1 < bi.x0:
                    continue
                lo, hi = (i, j) if bi.y0 + shift[i] <= bj.y0 + shift[j] else (j, i)
                overlap = boxes[lo].y1 + shift[lo] + gap - (boxes[hi].y0 + shift[hi])
                if overlap > 0:
                    shift[lo] -= overlap / 2
                    shift[hi] += overlap / 2
                    moved = True
        if not moved:
            break
    for t, dy in zip(texts, shift):
        x, y = t.get_position()
        t.set_position((x, y + dy * 72 / fig.dpi))


# Labelled on the left of their point, where the right is crowded
LABEL_LEFT = {"kfac_p64"}


def pareto_plot(models, out):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    style(ax)
    texts = []
    for m, model in models.items():
        label, family, lds = METHODS[m]
        t = float(hours(model.total(*REFERENCE)))
        color = FAMILIES[family][0]
        ax.plot(t, lds, "o", markersize=8, color=color, zorder=4)
        r = t
        if m in MEASURED:
            # A measured run of a method on checkpoints trained its model too
            r = hours(MEASURED[m] - (features(*REFERENCE) @ getattr(model, "training", np.zeros(4))))
        # Label beyond the outer point so it never sits on the estimate-measured segment.
        left = m in LABEL_LEFT
        texts.append(ax.annotate(label, (min(t, r) if left else max(t, r), lds),
                                 xytext=(-8 if left else 8, 0), textcoords="offset points",
                                 ha="right" if left else "left", va="center", fontsize=10.5, color=INK))
        if m in MEASURED:
            ax.plot(r, lds, "o", markersize=8, markerfacecolor=SURFACE,
                    markeredgecolor=color, markeredgewidth=1.6, zorder=4)
            ax.plot([t, r], [lds, lds], color=color, linewidth=1, zorder=3)
    train = float(training_hours(models, REFERENCE[0]))
    ax.axvline(train, color=MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.annotate("Training the model", (train, 0.99), xytext=(4, 0), textcoords="offset points",
                va="top", fontsize=10, color=MUTED)
    plain_log(ax)
    ax.set_xlabel("Hours on one 8x A40 node (log scale)", fontsize=12.5, labelpad=8)
    ax.set_ylabel("LDS", fontsize=12.5)
    ax.set_ylim(0, 1)
    handles, labels = family_handles()
    handles += [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=MUTED),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.6),
    ]
    labels += ["Estimated", "Measured"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
              frameon=False, fontsize=10.5, borderaxespad=0)
    fig.tight_layout()
    dodge(fig, ax, texts)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


AMORTIZED = ["magic", "unrolled_gauss_newton", "ekfac_cg", "ekfac_astra", "lissa", "ekfac", "kfac",
             "trackstar_p64", "trak", "gradient_cosine"]


def amortization_plot(models, out):
    queries = np.geomspace(1, 1000, 60)
    sizes = [REFERENCE[0], 294912]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    texts = {ax: [] for ax in axes}
    for ax, n in zip(axes, sizes):
        style(ax)
        for m in AMORTIZED:
            if m not in models:
                continue
            label, family, _ = METHODS[m]
            t = hours(models[m].total(np.full_like(queries, n), queries))
            ax.plot(queries, t, color=FAMILIES[family][0], linewidth=2)
            texts[ax].append(ax.annotate(label, (queries[-1], t[-1]), xytext=(5, 0),
                                         textcoords="offset points", va="center", fontsize=9.5))
        train = float(training_hours(models, n))
        ax.axhline(train, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.annotate("Training the model", (queries[-1], train), xytext=(0, 3), textcoords="offset points", ha="right",
                    va="bottom", fontsize=9, color=MUTED)
        plain_log(ax, y=True)
        ax.set_xlabel("Number of queries (log scale)", fontsize=12)
        ax.set_title(f"Dataset size {n:,} documents", fontsize=12.5, loc="left")
    axes[0].set_ylabel("Hours on one 8x A40 node (log scale)", fontsize=12)
    handles, labels = family_handles()
    axes[0].legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.08),
                   ncol=5, frameon=False, fontsize=10.5, borderaxespad=0)
    fig.tight_layout()
    for ax in axes:
        dodge(fig, ax, texts[ax])
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def share_plot(models, out):
    sizes = np.geomspace(1152, 1179648, 80)
    ms = [m for m in METHODS if m in models]
    cols = 4
    rows = -(-len(ms) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.1 * rows), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ms):
        stages = models[m].stages(sizes, np.full_like(sizes, REFERENCE[1]))
        total = sum(stages.values())
        order = [g for g, _, _ in GROUPS if g in stages]
        bottom = np.zeros_like(sizes)
        for g in order:
            share = 100 * np.clip(stages[g], 0, None) / total
            ax.fill_between(sizes, bottom, bottom + share, color=COLOR[g],
                            edgecolor=SURFACE, linewidth=1.5)
            bottom += share
        plain_log(ax)
        ax.set_xlim(sizes[0], sizes[-1])
        ax.set_ylim(0, 100)
        ax.set_title(METHODS[m][0], fontsize=10.5, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=9)
    for ax in axes.flat[len(ms):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Dataset size (documents, log scale)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of time (%)", fontsize=10)
    used = [g for g, _, _ in GROUPS if any(g in models[m].terms for m in ms)]
    fig.legend([Patch(facecolor=COLOR[g]) for g in used], used, loc="lower left",
               bbox_to_anchor=(0.01, 1.0), ncol=5, frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def crossovers(sizes, stages):
    """Dataset sizes where the stage taking the most time changes."""
    names = list(stages)
    leader = np.argmax(np.stack([stages[g] for g in names]), axis=0)
    out = []
    for i in np.nonzero(np.diff(leader))[0]:
        out.append((float(np.sqrt(sizes[i] * sizes[i + 1])), names[leader[i]], names[leader[i + 1]]))
    return out


def stage_curves_plot(models, out, methods=("ekfac", "magic")):
    sizes = np.geomspace(1152, 1179648, 200)
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 6), sharey=True)
    for ax, m in zip(axes, methods):
        style(ax)
        stages = models[m].stages(sizes, np.full_like(sizes, REFERENCE[1]))
        for g, _, _ in GROUPS:
            if g not in stages or np.max(stages[g]) < 1:
                continue
            ax.plot(sizes, hours(np.maximum(stages[g], 1e-3)), color=COLOR[g], linewidth=2, label=g)
            est = models[m].observed[g]
            pts = [(n, v) for (n, q), v in zip(models[m].points, est) if q == REFERENCE[1]]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, hours(ys), "o", markersize=6, color=COLOR[g],
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
        for n, before, after in crossovers(sizes, stages):
            ax.axvline(n, color=MUTED, linewidth=1, linestyle="--")
            ax.annotate(f"{after} overtakes\n{before.lower()} at {float(f'{n:.2g}'):,.0f}", (n, 0.98),
                        xycoords=("data", "axes fraction"), xytext=(4, 0),
                        textcoords="offset points", va="top", fontsize=9, color=INK)
        plain_log(ax, y=True)
        ax.set_title(f"{METHODS[m][0]}, {REFERENCE[1]} queries", fontsize=12.5, loc="left")
        ax.set_xlabel("Dataset size (documents, log scale)", fontsize=12)
    axes[0].set_ylabel("Hours on one 8x A40 node (log scale)", fontsize=12)
    used = [g for g, _, _ in GROUPS if any(g in models[m].terms for m in methods)]
    fig.legend([plt.Line2D([], [], color=COLOR[g], linewidth=2) for g in used], used,
               loc="lower left", bbox_to_anchor=(0.01, 1.0), ncol=4, frameon=False,
               fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# Estimates known to be low; kept off the regime map so they can't win regions they don't.
UNDERESTIMATED = {"lissa"}
REGIME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# LDS floors of the regime map's panels: EKFAC's, ASTRA's and the metadifferentiation methods'
FLOORS = [0.45, 0.6, 0.8]


def regime_plot(models, out):
    """For each LDS floor, the fastest method reaching it at each dataset size and
    number of queries."""
    # Fixed by each method's place among all that could qualify, so a method keeps its color
    candidates = [m for m in METHODS if METHODS[m][2] >= min(FLOORS) and m not in UNDERESTIMATED]
    colors = {m: REGIME_COLORS[i] for i, m in enumerate(candidates)}
    tint = lambda m: 1 - 0.45 * (1 - np.array(plt.matplotlib.colors.to_rgb(colors[m])))  # noqa: E731
    sizes = np.geomspace(1152, 1179648, 120)
    queries = np.geomspace(1, 1000, 100)
    n, q = np.meshgrid(sizes, queries)
    fig, axes = plt.subplots(1, len(FLOORS), figsize=(16, 6), sharey=True)
    won = set()
    for ax, bar in zip(axes, FLOORS):
        ms = [m for m in candidates if m in models and METHODS[m][2] >= bar]
        times = np.stack([models[m].total(n, q) for m in ms])
        best = np.argmin(times, axis=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        rgb = np.array([tint(ms[i]) for i in best.ravel()])
        ax.pcolormesh(n, q, rgb.reshape(*best.shape, 3), shading="auto")
        total = hours(np.min(times, axis=0))
        levels = [lvl for lvl in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000) if total.min() < lvl < total.max()]
        cs = ax.contour(n, q, total, levels=levels, colors=INK, linewidths=0.8)
        ax.clabel(cs, fmt=lambda v: f"{v:g} h", fontsize=8.5)
        for i, m in enumerate(ms):
            mask = best == i
            if mask.any():
                won.add(m)
            if mask.sum() > 300:
                ax.annotate(METHODS[m][0], (np.exp(np.log(n[mask]).mean()), np.exp(np.log(q[mask]).mean())),
                            ha="center", va="center", fontsize=10, color=INK, weight="bold")
        plain_log(ax, y=True)
        ax.set_title(f"LDS at least {bar:.2f}", fontsize=12, loc="left")
        ax.set_xlabel("Dataset size (documents, log scale)", fontsize=11)
    axes[0].set_ylabel("Number of queries (log scale)", fontsize=11)
    order = [m for m in candidates if m in won]
    fig.legend([Patch(facecolor=tint(m)) for m in order], [METHODS[m][0] for m in order],
               loc="lower left", bbox_to_anchor=(0.01, 1.0), ncol=len(order), frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def tables(models) -> str:
    rows = []
    for m, model in models.items():
        label, _, lds = METHODS[m]
        # The estimate itself, with each method's own stage names
        stages = model.estimated[REFERENCE]
        total = sum(stages.values())
        parts = sorted(((v, g) for g, v in stages.items() if v >= 0.5), reverse=True)
        dominant = parts[0][1]
        peak = model.peak.get(REFERENCE, 0)
        rows.append(
            f"<tr><td>{escape(label)}</td><td class=num>{lds:.3f}</td>"
            f"<td class=num>{duration(total)}</td><td>{escape(dominant)}</td>"
            f"<td>{' · '.join(f'{escape(g)} {duration(v)}' for v, g in parts)}</td>"
            f"<td class=num>{peak:.1f}</td></tr>"
        )
    stage_table = (
        "<table><caption>Each method's stages at the leaderboard's 4,608 training "
        "documents and 50 queries, estimated for one 8x A40 node. LiSSA's one real "
        "run took 25.4 hours, about four times its estimate.</caption>"
        "<thead><tr><th>Method</th><th>LDS</th><th>Total</th><th>Longest stage</th>"
        "<th>Stages, longest first</th><th>Peak GiB per GPU</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )

    rows = []
    for m, model in models.items():
        a, b, c, d = sum(model.terms.values())
        n_star = a / b if b > 0 else float("inf")
        q_star = (a + b * REFERENCE[0]) / (c + d * REFERENCE[0]) if c + d > 0 else float("inf")
        rows.append(
            f"<tr><td>{escape(METHODS[m][0])}</td><td class=num>{duration(a)}</td>"
            f"<td class=num>{duration(b * 1000)}</td><td class=num>{duration(c)}</td>"
            f"<td class=num>{duration(d * 1e6)}</td>"
            f"<td class=num>{'never' if n_star == float('inf') else f'{n_star:,.0f}'}</td>"
            f"<td class=num>{'never' if q_star == float('inf') else f'{q_star:,.0f}'}</td>"
            f"<td class=num>{100 * model.fit_error():.0f}%</td></tr>"
        )
    model_table = (
        "<table><caption>The fitted cost model, total time = fixed + per document "
        "× dataset size + per query × number of queries + per pair × dataset size "
        "× number of queries, summed over stages. The last columns give the dataset "
        "size at which per-document time equals the fixed time, the number of "
        "queries at which per-query time equals the rest at 4,608 documents, and the "
        "model's largest error against the estimates it was fitted to.</caption>"
        "<thead><tr><th>Method</th><th>Fixed</th><th>Per 1,000 documents</th>"
        "<th>Per query</th><th>Per million document-query pairs</th>"
        "<th>Documents where per-document time passes fixed</th>"
        "<th>Queries where per-query time passes the rest</th><th>Fit error</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return {"stages": stage_table, "model": model_table}


def node_ratios(models) -> dict:
    """Each method's stage groups on n nodes of 4x GH200 as multiples of their
    time on one 8x A40 node, at each anchor dataset size with 50 queries: the
    exchange rate the page's calculator prices other nodes with."""
    path = HERE / "speed" / "estimates_gh200.json"
    if not path.exists():
        return {}
    data = json.load(open(path))
    out = {"anchors": ANCHORS, "counts": sorted({int(n) for n in data}), "ratios": {}}
    for n, methods in data.items():
        for m, rows in methods.items():
            if m not in models:
                continue
            by_size = {r["dataset_size"]: r["stages"] for r in rows if r["queries"] == REFERENCE[1]}
            ratios = {}
            for g, terms in models[m].terms.items():
                ratio = []
                for size in ANCHORS:
                    stages = by_size.get(size)
                    if stages is None:
                        break
                    gh200 = sum(v for st, v in stages.items() if GROUP_OF.get(st, st) == g)
                    a40 = float(features(size, REFERENCE[1]) @ terms)
                    ratio.append(gh200 / a40 if a40 > 1e-3 else 1.0)
                if len(ratio) == len(ANCHORS):
                    ratios[g] = ratio
            out["ratios"].setdefault(m, {})[n] = ratios
    return out


# Methods bergson runs on one node only: ASTRA and LiSSA refuse more
SINGLE_NODE = {"ekfac_cg", "ekfac_astra", "shampoo_astra", "lissa"}
# The dataset sizes node counts were estimated at, with 50 queries
ANCHORS = [4608, 73728]


def share_data(models) -> str:
    """The fitted cost model of every stage group, for the page's interactive share chart."""
    data = {
        "groups": [[g, COLOR[g]] for g, _, _ in GROUPS],
        "sizes": [1152, 1179648],
        "methods": [
            {
                "key": m,
                "label": METHODS[m][0],
                "family": FAMILIES[METHODS[m][1]][1],
                "color": FAMILIES[METHODS[m][1]][0],
                "lds": METHODS[m][2],
                "terms": {g: [float(c) for c in t] for g, t in model.terms.items()},
            }
            for m, model in models.items()
        ],
        "nodes": node_ratios(models),
        "single_node": sorted(SINGLE_NODE),
    }
    return f'<script type="application/json" id="share-data">{json.dumps(data)}</script>'


def fill(page: Path, blocks: dict[str, str]):
    """Put each block between its markers in ``page``."""
    text = page.read_text()
    for name, block in blocks.items():
        start, end = f"<!-- speed-tables:{name} -->", f"<!-- /speed-tables:{name} -->"
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = f"{before}{start}\n  {block}\n{end}{after}"
    page.write_text(text)


if __name__ == "__main__":
    models = load()
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    pareto_plot(models, figures / "speed_pareto.png")
    amortization_plot(models, figures / "speed_queries.png")
    share_plot(models, figures / "speed_share.png")
    stage_curves_plot(models, figures / "speed_stages.png")
    regime_plot(models, figures / "speed_regime.png")
    blocks = {name: f'<div class="table-wrap">{t}</div>' for name, t in tables(models).items()}
    blocks["share"] = share_data(models)
    fill(HERE / "index.html", blocks)
    print(f"Wrote figures/speed_*.png and the tables in index.html for {len(models)} methods")
