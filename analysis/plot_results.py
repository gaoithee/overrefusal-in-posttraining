"""
plot_results.py

Unified plotting script for overrefusal experiments.

Produces both keyword-based and judge-based plots from experiment results.

Usage
-----
    # Plot all metrics for a specific results directory
    python plot_results.py --results-dir results/olmo2

    # Plot multiple directories together (comparison)
    python plot_results.py --results-dir results/olmo2 results/olmo3 --compare

    # Select specific plot types
    python plot_results.py --results-dir results/olmo2 --plots refusal_rate judge_breakdown

Available plots
---------------
  refusal_rate     — keyword-based refusal rate across checkpoints
  judge_breakdown  — 3-class stacked bar (compliance / partial / full refusal)
  fp_fn            — over-refusal (FP) vs under-refusal (FN) rates
  heatmap          — refusal rate heatmap by checkpoint × category
  judge_heatmap    — judge-label heatmap by checkpoint × category
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COMPLIANCE_COLOR  = "#4CAF50"   # green
PARTIAL_COLOR     = "#FFC107"   # amber
FULL_REF_COLOR    = "#F44336"   # red
INCOHERENT_COLOR  = "#9E9E9E"   # grey

CHECKPOINT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    return pd.read_csv(path)


def _savefig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: keyword refusal rate across checkpoints
# ---------------------------------------------------------------------------

def plot_refusal_rate(raw_df: pd.DataFrame, out_dir: Path) -> None:
    if "checkpoint" not in raw_df.columns or "predicted_refusal" not in raw_df.columns:
        logger.warning("Skipping refusal_rate plot: missing columns.")
        return

    stats = (
        raw_df.groupby("checkpoint")["predicted_refusal"]
        .agg(["mean", "sem"])
        .rename(columns={"mean": "rate", "sem": "err"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.2), 5))
    x = range(len(stats))
    bars = ax.bar(x, stats["rate"], yerr=stats["err"],
                  color=CHECKPOINT_COLORS[:len(stats)], capsize=4, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats["checkpoint"], rotation=25, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Refusal rate (keyword)")
    ax.set_title("Keyword-based Refusal Rate by Checkpoint")
    ax.set_ylim(0, min(1.05, stats["rate"].max() * 1.3 + 0.05))
    for bar, rate in zip(bars, stats["rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_dir / "refusal_rate.png")


# ---------------------------------------------------------------------------
# Plot 2: judge 3-class stacked bar
# ---------------------------------------------------------------------------

def plot_judge_breakdown(raw_df: pd.DataFrame, out_dir: Path) -> None:
    if "judge_label" not in raw_df.columns:
        logger.warning("Skipping judge_breakdown: missing judge_label column.")
        return

    counts = (
        raw_df.groupby(["checkpoint", "judge_label"])
        .size()
        .unstack(fill_value=0)
    )
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0)

    cols_order = ["1_full_compliance", "3_partial_refusal", "2_full_refusal"]
    colors     = [COMPLIANCE_COLOR, PARTIAL_COLOR, FULL_REF_COLOR]
    labels     = ["Full compliance", "Partial refusal", "Full refusal"]

    # Add incoherent column if available
    if "is_coherent" in raw_df.columns:
        incoherent = raw_df.groupby("checkpoint")["is_coherent"].apply(
            lambda s: (~s.fillna(False)).sum() / len(s)
        ).rename("incoherent")
        pct["incoherent"] = incoherent.reindex(pct.index).fillna(0)
        cols_order.append("incoherent")
        colors.append(INCOHERENT_COLOR)
        labels.append("Incoherent")

    present = [c for c in cols_order if c in pct.columns]
    pct = pct.reindex(columns=present, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(6, len(pct) * 1.3), 6))
    x = np.arange(len(pct))
    bottoms = np.zeros(len(pct))

    for col, color, label in zip(
        present,
        colors[:len(present)],
        labels[:len(present)],
    ):
        vals = pct[col].values
        ax.bar(x, vals, bottom=bottoms, color=color, label=label, width=0.65)
        # Annotate segments > 5%
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(xi, b + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        bottoms += vals

    ax.set_xticks(list(x))
    ax.set_xticklabels(pct.index, rotation=25, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Proportion of responses")
    ax.set_title("Judge Classification by Checkpoint\n(3-class: compliance / partial / full refusal)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, out_dir / "judge_breakdown.png")


# ---------------------------------------------------------------------------
# Plot 3: FP vs FN scatter (keyword or judge)
# ---------------------------------------------------------------------------

def plot_fp_fn(raw_df: pd.DataFrame, out_dir: Path, use_judge: bool = False) -> None:
    if "label" not in raw_df.columns:
        logger.warning("Skipping fp_fn plot: missing label column.")
        return

    from evaluation.metrics import compute_metrics, compute_judge_metrics

    try:
        if use_judge and "judge_label" in raw_df.columns:
            m = compute_judge_metrics(raw_df)
            title = "Over-refusal (FP) vs Under-refusal (FN) — Judge"
            suffix = "_judge"
        else:
            m = compute_metrics(raw_df)
            title = "Over-refusal (FP) vs Under-refusal (FN) — Keyword"
            suffix = "_keyword"
    except Exception as exc:
        logger.warning("Could not compute metrics for FP/FN plot: %s", exc)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = CHECKPOINT_COLORS

    for i, (ckpt, row) in enumerate(m.iterrows()):
        fp = row.get("fp_rate", float("nan"))
        fn = row.get("fn_rate", float("nan"))
        if np.isnan(fp) and np.isnan(fn):
            continue
        ax.scatter(fp, fn, color=colors[i % len(colors)], s=100, zorder=3)
        ax.annotate(str(ckpt), (fp, fn), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Over-refusal rate (FP) — safe prompts refused")
    ax.set_ylabel("Under-refusal rate (FN) — harmful prompts answered")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    _savefig(fig, out_dir / f"fp_fn{suffix}.png")


# ---------------------------------------------------------------------------
# Plot 4: heatmap (checkpoint × category)
# ---------------------------------------------------------------------------

def _heatmap(matrix: pd.DataFrame, title: str, fmt: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.8),
                                    max(4, len(matrix) * 0.5)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for r in range(len(matrix)):
        for c in range(len(matrix.columns)):
            v = matrix.values[r, c]
            ax.text(c, r, format(v, fmt),
                    ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 or v < 0.2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, path)


def plot_heatmap(raw_df: pd.DataFrame, out_dir: Path) -> None:
    needed = {"checkpoint", "category", "predicted_refusal"}
    if not needed.issubset(raw_df.columns):
        logger.warning("Skipping heatmap: missing columns %s", needed - set(raw_df.columns))
        return
    matrix = raw_df.pivot_table(
        index="checkpoint", columns="category",
        values="predicted_refusal", aggfunc="mean"
    )
    _heatmap(matrix, "Refusal Rate Heatmap (keyword)\nCheckpoint × Category",
             ".0%", out_dir / "heatmap_keyword.png")


def plot_judge_heatmap(raw_df: pd.DataFrame, out_dir: Path) -> None:
    needed = {"checkpoint", "category", "judge_label"}
    if not needed.issubset(raw_df.columns):
        logger.warning("Skipping judge_heatmap: missing columns %s", needed - set(raw_df.columns))
        return
    raw_df = raw_df.copy()
    raw_df["judge_refused"] = (raw_df["judge_label"] != "1_full_compliance").astype(float)
    matrix = raw_df.pivot_table(
        index="checkpoint", columns="category",
        values="judge_refused", aggfunc="mean"
    )
    _heatmap(matrix, "Refusal Rate Heatmap (judge)\nCheckpoint × Category",
             ".0%", out_dir / "heatmap_judge.png")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALL_PLOTS = {
    "refusal_rate":   plot_refusal_rate,
    "judge_breakdown": plot_judge_breakdown,
    "heatmap":        plot_heatmap,
    "judge_heatmap":  plot_judge_heatmap,
}


def run_plots(
    results_dir: Path,
    plots: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
) -> None:
    raw_path = results_dir / "raw_results.csv"
    if not raw_path.exists():
        logger.error("raw_results.csv not found in %s", results_dir)
        return

    df = pd.read_csv(raw_path)
    out = out_dir or results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    selected = plots or list(ALL_PLOTS.keys())
    for name in selected:
        fn = ALL_PLOTS.get(name)
        if fn is None:
            logger.warning("Unknown plot: %s", name)
            continue
        logger.info("Generating plot: %s", name)
        try:
            if name == "fp_fn":
                plot_fp_fn(df, out, use_judge=False)
                plot_fp_fn(df, out, use_judge=True)
            else:
                fn(df, out)
        except Exception as exc:
            logger.error("Plot %s failed: %s", name, exc)

    logger.info("All plots saved to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from overrefusal experiment results."
    )
    parser.add_argument(
        "--results-dir", nargs="+", required=True,
        help="Results directory / directories containing raw_results.csv.",
    )
    parser.add_argument(
        "--plots", nargs="+", default=None,
        choices=list(ALL_PLOTS.keys()) + ["fp_fn"],
        help="Which plots to generate. Default: all.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for plots. Default: <results-dir>/plots/",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")

    for rdir in args.results_dir:
        run_plots(
            results_dir=Path(rdir),
            plots=args.plots,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )


if __name__ == "__main__":
    main()
