"""
analysis/plot_entanglement.py

Visualisations for the three entanglement experiments.

All plots read from the summary CSVs produced by run_representation_analysis.py
and/or the individual .npz geometry files, so this script requires no GPU.

Usage
-----
# Exp 1: entanglement profile for a single checkpoint
python -m analysis.plot_entanglement \
    --results-dir results/olmo2 \
    --plot entanglement_profile \
    --checkpoint sft__none

# Exp 2: entanglement evolution heatmap across training stages
python -m analysis.plot_entanglement \
    --results-dir results/olmo2 \
    --plot evolution_heatmap \
    --system-prompt none

# Exp 3: system-prompt comparison
python -m analysis.plot_entanglement \
    --results-dir results/olmo2 \
    --plot system_prompt_comparison

# All plots at once
python -m analysis.plot_entanglement \
    --results-dir results/olmo2 \
    --plot all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.representation_analysis import load_geometry, CheckpointGeometry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── colour / style ──────────────────────────────────────────────────────────
PALETTE = {
    "base":        "#7f7f7f",
    "sft":         "#1f77b4",
    "dpo":         "#ff7f0e",
    "instruct":    "#2ca02c",
    "rlhf":        "#9467bd",
    "none":        "#8c8c8c",
    "mistral_safety": "#d62728",
}

POST_TRAINING_ORDER = ["base", "sft", "dpo", "instruct"]


def _stage_color(tag: str) -> str:
    for key, color in PALETTE.items():
        if key in tag.lower():
            return color
    return "#333333"


def _load_summary(results_dir: Path, exp_suffix: str) -> pd.DataFrame | None:
    path = results_dir / "geometry" / f"summary_{exp_suffix}.csv"
    if not path.exists():
        logger.warning("Summary CSV not found: %s", path)
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plot 1 — Entanglement profile (single checkpoint, all layers)
# ---------------------------------------------------------------------------

def plot_entanglement_profile(
    results_dir: Path,
    checkpoint: str,
    system_prompt: str = "none",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Line plot of entanglement cos(v_ref, v_over) per layer,
    with boundary_margin on a twin axis.
    """
    safe_ckpt = checkpoint.replace("/", "_").replace(" ", "_")
    cache_path = results_dir / "geometry" / f"{safe_ckpt}__{system_prompt}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"No geometry file at {cache_path}")

    geom = load_geometry(cache_path)
    layers = geom.layer_indices
    ent    = geom.entanglement_curve
    margin = geom.margin_curve

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    color_ent    = "#1f77b4"
    color_margin = "#d62728"

    ax1.plot(layers, ent,    color=color_ent,    lw=2,   label="entanglement cos(v_ref, v_over)")
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Entanglement  cos(v_ref, v_over)", color=color_ent)
    ax1.tick_params(axis="y", labelcolor=color_ent)
    ax1.set_ylim(-1.05, 1.05)

    ax2.plot(layers, margin, color=color_margin, lw=2, ls=":", label="boundary margin")
    ax2.set_ylabel("Boundary margin (over-refusal side > 0)", color=color_margin)
    ax2.tick_params(axis="y", labelcolor=color_margin)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(f"Entanglement profile — {checkpoint}  |  system_prompt: {system_prompt}")
    fig.tight_layout()

    _save_or_show(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Evolution heatmap: layers × training stage
# ---------------------------------------------------------------------------

def plot_evolution_heatmap(
    results_dir: Path,
    system_prompt: str = "none",
    out_path: Path | None = None,
    metric: str = "entanglement",
) -> plt.Figure:
    """
    2-D heatmap  (rows = training stages in order, cols = layers)
    showing entanglement or boundary_margin.
    """
    df = _load_summary(results_dir, "exp2")
    if df is None:
        raise FileNotFoundError("Run Exp 2 first (--experiment evolution).")

    df = df[df["system_prompt"] == system_prompt]
    if df.empty:
        raise ValueError(f"No data for system_prompt='{system_prompt}'")

    # Sort checkpoints in post-training order
    def _order(tag: str) -> int:
        for i, stage in enumerate(POST_TRAINING_ORDER):
            if stage in tag.lower():
                return i
        return 99

    checkpoints = sorted(df["checkpoint"].unique(), key=_order)
    layers = sorted(df["layer"].unique())

    matrix = np.full((len(checkpoints), len(layers)), np.nan)
    for i, ckpt in enumerate(checkpoints):
        for j, layer in enumerate(layers):
            val = df[(df["checkpoint"] == ckpt) & (df["layer"] == layer)][metric]
            if len(val):
                matrix[i, j] = val.values[0]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.4), max(4, len(checkpoints) * 0.7)))

    if np.all(np.isnan(matrix)):
        raise ValueError("Entanglement matrix is entirely NaN — no valid geometry data found.")
    vabs = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    cmap = "RdBu_r" if metric == "entanglement" else "viridis"
    vmin, vmax = (-vabs, vabs) if metric == "entanglement" else (None, None)

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=layers,
        yticklabels=[c.split("__")[0] for c in checkpoints],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0 if metric == "entanglement" else None,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": metric},
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Training stage")
    ax.set_title(
        f"{'Entanglement' if metric=='entanglement' else 'Boundary margin'} "
        f"evolution during post-training  |  system_prompt: {system_prompt}"
    )
    # mark x-axis every 4 layers for readability
    step = max(1, len(layers) // 16)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels(layers[::step], rotation=0, fontsize=8)
    fig.tight_layout()

    _save_or_show(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — System prompt comparison: entanglement curves per system prompt
# ---------------------------------------------------------------------------

def plot_system_prompt_comparison(
    results_dir: Path,
    checkpoint: str | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    For each system prompt, plot the mean entanglement curve (across training
    stages if checkpoint is None, or for a specific checkpoint).
    """
    df = _load_summary(results_dir, "exp3")
    if df is None:
        raise FileNotFoundError("Run Exp 3 first (--experiment system_prompt).")

    if checkpoint:
        df = df[df["checkpoint"] == checkpoint]
        title = f"System-prompt influence — {checkpoint}"
    else:
        title = "System-prompt influence — mean across all checkpoints"

    system_prompts = df["system_prompt"].unique()
    layers = sorted(df["layer"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    for sp in system_prompts:
        sub = df[df["system_prompt"] == sp]
        mean_ent = [
            sub[sub["layer"] == l]["entanglement"].mean() for l in layers
        ]
        ax.plot(layers, mean_ent, label=sp, color=PALETTE.get(sp, "#333333"), lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entanglement  cos(v_ref, v_over)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title="System prompt", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()

    _save_or_show(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — Entanglement vs FP rate scatter (Exp 1 correlation)
# ---------------------------------------------------------------------------

def plot_entanglement_vs_fp(
    results_dir: Path,
    checkpoint: str,
    system_prompt: str = "none",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Scatter: x = entanglement per layer, y = FP rate (same for all layers
    within a checkpoint, so the scatter shows variance driven by layer).
    """
    corr_path = (
        results_dir / "geometry"
        / f"{checkpoint}__{system_prompt}_io_corr.csv"
    )
    if not corr_path.exists():
        raise FileNotFoundError(
            f"No I/O correlation CSV found at {corr_path}. "
            "Make sure run_experiment.py was run before run_representation_analysis.py."
        )

    df = pd.read_csv(corr_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, x_col, xlabel in [
        (axes[0], "entanglement",   "Entanglement  cos(v_ref, v_over)"),
        (axes[1], "boundary_margin","Boundary margin"),
    ]:
        ax.scatter(
            df[x_col], df["fp_rate"],
            c=df["layer"], cmap="viridis", alpha=0.8, s=40,
        )
        r = df[[x_col, "fp_rate"]].corr().iloc[0, 1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("FP rate (over-refusal)")
        ax.set_title(f"r = {r:.3f}")
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(df["layer"].min(), df["layer"].max()),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="layer")

    fig.suptitle(
        f"Geometry vs. I/O over-refusal rate — {checkpoint} | {system_prompt}",
        y=1.02,
    )
    fig.tight_layout()
    _save_or_show(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — Probe accuracy across layers (sanity check)
# ---------------------------------------------------------------------------

def plot_probe_accuracy(
    results_dir: Path,
    checkpoint: str,
    system_prompt: str = "none",
    out_path: Path | None = None,
) -> plt.Figure:
    safe_ckpt = checkpoint.replace("/", "_").replace(" ", "_")
    cache_path = results_dir / "geometry" / f"{safe_ckpt}__{system_prompt}.npz"
    geom = load_geometry(cache_path)

    layers   = geom.layer_indices
    acc_ref  = [lg.probe_ref_acc  for lg in geom.layers]
    acc_over = [lg.probe_over_acc for lg in geom.layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, acc_ref,  label="probe_ref  (harmful vs safe)",     lw=2)
    ax.plot(layers, acc_over, label="probe_over (over-refused vs safe)", lw=2, ls="--")
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.4, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Train accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.set_title(f"Probe accuracy — {checkpoint} | {system_prompt}")
    fig.tight_layout()
    _save_or_show(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, out_path: Path | None) -> None:
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure to %s", out_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument(
        "--plot",
        required=True,
        choices=[
            "entanglement_profile",
            "evolution_heatmap",
            "system_prompt_comparison",
            "entanglement_vs_fp",
            "probe_accuracy",
            "all",
        ],
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--system-prompt", default="none")
    p.add_argument("--metric", default="entanglement",
                   choices=["entanglement", "boundary_margin"],
                   help="Metric for evolution heatmap")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If set, save figures here instead of showing interactively")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    rd     = args.results_dir
    od     = args.out_dir
    ckpt   = args.checkpoint
    sp     = args.system_prompt

    def _out(name: str) -> Path | None:
        return (od / name) if od else None

    plots = (
        ["entanglement_profile", "evolution_heatmap",
         "system_prompt_comparison", "entanglement_vs_fp", "probe_accuracy"]
        if args.plot == "all"
        else [args.plot]
    )

    for plot_name in plots:
        try:
            if plot_name == "entanglement_profile":
                if not ckpt:
                    logger.warning("--checkpoint required for entanglement_profile, skipping.")
                    continue
                plot_entanglement_profile(rd, ckpt, sp, _out("entanglement_profile.png"))

            elif plot_name == "evolution_heatmap":
                plot_evolution_heatmap(rd, sp, _out("evolution_heatmap.png"), args.metric)

            elif plot_name == "system_prompt_comparison":
                plot_system_prompt_comparison(rd, ckpt, _out("system_prompt_comparison.png"))

            elif plot_name == "entanglement_vs_fp":
                if not ckpt:
                    logger.warning("--checkpoint required for entanglement_vs_fp, skipping.")
                    continue
                plot_entanglement_vs_fp(rd, ckpt, sp, _out("entanglement_vs_fp.png"))

            elif plot_name == "probe_accuracy":
                if not ckpt:
                    logger.warning("--checkpoint required for probe_accuracy, skipping.")
                    continue
                plot_probe_accuracy(rd, ckpt, sp, _out("probe_accuracy.png"))

        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", plot_name, e)


if __name__ == "__main__":
    main()