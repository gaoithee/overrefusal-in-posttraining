"""
plot_entanglement_curves.py

Produce una figura per ogni combinazione token_position x method,
più una figura riassuntiva. Ogni figura ha due pannelli: entanglement e boundary_margin.
Solo checkpoints __none.

Uso:
    python plot_entanglement_curves.py \
        --csv-last-meandiff  results/olmo2/geometry/ent_last_meandiff.csv \
        --csv-last-logistic  results/olmo2/geometry/ent_last_logistic.csv \
        --csv-first-meandiff results/olmo2/geometry/ent_first_meandiff.csv \
        --csv-first-logistic results/olmo2/geometry/ent_first_logistic.csv \
        --out results/olmo2/geometry/plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

CHECKPOINT_ORDER = [
    "base__none",
    "sft__none",
    "dpo__none",
    "final__none",
]

STAGE_COLORS = {
    "base__none":  "#4878CF",
    "sft__none":   "#D65F5F",
    "dpo__none":   "#6ACC65",
    "final__none": "#B47CC7",
}

LABELS = {
    "base__none":  "Base",
    "sft__none":   "SFT",
    "dpo__none":   "DPO",
    "final__none": "Final",
}

COMBO_TITLES = {
    "last_meandiff":  "last prompt token  ·  mean difference",
    "last_logistic":  "last prompt token  ·  logistic probe",
    "first_meandiff": "first generated token  ·  mean difference",
    "first_logistic": "first generated token  ·  logistic probe",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_none(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["checkpoint"].isin(CHECKPOINT_ORDER)].copy()


def plot_panel(df: pd.DataFrame, metric: str, ax: plt.Axes, ylabel: str, title: str = ""):
    for ckpt in CHECKPOINT_ORDER:
        sub = df[df["checkpoint"] == ckpt].sort_values("layer")
        if sub.empty or sub[metric].isna().all():
            continue
        ax.plot(
            sub["layer"], sub[metric],
            color=STAGE_COLORS[ckpt], linewidth=2.2,
            marker="o", markersize=6,
            label=LABELS[ckpt],
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)


def save(fig: plt.Figure, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"  → {path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-last-meandiff",  default=None)
    parser.add_argument("--csv-last-logistic",  default=None)
    parser.add_argument("--csv-first-meandiff", default=None)
    parser.add_argument("--csv-first-logistic", default=None)
    parser.add_argument("--out", default="results/olmo2/geometry/plots/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_paths = {
        "last_meandiff":  args.csv_last_meandiff,
        "last_logistic":  args.csv_last_logistic,
        "first_meandiff": args.csv_first_meandiff,
        "first_logistic": args.csv_first_logistic,
    }
    dfs = {k: filter_none(pd.read_csv(v)) for k, v in combo_paths.items() if v}

    if not dfs:
        print("Nessun CSV fornito.")
        return

    print("Generating figures...")

    # -----------------------------------------------------------------------
    # Figure 1-4: una per combinazione, due pannelli (entanglement + margin)
    # -----------------------------------------------------------------------
    for combo, df in dfs.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"OLMo2 — {COMBO_TITLES[combo]}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plot_panel(df, "entanglement",    axes[0],
                   ylabel="cos(v_ref, f)",
                   title="Entanglement")
        plot_panel(df, "boundary_margin", axes[1],
                   ylabel="signed distance from boundary",
                   title="Boundary margin (pseudo-harmful)")
        plt.tight_layout()
        save(fig, out_dir / f"fig_{combo}.pdf")

    # -----------------------------------------------------------------------
    # Figure 5: entanglement a confronto tra le 4 combinazioni (2x2 grid)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    fig.suptitle("Entanglement per layer — tutte le combinazioni",
                 fontsize=13, fontweight="bold")
    for ax, (combo, df) in zip(axes.flat, dfs.items()):
        plot_panel(df, "entanglement", ax,
                   ylabel="cos(v_ref, f)",
                   title=COMBO_TITLES[combo])
    plt.tight_layout()
    save(fig, out_dir / "fig_entanglement_all_combos.pdf")

    # -----------------------------------------------------------------------
    # Figure 6: boundary margin a confronto tra le 4 combinazioni (2x2 grid)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    fig.suptitle("Boundary margin per layer — tutte le combinazioni",
                 fontsize=13, fontweight="bold")
    for ax, (combo, df) in zip(axes.flat, dfs.items()):
        plot_panel(df, "boundary_margin", ax,
                   ylabel="signed distance",
                   title=COMBO_TITLES[combo])
    plt.tight_layout()
    save(fig, out_dir / "fig_boundary_all_combos.pdf")

    # -----------------------------------------------------------------------
    # Figure 7: confronto last vs first, solo mean_diff, entanglement
    # -----------------------------------------------------------------------
    if "last_meandiff" in dfs and "first_meandiff" in dfs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("last_prompt vs first_gen  ·  mean difference",
                     fontsize=13, fontweight="bold")
        plot_panel(dfs["last_meandiff"],  "entanglement", axes[0],
                   ylabel="cos(v_ref, f)", title="last prompt token")
        plot_panel(dfs["first_meandiff"], "entanglement", axes[1],
                   ylabel="cos(v_ref, f)", title="first generated token")
        plt.tight_layout()
        save(fig, out_dir / "fig_last_vs_first_entanglement.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("last_prompt vs first_gen  ·  mean difference — boundary margin",
                     fontsize=13, fontweight="bold")
        plot_panel(dfs["last_meandiff"],  "boundary_margin", axes[0],
                   ylabel="signed distance", title="last prompt token")
        plot_panel(dfs["first_meandiff"], "boundary_margin", axes[1],
                   ylabel="signed distance", title="first generated token")
        plt.tight_layout()
        save(fig, out_dir / "fig_last_vs_first_margin.pdf")

    # -----------------------------------------------------------------------
    # Figure 8: confronto mean_diff vs logistic, solo first_gen
    # -----------------------------------------------------------------------
    if "first_meandiff" in dfs and "first_logistic" in dfs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("first_gen  ·  mean_diff vs logistic — entanglement",
                     fontsize=13, fontweight="bold")
        plot_panel(dfs["first_meandiff"], "entanglement", axes[0],
                   ylabel="cos(v_ref, f)", title="mean difference")
        plot_panel(dfs["first_logistic"], "entanglement", axes[1],
                   ylabel="cos(v_ref, f)", title="logistic probe")
        plt.tight_layout()
        save(fig, out_dir / "fig_meandiff_vs_logistic_entanglement.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("first_gen  ·  mean_diff vs logistic — boundary margin",
                     fontsize=13, fontweight="bold")
        plot_panel(dfs["first_meandiff"], "boundary_margin", axes[0],
                   ylabel="signed distance", title="mean difference")
        plot_panel(dfs["first_logistic"], "boundary_margin", axes[1],
                   ylabel="signed distance", title="logistic probe")
        plt.tight_layout()
        save(fig, out_dir / "fig_meandiff_vs_logistic_margin.pdf")

    # -----------------------------------------------------------------------
    # Tabelle numeriche
    # -----------------------------------------------------------------------
    print("\n=== Tabelle numeriche ===")
    for combo, df in dfs.items():
        print(f"\n--- {COMBO_TITLES[combo]} ---")
        pivot_ent = df.pivot(index="layer", columns="checkpoint",
                             values="entanglement")
        pivot_ent = pivot_ent.reindex(
            columns=[c for c in CHECKPOINT_ORDER if c in pivot_ent.columns]
        )
        pivot_mar = df.pivot(index="layer", columns="checkpoint",
                             values="boundary_margin")
        pivot_mar = pivot_mar.reindex(
            columns=[c for c in CHECKPOINT_ORDER if c in pivot_mar.columns]
        )
        print("Entanglement:")
        print(pivot_ent.round(4).to_string())
        print("Boundary margin:")
        print(pivot_mar.round(4).to_string())

    print(f"\nTutti i file in: {out_dir}")


if __name__ == "__main__":
    main()
