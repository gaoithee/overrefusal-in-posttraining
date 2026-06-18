"""
plot_2d_refusal_space.py

Proietta le attivazioni di OLMo2 su due direzioni:
  v_ref  = mean(h_harmful)  - mean(h_harmless)   # refusal corretto
  v_over = mean(h_pseudo)   - mean(h_harmless)   # over-refusal

Produce un grid di scatter plot 2D (righe=layer, colonne=checkpoint).
I colori rappresentano sempre i tre gruppi (harmful/pseudo_harm/harmless),
splittati per predicted_refusal cosi' da rendere palese il comportamento del modello.

Modalita' di split delle figure:
  --split-by none      : una figura sola (comportamento originale)
  --split-by source    : una figura per ogni valore di 'source'
  --split-by category  : una figura per ogni valore di 'category'

Le direzioni sono sempre calcolate sull'intero dataset del checkpoint
(non sul sottoinsieme), cosi' i plot restano comparabili tra loro.

Usage:
    python plot_2d_refusal_space.py \
        [--hf-dataset saracandu/olmo-activations] \
        [--hf-token <token>]           \
        [--position first_gen]         \
        [--layers 8 16 19 24 26 31]    \
        [--checkpoints base__none sft__none dpo__none final__none] \
        [--split-by none|source|category] \
        [--exclude-sources beavertails]    \
        [--orthogonalize]              \
        [--sample 500]                 \
        [--output-dir figures/]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ---------------------------------------------------------------------------
# Gruppi
# ---------------------------------------------------------------------------

def assign_group(df: pd.DataFrame) -> pd.Series:
    pseudo_sources = {"or_bench", "false_reject"}
    groups = pd.Series("harmless", index=df.index)
    groups[df["label"] == 1] = "harmful"
    groups[(df["label"] == 0) & (df["source"].isin(pseudo_sources))] = "pseudo_harm"
    return groups


# ---------------------------------------------------------------------------
# Caricamento da HuggingFace
# ---------------------------------------------------------------------------

def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    token_path = Path("~/.hf_token").expanduser()
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def load_full_dataset(hf_dataset, layers, position, token, checkpoints, extra_cols=None):
    """
    Carica ogni checkpoint separatamente per evitare CastError da schema misto
    (es. base__none ha 3 post_instr, instruct ne hanno 7).
    Le colonne post_instr_* vengono scartate da select_columns — non servono per il plot 2D.

    extra_cols: colonne aggiuntive da includere (es. ['category'])
    """
    from datasets import load_dataset

    base_cols = ["label", "source", "checkpoint", "predicted_refusal"]
    if extra_cols:
        base_cols += [c for c in extra_cols if c not in base_cols]
    act_cols = [f"layer_{l}_{position}" for l in layers]
    needed_cols = base_cols + act_cols

    print(f"[.] Carico dataset '{hf_dataset}' per {len(checkpoints)} checkpoint ...")
    dfs = []
    for ckpt in checkpoints:
        try:
            ds = load_dataset(
                hf_dataset,
                data_files={"train": f"data/{ckpt}/*.parquet"},
                split="train",
                token=token,
            )
            available = [c for c in needed_cols if c in ds.column_names]
            missing   = [c for c in needed_cols if c not in ds.column_names]
            if missing:
                print(f"    [warn] {ckpt}: colonne mancanti: {missing}")
            df = ds.select_columns(available).to_pandas()
            print(f"  {ckpt}: {len(df)} righe")
            dfs.append(df)
        except Exception as e:
            print(f"    [error] {ckpt}: {e}")

    if not dfs:
        return None
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"ok ({len(df_full)} righe totali)")
    return df_full


# ---------------------------------------------------------------------------
# Geometria  (direzioni calcolate sull'intero checkpoint)
# ---------------------------------------------------------------------------

def compute_directions(h_harmful, h_pseudo, h_harmless, orthogonalize=False):
    mu_harmful  = h_harmful.mean(axis=0)
    mu_pseudo   = h_pseudo.mean(axis=0)
    mu_harmless = h_harmless.mean(axis=0)

    v_ref  = mu_harmful - mu_harmless
    v_over = mu_pseudo  - mu_harmless

    cd_ref  = float(np.linalg.norm(v_ref))
    cd_over = float(np.linalg.norm(v_over))

    v_ref_hat  = v_ref  / (cd_ref  + 1e-12)
    v_over_hat = v_over / (cd_over + 1e-12)

    if orthogonalize:
        v_over_hat = v_over_hat - np.dot(v_over_hat, v_ref_hat) * v_ref_hat
        norm_orth = float(np.linalg.norm(v_over_hat))
        if norm_orth < 1e-8:
            print("  [warn] v_over quasi parallelo a v_ref dopo Gram-Schmidt")
        else:
            v_over_hat = v_over_hat / norm_orth

    return v_ref_hat, v_over_hat, cd_ref, cd_over


def project_2d(h, v_ref_hat, v_over_hat, cd_ref, cd_over, mu_harmless):
    h_c = h - mu_harmless
    x = (h_c @ v_ref_hat)  / (cd_ref  / 2.0 + 1e-12)
    y = (h_c @ v_over_hat) / (cd_over / 2.0 + 1e-12)
    return x, y


# ---------------------------------------------------------------------------
# Colori e stili
# ---------------------------------------------------------------------------

# Colori per (group, predicted_refusal)
POINT_COLOR = {
    ("harmful",     1): "#e63946",   # rosso vivo     — harmful rifiutato (corretto)
    ("harmful",     0): "#ffb3b3",   # rosa chiaro    — harmful accettato (errore sicurezza)
    ("pseudo_harm", 1): "#f4a261",   # arancione      — pseudo-harmful rifiutato (over-refusal)
    ("pseudo_harm", 0): "#2a9d8f",   # verde          — pseudo-harmful accettato (corretto)
    ("harmless",    1): "#9b2226",   # bordeaux       — harmless rifiutato (over-refusal grave)
    ("harmless",    0): "#457b9d",   # blu            — harmless accettato (corretto)
}

POINT_MARKER = {
    "harmful":     "x",
    "pseudo_harm": "^",
    "harmless":    "o",
}

CENTROID_STYLE = dict(s=130, edgecolors="black", linewidths=1.3, zorder=5)


# ---------------------------------------------------------------------------
# Calcolo proiezioni per un checkpoint e layer
# ---------------------------------------------------------------------------

def compute_projections_for_layer(df_ckpt, layer, position, orthogonalize, sample, rng,
                                   exclude_from_directions=None):
    """
    Ritorna un dict con:
      'geometry': (v_ref_hat, v_over_hat, cd_ref, cd_over, mu_harmless, entanglement)
      'proj': DataFrame con colonne x, y, group, source, category (se presente)
    oppure None se i dati non bastano.

    exclude_from_directions: set di source da escludere nel calcolo di v_ref e v_over.
    I punti esclusi vengono comunque proiettati e plottati (con marker diverso).
    """
    col = f"layer_{layer}_{position}"
    if col not in df_ckpt.columns:
        return None

    acts = np.stack(
        df_ckpt[col].apply(lambda v: np.asarray(v, dtype=np.float32)).values
    )

    # Maschera per il calcolo delle direzioni (esclude le source indicate)
    if exclude_from_directions:
        incl = ~df_ckpt["source"].isin(exclude_from_directions)
    else:
        incl = pd.Series(True, index=df_ckpt.index)

    mask_h  = (df_ckpt["group"] == "harmful")  & incl
    mask_p  = (df_ckpt["group"] == "pseudo_harm") & incl
    mask_hl = (df_ckpt["group"] == "harmless") & incl

    mask_h  = mask_h.values
    mask_p  = mask_p.values
    mask_hl = mask_hl.values

    h_harmful  = acts[mask_h]
    h_pseudo   = acts[mask_p]
    h_harmless = acts[mask_hl]

    if min(len(h_harmful), len(h_pseudo), len(h_harmless)) < 10:
        return None

    v_ref_hat, v_over_hat, cd_ref, cd_over = compute_directions(
        h_harmful, h_pseudo, h_harmless, orthogonalize=orthogonalize
    )
    mu_harmless = h_harmless.mean(axis=0)

    v_ref_raw  = h_harmful.mean(axis=0) - mu_harmless
    v_over_raw = h_pseudo.mean(axis=0)  - mu_harmless
    entanglement = float(
        np.dot(v_ref_raw, v_over_raw)
        / (np.linalg.norm(v_ref_raw) * np.linalg.norm(v_over_raw) + 1e-12)
    )

    # Proietta tutti i punti (campionamento opzionale per plot)
    if sample:
        idx_sample = rng.choice(len(acts), min(sample * 3, len(acts)), replace=False)
        acts_plot  = acts[idx_sample]
        df_plot    = df_ckpt.iloc[idx_sample].reset_index(drop=True)
    else:
        acts_plot = acts
        df_plot   = df_ckpt.reset_index(drop=True)

    x, y = project_2d(acts_plot, v_ref_hat, v_over_hat, cd_ref, cd_over, mu_harmless)

    proj = df_plot[["group", "source", "predicted_refusal"] +
                   (["category"] if "category" in df_plot.columns else [])].copy()
    proj["x"] = x
    proj["y"] = y
    # Marca i punti esclusi dal calcolo delle direzioni
    if exclude_from_directions:
        proj["excluded"] = df_plot["source"].isin(exclude_from_directions).values
    else:
        proj["excluded"] = False

    # Centroidi sull'intero dataset (non campionato)
    centroids = {}
    for g, h in [("harmful", h_harmful), ("pseudo_harm", h_pseudo), ("harmless", h_harmless)]:
        xc, yc = project_2d(h.mean(axis=0, keepdims=True),
                             v_ref_hat, v_over_hat, cd_ref, cd_over, mu_harmless)
        centroids[g] = (float(xc[0]), float(yc[0]))

    return {
        "entanglement": entanglement,
        "proj":         proj,
        "centroids":    centroids,
    }


# ---------------------------------------------------------------------------
# Legenda
# ---------------------------------------------------------------------------

def make_legend_handles():
    entries = [
        ("harmful — rifiutato ✓",              ("harmful",     1), "x"),
        ("harmful — accettato (miss)",          ("harmful",     0), "x"),
        ("pseudo-harmful — rifiutato (over-r)", ("pseudo_harm", 1), "^"),
        ("pseudo-harmful — accettato ✓",        ("pseudo_harm", 0), "^"),
        ("harmless — rifiutato (over-r grave)", ("harmless",    1), "o"),
        ("harmless — accettato ✓",              ("harmless",    0), "o"),
    ]
    handles = []
    for label, key, marker in entries:
        handles.append(plt.scatter([], [], color=POINT_COLOR[key], marker=marker,
                                   alpha=0.8, s=35, label=label))
    # centroide
    handles.append(plt.scatter([], [], color="gray", marker="*", s=80,
                               edgecolors="black", linewidths=1, label="centroide"))
    return handles


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_grid(
    results,
    checkpoints,
    layers,
    orthogonalize,
    output_path,
    position,
    subset_label,
    subset_mask_fn,
    excl_label="",
):
    n_rows = len(layers)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.1 * n_rows),
                             squeeze=False)
    orth_label = "ortho" if orthogonalize else "naive"

    for r, layer in enumerate(layers):
        for c, ckpt in enumerate(checkpoints):
            ax = axes[r][c]
            key = (ckpt, layer)
            if key not in results:
                ax.set_visible(False)
                continue

            data = results[key]
            proj = data["proj"]

            if subset_mask_fn is not None:
                mask = subset_mask_fn(proj)
                proj_show = proj[mask]
            else:
                proj_show = proj

            # --- Scatter: split per (group, predicted_refusal) ---
            for group in ["harmless", "pseudo_harm", "harmful"]:
                sub = proj_show[proj_show["group"] == group]
                if len(sub) == 0:
                    continue

                sub_in  = sub[~sub["excluded"]] if "excluded" in sub.columns else sub
                sub_out = sub[sub["excluded"]]  if "excluded" in sub.columns else sub.iloc[0:0]

                if "predicted_refusal" in sub_in.columns:
                    for refused_val in [0, 1]:
                        part = sub_in[sub_in["predicted_refusal"] == refused_val]
                        if len(part) == 0:
                            continue
                        color = POINT_COLOR[(group, refused_val)]
                        ax.scatter(part["x"], part["y"],
                                   color=color,
                                   marker=POINT_MARKER[group],
                                   alpha=0.5, s=18, linewidths=0.5)
                else:
                    if len(sub_in) > 0:
                        ax.scatter(sub_in["x"], sub_in["y"],
                                   color=POINT_COLOR[(group, 0)],
                                   marker=POINT_MARKER[group],
                                   alpha=0.4, s=13, linewidths=0.5)

                # Punti esclusi dalle direzioni → grigio
                if len(sub_out) > 0:
                    ax.scatter(sub_out["x"], sub_out["y"],
                               color="gray",
                               marker=POINT_MARKER[group],
                               alpha=0.2, s=10, linewidths=0.5)

            # Centroidi (stelle)
            for group, (cx, cy) in data["centroids"].items():
                ax.scatter(cx, cy, color=POINT_COLOR[(group, 0)],
                           marker="*", **CENTROID_STYLE)

            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")

            if r == 0:
                ax.set_title(ckpt.replace("__", "\n"), fontsize=8, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"layer {layer}\nv_over ->", fontsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("v_ref ->", fontsize=7)
            ax.tick_params(labelsize=6)

            ent = data.get("entanglement")
            if ent is not None:
                ax.text(0.97, 0.03, f"ent={ent:.2f}",
                        transform=ax.transAxes,
                        fontsize=6, ha="right", va="bottom", color="gray")

            ax.text(0.03, 0.97, f"n={len(proj_show)}",
                    transform=ax.transAxes,
                    fontsize=6, ha="left", va="top", color="gray")

    fig.legend(handles=make_legend_handles(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    excl_note = f"  |  escluso da direzioni: {excl_label}" if excl_label else ""
    fig.suptitle(
        f"OLMo2 -- 2D refusal space  |  {subset_label}  |  {position}  |  {orth_label}{excl_note}\n"
        f"x = v_ref (harmful-harmless),  y = v_over (pseudo_harm-harmless)"
        f"  [grigio = escluso dal calcolo direzioni]\n"
        f"colori = (gruppo, predicted_refusal):  ✓ = corretto  |  over-r = over-refusal  |  miss = harmful accettato",
        fontsize=8, y=1.01
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  [ok] {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset",    default="saracandu/olmo-activations")
    parser.add_argument("--hf-token",      default=None)
    parser.add_argument("--position",      default="first_gen",
                        choices=["first_gen", "last_prompt"])
    parser.add_argument("--layers",        nargs="+", type=int,
                        default=[8, 16, 19, 24, 26, 31])
    parser.add_argument("--checkpoints",   nargs="+",
                        default=["base__none", "sft__none", "dpo__none", "final__none"])
    parser.add_argument("--split-by",      default="none",
                        choices=["none", "source", "category"],
                        help="Produce una figura separata per ogni valore del campo scelto")
    parser.add_argument("--exclude-from-directions", nargs="*", default=None,
                        metavar="SOURCE",
                        help="Source da escludere dal calcolo di v_ref e v_over "
                             "(es. --exclude-from-directions beavertails). "
                             "I punti vengono comunque plottati in grigio.")
    parser.add_argument("--exclude-sources", nargs="*", default=None,
                        metavar="SOURCE",
                        help="Source da escludere completamente dal dataset "
                             "(es. --exclude-sources beavertails). "
                             "I punti non vengono ne' calcolati ne' plottati.")
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--sample",        type=int, default=None)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output-dir",    default="figures/")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    token = load_hf_token(args.hf_token)

    extra_cols = ["category"] if args.split_by == "category" else []

    df_full = load_full_dataset(
        args.hf_dataset, args.layers, args.position, token,
        checkpoints=args.checkpoints, extra_cols=extra_cols
    )
    if df_full is None:
        return

    df_full["group"] = assign_group(df_full)

    if args.exclude_sources:
        before = len(df_full)
        df_full = df_full[~df_full["source"].isin(args.exclude_sources)].reset_index(drop=True)
        print(f"[.] Escluse source {args.exclude_sources}: {before} -> {len(df_full)} righe")

    results = {}
    for ckpt in args.checkpoints:
        print(f"\n[.] Checkpoint: {ckpt}")
        df_ckpt = df_full[df_full["checkpoint"] == ckpt].reset_index(drop=True)
        if len(df_ckpt) == 0:
            print(f"    [warn] Checkpoint non trovato. Disponibili: "
                  f"{sorted(df_full['checkpoint'].unique())}")
            continue
        print(f"    gruppi: {df_ckpt['group'].value_counts().to_dict()}")

        for layer in args.layers:
            print(f"  layer {layer} ...", end=" ", flush=True)
            excl = set(args.exclude_from_directions) if args.exclude_from_directions else None
            res = compute_projections_for_layer(
                df_ckpt, layer, args.position, args.orthogonalize, args.sample, rng,
                exclude_from_directions=excl,
            )
            if res is None:
                print("salto (dati insufficienti o colonna mancante).")
                continue
            results[(ckpt, layer)] = res
            print(f"ok  (ent={res['entanglement']:.3f})")

    if not results:
        print("\n[!] Nessun risultato calcolato.")
        return

    orth_tag = "_ortho" if args.orthogonalize else "_naive"

    if args.split_by == "none":
        subsets = [("all", None)]

    elif args.split_by == "source":
        sources = sorted(df_full["source"].dropna().unique())
        subsets = [(f"source={s}", lambda proj, s=s: proj["source"] == s)
                   for s in sources]
        subsets = [("all_sources", None)] + subsets

    elif args.split_by == "category":
        if "category" not in df_full.columns:
            print("[!] Colonna 'category' non presente nel dataset.")
            return
        categories = sorted(df_full["category"].dropna().unique())
        print(f"\n[.] Categorie trovate ({len(categories)}): {categories}")
        subsets = [(f"category={c}", lambda proj, c=c: proj["category"] == c)
                   for c in categories]
        subsets = [("all_categories", None)] + subsets

    print(f"\n[.] Genero {len(subsets)} figure in '{output_dir}' ...")
    for label, mask_fn in subsets:
        safe_label = label.replace("=", "_").replace(" ", "_").replace("/", "-")
        fname = f"2d_{args.position}{orth_tag}_{safe_label}.png"
        excl_label = ",".join(sorted(args.exclude_from_directions)) \
                     if args.exclude_from_directions else ""
        plot_grid(
            results,
            checkpoints=args.checkpoints,
            layers=args.layers,
            orthogonalize=args.orthogonalize,
            output_path=output_dir / fname,
            position=args.position,
            subset_label=label,
            subset_mask_fn=mask_fn,
            excl_label=excl_label,
        )

    print("\n[ok] Fatto.")


if __name__ == "__main__":
    main()