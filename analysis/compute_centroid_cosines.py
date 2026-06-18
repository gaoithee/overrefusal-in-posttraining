"""
compute_centroid_cosines.py

Per ogni layer e categoria (harmful, pseudo_harm, harmless),
calcola il coseno tra i centroidi delle attivazioni first_gen
tra coppie di checkpoint.

Mostra quanto la geometria cambia tra base→SFT vs SFT→DPO→Final.

Usage:
    python compute_centroid_cosines.py
    python compute_centroid_cosines.py --layer 19 24 31
    python compute_centroid_cosines.py --exclude-sources beavertails
    python compute_centroid_cosines.py --out results/olmo2/geometry/centroid_cosines.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

HF_REPO = "saracandu/olmo-activations"
PSEUDO_SOURCES = {"or_bench", "false_reject"}
CHECKPOINTS = ["base__none", "sft__none", "dpo__none", "final__none"]
LAYERS = [8, 16, 19, 24, 26, 31]
CATEGORIES = ["harmful", "pseudo_harm", "harmless"]
PAIRS = [
    ("base__none", "sft__none",   "base-sft"),
    ("base__none", "dpo__none",   "base-dpo"),
    ("base__none", "final__none", "base-fin"),
    ("sft__none",  "dpo__none",   "sft-dpo"),
    ("sft__none",  "final__none", "sft-fin"),
    ("dpo__none",  "final__none", "dpo-fin"),
]


def assign_3cat(row):
    if row["label"] == 1:
        return "harmful"
    if row["source"] in PSEUDO_SOURCES:
        return "pseudo_harm"
    return "harmless"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--out", default="results/olmo2/geometry/centroid_cosines.csv")
    parser.add_argument("--exclude-sources", nargs="*", default=None,
                        metavar="SOURCE",
                        help="Source da escludere completamente "
                             "(es. --exclude-sources beavertails)")
    args = parser.parse_args()

    exclude = set(args.exclude_sources) if args.exclude_sources else None

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    rows = []

    print(f"\n{'cat':>12}  {'L':>3}  {'base-sft':>9}  {'base-dpo':>9}  "
          f"{'base-fin':>9}  {'sft-dpo':>8}  {'sft-fin':>8}  {'dpo-fin':>8}")

    for layer in args.layers:
        col = f"layer_{layer}_first_gen"

        # Carica centroidi per tutti i checkpoint
        centroids = {}
        for ckpt in CHECKPOINTS:
            ds = load_dataset(
                args.hf_repo,
                data_files={"train": f"data/{ckpt}/*.parquet"},
                split="train",
                token=token,
            )
            df = ds.select_columns(["label", "source", col]).to_pandas()

            # Escludi source se richiesto
            if exclude:
                before = len(df)
                df = df[~df["source"].isin(exclude)].reset_index(drop=True)
                print(f"    [exclude] {ckpt} layer {layer}: {before} -> {len(df)} righe")

            df["cat3"] = df.apply(assign_3cat, axis=1)
            centroids[ckpt] = {
                cat: np.stack(df[df["cat3"] == cat][col].values).astype(np.float32).mean(0)
                for cat in CATEGORIES
            }

        # Calcola coseni per ogni coppia e categoria
        for cat in CATEGORIES:
            pair_cos = {}
            for ckpt1, ckpt2, label in PAIRS:
                c = cosine(centroids[ckpt1][cat], centroids[ckpt2][cat])
                pair_cos[label] = c
                rows.append({
                    "layer":    layer,
                    "category": cat,
                    "pair":     label,
                    "cosine":   c,
                })

            print(f"{cat:>12}  {layer:>3}  "
                  f"{pair_cos['base-sft']:>9.4f}  "
                  f"{pair_cos['base-dpo']:>9.4f}  "
                  f"{pair_cos['base-fin']:>9.4f}  "
                  f"{pair_cos['sft-dpo']:>8.4f}  "
                  f"{pair_cos['sft-fin']:>8.4f}  "
                  f"{pair_cos['dpo-fin']:>8.4f}")

        print()

    # Salva CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Salvato in {out}")


if __name__ == "__main__":
    main()