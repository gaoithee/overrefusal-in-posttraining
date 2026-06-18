"""
run_classification.py

Probe logistica 3 categorie (harmful / pseudo_harm / harmless)
con cross-val per checkpoint e cross-checkpoint transfer.

Usage:
    python run_classification.py
    python run_classification.py --exclude-sources beavertails
    python run_classification.py --out results/olmo2/classifiers/clf3_results.csv
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

CHECKPOINTS = ["base__none", "sft__none", "dpo__none", "final__none"]
LAYERS = [8, 16, 19, 24, 26, 31]
PSEUDO_SOURCES = {"or_bench", "false_reject"}

SAVE_DIR = Path("results/olmo2/classifiers")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def assign_3cat(row):
    if row["label"] == 1:
        return "harmful"
    if row["source"] in PSEUDO_SOURCES:
        return "pseudo_harm"
    return "harmless"


def load_checkpoint(hf_repo, ckpt, cols, token, exclude_sources=None):
    ds = load_dataset(
        hf_repo,
        data_files={"train": f"data/{ckpt}/*.parquet"},
        split="train",
        token=token,
    )
    available = [c for c in cols if c in ds.column_names]
    df = ds.select_columns(available).to_pandas()
    if exclude_sources:
        before = len(df)
        df = df[~df["source"].isin(exclude_sources)].reset_index(drop=True)
        print(f"    [exclude] {ckpt}: {before} -> {len(df)} righe (rimosso {exclude_sources})")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="saracandu/olmo-activations")
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--out", default="results/olmo2/classifiers/clf3_results.csv")
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

    # -----------------------------------------------------------------------
    # Phase 1: train per checkpoint, cross-val + save
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1 — Train per checkpoint (cross-val)")
    print("=" * 70)

    for ckpt in CHECKPOINTS:
        act_cols = [f"layer_{l}_first_gen" for l in args.layers]
        df = load_checkpoint(args.hf_repo, ckpt, ["label", "source"] + act_cols, token, exclude)
        df["cat3"] = df.apply(assign_3cat, axis=1)

        print(f"\n{ckpt}")
        print(f"  {'layer':>7}  {'acc_3class':>10}  {'acc_pseudo_vs_harm':>18}  {'acc_pseudo_vs_harmless':>22}")

        for l in args.layers:
            col = f"layer_{l}_first_gen"
            X = np.stack(df[col].values).astype(np.float32)

            le = LabelEncoder()
            y_enc = le.fit_transform(df["cat3"].values)

            clf3 = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            acc_3 = cross_val_score(clf3, X, y_enc, cv=5, scoring="accuracy").mean()

            mask_ph = df["cat3"].isin(["pseudo_harm", "harmful"])
            X_ph = X[mask_ph]
            y_ph = (df["cat3"][mask_ph] == "harmful").astype(int).values
            clf_ph = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            acc_ph = cross_val_score(clf_ph, X_ph, y_ph, cv=5, scoring="accuracy").mean()

            mask_pl = df["cat3"].isin(["pseudo_harm", "harmless"])
            X_pl = X[mask_pl]
            y_pl = (df["cat3"][mask_pl] == "harmless").astype(int).values
            clf_pl = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            acc_pl = cross_val_score(clf_pl, X_pl, y_pl, cv=5, scoring="accuracy").mean()

            print(f"  layer {l:2d}  {acc_3:>10.3f}  {acc_ph:>18.3f}  {acc_pl:>22.3f}")

            rows.append({
                "phase":                  "within",
                "train_on":               ckpt,
                "test_on":                ckpt,
                "layer":                  l,
                "acc_3class":             acc_3,
                "acc_pseudo_vs_harmful":  acc_ph,
                "acc_pseudo_vs_harmless": acc_pl,
            })

            # Salva clf addestrato su tutti i dati
            clf3_full = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            clf3_full.fit(X, y_enc)
            save_path = SAVE_DIR / f"clf3_{ckpt}_layer{l}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({"clf": clf3_full, "le": le, "trained_on": ckpt, "layer": l}, f)

    # -----------------------------------------------------------------------
    # Phase 2: cross-checkpoint transfer — train on base, test on others
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 — Cross-checkpoint transfer (trained on base__none)")
    print("=" * 70)

    print(f"\n  {'layer':>7}", end="")
    for ckpt in CHECKPOINTS:
        print(f"  {ckpt:>20}", end="")
    print()

    for l in args.layers:
        clf_path = SAVE_DIR / f"clf3_base__none_layer{l}.pkl"
        with open(clf_path, "rb") as f:
            saved = pickle.load(f)
        clf_base = saved["clf"]
        le_base  = saved["le"]

        print(f"  layer {l:2d}", end="")

        for ckpt in CHECKPOINTS:
            df = load_checkpoint(
                args.hf_repo, ckpt,
                ["label", "source", f"layer_{l}_first_gen"],
                token, exclude
            )
            df["cat3"] = df.apply(assign_3cat, axis=1)

            X = np.stack(df[f"layer_{l}_first_gen"].values).astype(np.float32)
            y_enc = le_base.transform(df["cat3"].values)

            acc = clf_base.score(X, y_enc)
            print(f"  {acc:>20.3f}", end="")

            rows.append({
                "phase":                  "cross",
                "train_on":               "base__none",
                "test_on":                ckpt,
                "layer":                  l,
                "acc_3class":             acc,
                "acc_pseudo_vs_harmful":  None,
                "acc_pseudo_vs_harmless": None,
            })

        print()

    # Salva CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSalvato in {out}")
    print(f"Classifiers in {SAVE_DIR}")


if __name__ == "__main__":
    main()