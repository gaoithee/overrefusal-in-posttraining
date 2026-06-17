"""
compute_behavioral_probe.py

Addestra una probe logistica che predice predicted_refusal (0/1)
dalle attivazioni first_gen, separatamente per ogni checkpoint e layer.

Poi fa cross-checkpoint transfer: addestra sul base e testa su SFT/DPO/Final.

Se la probe semantica (cat3) è stabile tra checkpoint ma quella comportamentale
migliora da SFT a DPO, la dissociazione geometria/comportamento è dimostrata
su due probe indipendenti.

Usage:
    python compute_behavioral_probe.py
    python compute_behavioral_probe.py --out results/olmo2/geometry/behavioral_probe.csv
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

HF_REPO = "saracandu/olmo-activations"
PSEUDO_SOURCES = {"or_bench", "false_reject"}
CHECKPOINTS = ["base__none", "sft__none", "dpo__none", "final__none"]
LAYERS = [8, 16, 19, 24, 26, 31]

SAVE_DIR = Path("results/olmo2/classifiers")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--out", default="results/olmo2/geometry/behavioral_probe.csv")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    rows = []

    # -----------------------------------------------------------------------
    # Phase 1: train per checkpoint, cross-val accuracy
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("PHASE 1 — Behavioral probe per checkpoint (cross-val)")
    print("=" * 65)
    print(f"\n{'checkpoint':>20}  {'layer':>6}  {'acc_refusal':>12}  {'acc_pseudo_only':>16}")

    for ckpt in CHECKPOINTS:
        act_cols = [f"layer_{l}_first_gen" for l in args.layers]
        ds = load_dataset(
            args.hf_repo,
            data_files={"train": f"data/{ckpt}/*.parquet"},
            split="train",
            token=token,
        )
        df = ds.select_columns(["label", "source", "predicted_refusal"] + act_cols).to_pandas()

        # Subset pseudo-harmful only
        pseudo_mask = (df["label"] == 0) & (df["source"].isin(PSEUDO_SOURCES))

        for l in args.layers:
            col = f"layer_{l}_first_gen"

            # All prompts — predice predicted_refusal
            X_all = np.stack(df[col].values).astype(np.float32)
            y_all = df["predicted_refusal"].values.astype(int)
            clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            acc_all = cross_val_score(clf, X_all, y_all, cv=5, scoring="accuracy").mean()

            # Pseudo-harmful only — predice se vengono rifiutati
            X_ps = np.stack(df[pseudo_mask][col].values).astype(np.float32)
            y_ps = df[pseudo_mask]["predicted_refusal"].values.astype(int)
            clf_ps = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            acc_ps = cross_val_score(clf_ps, X_ps, y_ps, cv=5, scoring="accuracy").mean()

            print(f"{ckpt:>20}  {l:>6}  {acc_all:>12.3f}  {acc_ps:>16.3f}")

            rows.append({
                "phase":      "within",
                "train_on":   ckpt,
                "test_on":    ckpt,
                "layer":      l,
                "acc_all":    acc_all,
                "acc_pseudo": acc_ps,
            })

            # Salva clf addestrato su tutti i dati
            clf_full = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            clf_full.fit(X_all, y_all)
            save_path = SAVE_DIR / f"clf_beh_{ckpt}_layer{l}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({"clf": clf_full, "trained_on": ckpt, "layer": l}, f)

        print()

    # -----------------------------------------------------------------------
    # Phase 2: cross-checkpoint transfer
    # Addestra su base → testa su tutti
    # Addestra su SFT  → testa su DPO/Final
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("PHASE 2 — Cross-checkpoint transfer")
    print("=" * 65)

    for train_ckpt, test_ckpts in [
        ("base__none", ["base__none", "sft__none", "dpo__none", "final__none"]),
        ("sft__none",  ["sft__none",  "dpo__none", "final__none"]),
    ]:
        print(f"\nTrained on: {train_ckpt}")
        print(f"{'layer':>6}", end="")
        for ckpt in test_ckpts:
            print(f"  {ckpt:>20}", end="")
        print()

        for l in args.layers:
            clf_path = SAVE_DIR / f"clf_beh_{train_ckpt}_layer{l}.pkl"
            with open(clf_path, "rb") as f:
                saved = pickle.load(f)
            clf_train = saved["clf"]

            print(f"{l:>6}", end="")

            for ckpt in test_ckpts:
                col = f"layer_{l}_first_gen"
                ds = load_dataset(
                    args.hf_repo,
                    data_files={"train": f"data/{ckpt}/*.parquet"},
                    split="train",
                    token=token,
                )
                df = ds.select_columns(["predicted_refusal", col]).to_pandas()
                X = np.stack(df[col].values).astype(np.float32)
                y = df["predicted_refusal"].values.astype(int)
                acc = clf_train.score(X, y)
                print(f"  {acc:>20.3f}", end="")

                rows.append({
                    "phase":      "cross",
                    "train_on":   train_ckpt,
                    "test_on":    ckpt,
                    "layer":      l,
                    "acc_all":    acc,
                    "acc_pseudo": None,
                })

            print()

    # Salva CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
