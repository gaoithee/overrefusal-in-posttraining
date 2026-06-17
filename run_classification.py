from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
from pathlib import Path
from datasets import load_dataset

token = open(os.path.expanduser("~/.hf_token")).read().strip()

checkpoints = ["base__none", "sft__none", "dpo__none", "final__none"]
layers = [8, 16, 19, 24, 26, 31]

SAVE_DIR = Path("results/olmo2/classifiers")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def assign_3cat(row):
    if row["label"] == 1:
        return "harmful"
    if row["source"] in ["or_bench", "false_reject"]:
        return "pseudo_harm"
    return "harmless"


# ---------------------------------------------------------------------------
# Phase 1: train on each checkpoint, cross-val + save
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 1 — Train per checkpoint (cross-val)")
print("=" * 70)

for ckpt in checkpoints:
    act_cols = [f"layer_{l}_first_gen" for l in layers]
    ds = load_dataset("saracandu/olmo-activations",
                      data_files={"train": f"data/{ckpt}/*.parquet"},
                      split="train", token=token)
    df = ds.select_columns(["label", "source"] + act_cols).to_pandas()
    df["cat3"] = df.apply(assign_3cat, axis=1)

    print(f"\n{ckpt}")
    print(f"  {'layer':>7}  {'acc_3class':>10}  {'acc_pseudo_vs_harm':>18}  {'acc_pseudo_vs_harmless':>22}")

    for l in layers:
        col = f"layer_{l}_first_gen"
        X = np.stack(df[col].values).astype(np.float32)
        y = df["cat3"].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

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

        # Train on ALL data and save
        clf3_full = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        clf3_full.fit(X, y_enc)
        save_path = SAVE_DIR / f"clf3_{ckpt}_layer{l}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"clf": clf3_full, "le": le, "trained_on": ckpt, "layer": l}, f)


# ---------------------------------------------------------------------------
# Phase 2: cross-checkpoint transfer — train on base, test on others
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 2 — Cross-checkpoint transfer (trained on base__none)")
print("=" * 70)

print(f"\n  {'layer':>7}", end="")
for ckpt in checkpoints:
    print(f"  {ckpt:>20}", end="")
print()

for l in layers:
    # Load base classifier
    clf_path = SAVE_DIR / f"clf3_base__none_layer{l}.pkl"
    with open(clf_path, "rb") as f:
        saved = pickle.load(f)
    clf_base = saved["clf"]
    le_base  = saved["le"]

    print(f"  layer {l:2d}", end="")

    for ckpt in checkpoints:
        act_cols = [f"layer_{l}_first_gen"]
        ds = load_dataset("saracandu/olmo-activations",
                          data_files={"train": f"data/{ckpt}/*.parquet"},
                          split="train", token=token)
        df = ds.select_columns(["label", "source"] + act_cols).to_pandas()
        df["cat3"] = df.apply(assign_3cat, axis=1)

        X = np.stack(df[f"layer_{l}_first_gen"].values).astype(np.float32)
        y_enc = le_base.transform(df["cat3"].values)

        acc = clf_base.score(X, y_enc)
        print(f"  {acc:>20.3f}", end="")

    print()

print("\nDone. Classifiers saved in", SAVE_DIR)