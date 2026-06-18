"""
compute_entanglement.py

Calcola entanglement e boundary margin per ogni layer e checkpoint
leggendo direttamente le attivazioni da saracandu/olmo-activations.

Gruppi (Wang et al. mapping):
  harmful     = qualsiasi source con label=1
  pseudo_harm = or_bench + false_reject (label=0, formulati per sembrare pericolosi)
  harmless    = tutti gli altri label=0 (genuinamente innocui)

Per ogni checkpoint e layer:
  v_ref  = mean(h_harmful)  - mean(h_harmless)   → true refusal direction
  f      = mean(h_pseudo)   - mean(h_harmless)   → false refusal direction
  entanglement      = cos(v_ref, f)
  boundary_margin   = mean projection of pseudo_harm onto v_ref minus midpoint
  boundary_margin_n = boundary_margin / cluster_distance
  v_beh  = mean(h_pseudo_refused) - mean(h_pseudo_not_refused)
  cos_vbeh_vref  = cos(v_beh, v_ref)
  cos_vbeh_vover = cos(v_beh, f)

Uso:
    python compute_entanglement.py
    python compute_entanglement.py --token-position first_gen
    python compute_entanglement.py --method logistic
    python compute_entanglement.py --exclude-sources beavertails
    python compute_entanglement.py --out results/olmo2/geometry/entanglement.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HF_REPO = "saracandu/olmo-activations"

PSEUDO_HARM_SOURCES = {"or_bench", "false_reject"}

CHECKPOINT_ORDER = [
    "base__none",
    "base__mistral_safety",
    "sft__none",
    "sft__mistral_safety",
    "dpo__none",
    "dpo__mistral_safety",
    "final__none",
    "final__mistral_safety",
]


# ---------------------------------------------------------------------------
# Direction fitting
# ---------------------------------------------------------------------------

def fit_mean_diff(X_pos: np.ndarray, X_neg: np.ndarray) -> tuple[np.ndarray, float]:
    v = X_pos.mean(0) - X_neg.mean(0)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v, 0.0
    v = v / norm
    threshold = ((X_pos @ v).mean() + (X_neg @ v).mean()) / 2.0
    scores = np.concatenate([X_pos @ v, X_neg @ v])
    labels = np.array([1] * len(X_pos) + [0] * len(X_neg))
    acc = ((scores > threshold).astype(int) == labels).mean()
    return v, float(acc)


def fit_logistic(X_pos: np.ndarray, X_neg: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.concatenate([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    clf = LogisticRegression(max_iter=1000, C=0.1, fit_intercept=True, random_state=0)
    clf.fit(X, y)
    v = clf.coef_[0]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v, 0.0
    v = v / norm
    return v, float(clf.score(X, y))


def fit_direction(X_pos, X_neg, method="mean_diff"):
    if method == "mean_diff":
        return fit_mean_diff(X_pos, X_neg)
    elif method == "logistic":
        return fit_logistic(X_pos, X_neg)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Geometry for one layer
# ---------------------------------------------------------------------------

def compute_geometry(
    h_harmful: np.ndarray,
    h_harmless: np.ndarray,
    h_pseudo: np.ndarray,
    predicted_refusal_pseudo: np.ndarray | None = None,
    method: str = "mean_diff",
) -> dict:

    empty = {
        "entanglement":      None,
        "boundary_margin":   None,
        "boundary_margin_n": None,
        "cluster_distance":  None,
        "acc_ref":           None,
        "acc_over":          None,
        "cos_vbeh_vref":     None,
        "cos_vbeh_vover":    None,
        "n_pseudo_refused":  None,
        "n_pseudo_accepted": None,
        "n_harmful":         len(h_harmful),
        "n_harmless":        len(h_harmless),
        "n_pseudo":          len(h_pseudo),
    }

    if len(h_harmful) == 0 or len(h_harmless) == 0 or len(h_pseudo) == 0:
        return empty

    # true refusal direction: harmful vs harmless
    v_ref, acc_ref = fit_direction(h_harmful, h_harmless, method)

    # false refusal direction: pseudo_harm vs harmless
    f, acc_over = fit_direction(h_pseudo, h_harmless, method)

    # entanglement = cosine similarity (both unit vectors)
    entanglement = float(np.dot(v_ref, f))

    # projections onto v_ref
    proj_harm     = h_harmful  @ v_ref
    proj_harmless = h_harmless @ v_ref
    proj_pseudo   = h_pseudo   @ v_ref

    mu_harmful  = proj_harm.mean()
    mu_harmless = proj_harmless.mean()
    midpoint    = (mu_harmful + mu_harmless) / 2.0

    boundary_margin  = float((proj_pseudo - midpoint).mean())
    cluster_distance = float(mu_harmful - mu_harmless)

    if abs(cluster_distance) > 1e-12:
        boundary_margin_n = boundary_margin / (cluster_distance / 2.0)
    else:
        boundary_margin_n = None

    # -------------------------------------------------------------------
    # v_beh: direzione comportamentale sulle pseudo-harmful
    # -------------------------------------------------------------------
    cos_vbeh_vref  = None
    cos_vbeh_vover = None
    n_refused  = None
    n_accepted = None

    if predicted_refusal_pseudo is not None:
        refused_mask = predicted_refusal_pseudo == 1
        h_refused  = h_pseudo[refused_mask]
        h_accepted = h_pseudo[~refused_mask]
        n_refused  = int(refused_mask.sum())
        n_accepted = int((~refused_mask).sum())

        if n_refused >= 5 and n_accepted >= 5:
            v_beh = h_refused.mean(0) - h_accepted.mean(0)
            norm_beh = np.linalg.norm(v_beh)
            if norm_beh > 1e-12:
                v_beh = v_beh / norm_beh
                cos_vbeh_vref  = float(np.dot(v_beh, v_ref))
                cos_vbeh_vover = float(np.dot(v_beh, f))
        else:
            logger.warning(
                "Troppo pochi esempi refused=%d / accepted=%d per calcolare v_beh",
                n_refused, n_accepted,
            )

    return {
        "entanglement":      entanglement,
        "boundary_margin":   boundary_margin,
        "boundary_margin_n": boundary_margin_n,
        "cluster_distance":  cluster_distance,
        "acc_ref":           acc_ref,
        "acc_over":          acc_over,
        "cos_vbeh_vref":     cos_vbeh_vref,
        "cos_vbeh_vover":    cos_vbeh_vover,
        "n_pseudo_refused":  n_refused,
        "n_pseudo_accepted": n_accepted,
        "n_harmful":         len(h_harmful),
        "n_harmless":        len(h_harmless),
        "n_pseudo":          len(h_pseudo),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--token-position", default="last_prompt",
                        choices=["last_prompt", "first_gen"])
    parser.add_argument("--method", default="mean_diff",
                        choices=["mean_diff", "logistic"])
    parser.add_argument("--out", default="results/olmo2/geometry/entanglement.csv")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max samples per group per checkpoint (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-sources", nargs="*", default=None,
                        metavar="SOURCE",
                        help="Source da escludere completamente dal dataset "
                             "(es. --exclude-sources beavertails). "
                             "Influenza sia il calcolo delle direzioni che v_beh.")
    args = parser.parse_args()

    logger.info("Loading dataset from %s ...", args.hf_repo)
    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    from datasets import load_dataset

    # Carica checkpoint per checkpoint per evitare CastError da schema misto
    # (i checkpoint mistral_safety hanno 7 colonne post_instr, gli altri 3)
    base_cols = ["label", "source", "checkpoint", "predicted_refusal"]
    dfs = []
    layers = None

    for ckpt in CHECKPOINT_ORDER:
        try:
            ds = load_dataset(
                args.hf_repo,
                data_files={"train": f"data/{ckpt}/*.parquet"},
                split="train",
                token=token,
            )
            act_cols = [c for c in ds.column_names
                        if c.startswith("layer_") and args.token_position in c]
            available = [c for c in base_cols + act_cols if c in ds.column_names]
            df_ckpt = ds.select_columns(available).to_pandas()
            dfs.append(df_ckpt)

            if layers is None:
                layers = sorted({int(c.split("_")[1]) for c in act_cols})

            logger.info("[%s] %d righe", ckpt, len(df_ckpt))
        except Exception as e:
            logger.warning("Skipping %s: %s", ckpt, e)

    if not dfs:
        logger.error("Nessun checkpoint caricato.")
        return

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Dataset totale: %d righe", len(df))

    # Esclusione source completa
    if args.exclude_sources:
        before = len(df)
        df = df[~df["source"].isin(args.exclude_sources)].reset_index(drop=True)
        logger.info(
            "Escluse source %s: %d -> %d righe",
            args.exclude_sources, before, len(df),
        )

    logger.info("Layers: %s | position: %s", layers, args.token_position)

    # assign groups
    df["group"] = None
    df.loc[df["label"] == 1, "group"] = "harmful"
    df.loc[(df["label"] == 0) & (df["source"].isin(PSEUDO_HARM_SOURCES)),  "group"] = "pseudo_harm"
    df.loc[(df["label"] == 0) & (~df["source"].isin(PSEUDO_HARM_SOURCES)), "group"] = "harmless"

    logger.info("Group sizes overall:\n%s", df["group"].value_counts().to_string())

    rows = []

    for ckpt in CHECKPOINT_ORDER:
        sub = df[df["checkpoint"] == ckpt]
        if len(sub) == 0:
            logger.warning("No rows for checkpoint %s, skipping.", ckpt)
            continue

        g_harmful  = sub[sub["group"] == "harmful"]
        g_harmless = sub[sub["group"] == "harmless"]
        g_pseudo   = sub[sub["group"] == "pseudo_harm"]

        if args.n_samples:
            g_harmful  = g_harmful.sample(min(args.n_samples, len(g_harmful)),   random_state=args.seed)
            g_harmless = g_harmless.sample(min(args.n_samples, len(g_harmless)), random_state=args.seed)
            g_pseudo   = g_pseudo.sample(min(args.n_samples, len(g_pseudo)),     random_state=args.seed)

        logger.info(
            "[%s] harmful=%d  harmless=%d  pseudo_harm=%d",
            ckpt, len(g_harmful), len(g_harmless), len(g_pseudo),
        )

        for layer in layers:
            col = f"layer_{layer}_{args.token_position}"

            h_harmful  = np.stack(g_harmful[col].values).astype(np.float32)
            h_harmless = np.stack(g_harmless[col].values).astype(np.float32)
            h_pseudo   = np.stack(g_pseudo[col].values).astype(np.float32)

            # predicted_refusal per le pseudo-harmful (per v_beh)
            pred_ref_pseudo = None
            if "predicted_refusal" in g_pseudo.columns:
                pred_ref_pseudo = g_pseudo["predicted_refusal"].values.astype(int)

            geo = compute_geometry(
                h_harmful, h_harmless, h_pseudo,
                predicted_refusal_pseudo=pred_ref_pseudo,
                method=args.method,
            )

            rows.append({
                "checkpoint":     ckpt,
                "layer":          layer,
                "token_position": args.token_position,
                "method":         args.method,
                **geo,
            })

        logger.info("[%s] done.", ckpt)

    result_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)

    # summary
    print("\n=== Mean entanglement per checkpoint (across layers) ===")
    print(result_df.groupby("checkpoint")["entanglement"].mean().reindex(CHECKPOINT_ORDER).round(4).to_string())

    print("\n=== Mean boundary_margin_n per checkpoint (across layers) ===")
    print(result_df.groupby("checkpoint")["boundary_margin_n"].mean().reindex(CHECKPOINT_ORDER).round(4).to_string())

    print("\n=== cos(v_beh, v_ref) per checkpoint (across layers) ===")
    print(result_df.groupby("checkpoint")["cos_vbeh_vref"].mean().reindex(CHECKPOINT_ORDER).round(4).to_string())

    print("\n=== cos(v_beh, v_over) per checkpoint (across layers) ===")
    print(result_df.groupby("checkpoint")["cos_vbeh_vover"].mean().reindex(CHECKPOINT_ORDER).round(4).to_string())

    print("\n=== boundary_margin_n per layer (base__none vs sft__none) ===")
    pivot = result_df[result_df["checkpoint"].isin(["base__none", "sft__none"])].pivot(
        index="layer", columns="checkpoint", values="boundary_margin_n"
    )
    print(pivot.round(4).to_string())

    print("\n=== cos(v_beh, v_ref) vs cos(v_beh, v_over) — sft/dpo/final per layer ===")
    for ckpt in ["sft__none", "dpo__none", "final__none"]:
        sub = result_df[result_df["checkpoint"] == ckpt][
            ["layer", "cos_vbeh_vref", "cos_vbeh_vover"]
        ].set_index("layer")
        print(f"\n  {ckpt}")
        print(sub.round(4).to_string())


if __name__ == "__main__":
    main()