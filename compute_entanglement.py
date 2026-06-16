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
                      (normalizzato per la distanza tra i due cluster)
                      0 = al midpoint
                     +1 = al centroide harmful
                     -1 = al centroide harmless
                     >1 = oltre il cluster harmful

Uso:
    python compute_entanglement.py
    python compute_entanglement.py --token-position first_gen
    python compute_entanglement.py --method logistic
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
    method: str = "mean_diff",
) -> dict:
    if len(h_harmful) == 0 or len(h_harmless) == 0 or len(h_pseudo) == 0:
        return {
            "entanglement":      None,
            "boundary_margin":   None,
            "boundary_margin_n": None,
            "cluster_distance":  None,
            "acc_ref":           None,
            "acc_over":          None,
            "n_harmful":         len(h_harmful),
            "n_harmless":        len(h_harmless),
            "n_pseudo":          len(h_pseudo),
        }

    # true refusal direction: harmful vs harmless
    v_ref, acc_ref = fit_direction(h_harmful, h_harmless, method)

    # false refusal direction: pseudo_harm vs harmless
    f, acc_over = fit_direction(h_pseudo, h_harmless, method)

    # entanglement = cosine similarity (both unit vectors)
    entanglement = float(np.dot(v_ref, f))

    # projections onto v_ref
    proj_harm    = h_harmful  @ v_ref
    proj_harmless = h_harmless @ v_ref
    proj_pseudo  = h_pseudo   @ v_ref

    # midpoint between the two cluster means
    mu_harmful  = proj_harm.mean()
    mu_harmless = proj_harmless.mean()
    midpoint    = (mu_harmful + mu_harmless) / 2.0

    # raw boundary margin
    boundary_margin = float((proj_pseudo - midpoint).mean())

    # cluster distance (harmful mean - harmless mean, along v_ref)
    # always positive because v_ref points from harmless to harmful
    cluster_distance = float(mu_harmful - mu_harmless)

    # normalized boundary margin:
    #   0   = pseudo at midpoint
    #  +0.5 = halfway between midpoint and harmful centroid
    #  +1   = at harmful centroid
    #  -1   = at harmless centroid
    if abs(cluster_distance) > 1e-12:
        boundary_margin_n = boundary_margin / (cluster_distance / 2.0)
    else:
        boundary_margin_n = None

    return {
        "entanglement":      entanglement,
        "boundary_margin":   boundary_margin,
        "boundary_margin_n": boundary_margin_n,
        "cluster_distance":  cluster_distance,
        "acc_ref":           acc_ref,
        "acc_over":          acc_over,
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
    args = parser.parse_args()

    logger.info("Loading dataset from %s ...", args.hf_repo)
    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.hf_token")
    ).read().strip()

    from datasets import load_dataset
    ds = load_dataset(args.hf_repo, split="train", token=token)
    df = ds.to_pandas()

    layer_cols = [c for c in df.columns if c.startswith("layer_") and args.token_position in c]
    layers = sorted({int(c.split("_")[1]) for c in layer_cols})
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

            geo = compute_geometry(h_harmful, h_harmless, h_pseudo, method=args.method)

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

    print("\n=== boundary_margin_n per layer (base__none vs sft__none) ===")
    pivot = result_df[result_df["checkpoint"].isin(["base__none", "sft__none"])].pivot(
        index="layer", columns="checkpoint", values="boundary_margin_n"
    )
    print(pivot.round(4).to_string())

    print("\n=== cluster_distance per layer (base__none) ===")
    sub = result_df[result_df["checkpoint"] == "base__none"][["layer", "cluster_distance"]]
    print(sub.set_index("layer").round(4).to_string())


if __name__ == "__main__":
    main()
