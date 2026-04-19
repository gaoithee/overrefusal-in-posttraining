"""
analysis/representation_analysis.py

Core module for studying the geometry of refusal and over-refusal directions
in the hidden representations of language models.

For each layer l, given:
  - a set of harmful prompts   (label=1, *should* be refused)
  - a set of over-refusal prompts (label=0, *should not* be refused but are)

We:
  1. Extract hidden states h at a chosen token position (last token by default).
  2. Fit a linear probe (logistic regression or mean-diff) to get:
        v_ref  — direction discriminating harmful vs. safe
        v_over — direction discriminating over-refused vs. complied-safe
  3. Decompose:
        v_over_perp = v_over - (v_over · v_ref) * v_ref
  4. Compute:
        entanglement(l) = cos(v_ref, v_over)  ∈ [-1, 1]
        boundary_margin(l) = mean signed distance of over-refusal samples from
                             the refusal boundary (positive → wrong side)

Usage
-----
See run_representation_analysis.py for the CLI entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerGeometry:
    """Geometric quantities computed for a single layer."""
    layer: int
    v_ref: np.ndarray          # unit refusal direction  (d,)
    v_over: np.ndarray         # unit over-refusal direction (d,)
    v_over_perp: np.ndarray    # component of v_over orthogonal to v_ref (d,)
    entanglement: float        # cos(v_ref, v_over)  ∈ [-1, 1]
    boundary_margin: float     # mean signed distance of over-refused samples
                               # from the refusal decision boundary
    probe_ref_acc: float       # accuracy of the refusal probe
    probe_over_acc: float      # accuracy of the over-refusal probe


@dataclass
class CheckpointGeometry:
    """Full geometry analysis for one model checkpoint."""
    checkpoint_tag: str
    system_prompt: str
    layers: list[LayerGeometry] = field(default_factory=list)

    # convenience accessors ------------------------------------------------
    @property
    def entanglement_curve(self) -> np.ndarray:
        return np.array([lg.entanglement for lg in self.layers])

    @property
    def margin_curve(self) -> np.ndarray:
        return np.array([lg.boundary_margin for lg in self.layers])

    @property
    def layer_indices(self) -> list[int]:
        return [lg.layer for lg in self.layers]


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def _format_prompt(
    prompt: str,
    system_prompt: str | None,
    tokenizer,
) -> str:
    """Apply chat template if available, otherwise concatenate naively."""
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return (f"<|system|>\n{system_prompt}\n" if system_prompt else "") + prompt


@torch.no_grad()
def extract_hidden_states(
    prompts: list[str],
    model,
    tokenizer,
    system_prompt: str | None = None,
    layers: list[int] | None = None,
    token_position: Literal["last", "first", "mean"] = "last",
    batch_size: int = 8,
    device: str | None = None,
    max_length: int = 512,
) -> dict[int, np.ndarray]:
    """
    Extract hidden states from `model` for each prompt.

    Parameters
    ----------
    prompts:
        List of N input strings.
    model:
        HuggingFace CausalLM with output_hidden_states support.
    tokenizer:
        Matching tokenizer.
    system_prompt:
        Optional system prompt prepended to every input.
    layers:
        Which transformer layers to collect (0 = embedding).
        If None, collects all layers.
    token_position:
        Which token position to read the hidden state from:
        "last"  → last non-padding token  (default)
        "first" → first token (BOS / prompt start)
        "mean"  → mean over all non-padding tokens
    batch_size:
        Number of prompts per forward pass.
    device:
        Torch device string.  Defaults to model's device.
    max_length:
        Tokenizer truncation length.

    Returns
    -------
    dict  layer_idx -> np.ndarray of shape (N, hidden_dim)
    """
    if device is None:
        device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(num_layers + 1))   # +1 for embedding layer

    # initialise storage
    storage: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    formatted = [
        _format_prompt(p, system_prompt, tokenizer) for p in prompts
    ]

    for start in tqdm(range(0, len(formatted), batch_size), desc="Extracting"):
        batch_texts = formatted[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )
        # out.hidden_states: tuple of (num_layers+1) tensors (B, T, D)
        attention_mask = enc["attention_mask"]

        for layer_idx in layers:
            hs = out.hidden_states[layer_idx]  # (B, T, D)

            if token_position == "last":
                # index of last real token for each item
                seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
                vecs = hs[
                    torch.arange(hs.size(0), device=device), seq_lens
                ]  # (B, D)
            elif token_position == "first":
                vecs = hs[:, 0, :]
            elif token_position == "mean":
                mask_f = attention_mask.unsqueeze(-1).float()
                vecs = (hs * mask_f).sum(1) / mask_f.sum(1)
            else:
                raise ValueError(f"Unknown token_position: {token_position}")

            storage[layer_idx].append(vecs.float().cpu().numpy())

    return {l: np.concatenate(storage[l], axis=0) for l in layers}


# ---------------------------------------------------------------------------
# Direction fitting
# ---------------------------------------------------------------------------

ProbeMethod = Literal["mean_diff", "logistic"]


def fit_direction(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    method: ProbeMethod = "logistic",
) -> tuple[np.ndarray, float]:
    """
    Fit a linear probe to separate positive from negative samples.

    Returns
    -------
    v : np.ndarray (d,)   Unit-normalised weight vector.
    acc : float           Train accuracy.
    """
    if method == "mean_diff":
        v = X_pos.mean(0) - X_neg.mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        # threshold = midpoint of projected class means (correct for any class balance)
        threshold = ((X_pos @ v).mean() + (X_neg @ v).mean()) / 2.0
        scores = np.concatenate([X_pos @ v, X_neg @ v])
        labels = np.array([1] * len(X_pos) + [0] * len(X_neg))
        preds  = (scores > threshold).astype(int)
        acc    = (preds == labels).mean()
        return v, float(acc)

    elif method == "logistic":
        X = np.concatenate([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))
        clf = LogisticRegression(
            max_iter=1000, C=0.1, fit_intercept=True, random_state=0
        )
        clf.fit(X, y)
        v   = clf.coef_[0]
        v   = v / (np.linalg.norm(v) + 1e-12)
        acc = clf.score(X, y)
        return v, float(acc)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Core geometry computation
# ---------------------------------------------------------------------------

def compute_layer_geometry(
    layer: int,
    hidden_harmful: np.ndarray,    # (N_harm, D) — harmful (should refuse)
    hidden_safe: np.ndarray,       # (N_safe, D) — clearly safe (complied)
    hidden_over: np.ndarray,       # (N_over, D) — over-refused (false positives)
    method: ProbeMethod = "logistic",
) -> LayerGeometry:
    """
    Compute refusal/over-refusal directions and entanglement for one layer.

    Parameters
    ----------
    hidden_harmful:
        Hidden states for prompts that the model correctly refuses.
    hidden_safe:
        Hidden states for prompts that the model correctly answers.
    hidden_over:
        Hidden states for prompts that the model refuses but shouldn't
        (the over-refusal / false positive set).
    """
    # --- refusal direction: harmful vs safe ---------------------------------
    v_ref, acc_ref = fit_direction(hidden_harmful, hidden_safe, method)

    # --- over-refusal direction: over-refused vs safe -----------------------
    # We treat over-refused as "positive" and safe as "negative"
    # so that v_over points toward the over-refusal region
    v_over, acc_over = fit_direction(hidden_over, hidden_safe, method)

    # --- orthogonal decomposition -------------------------------------------
    dot = float(np.dot(v_ref, v_over))   # this IS the cosine (both unit)
    v_over_perp = v_over - dot * v_ref
    norm_perp   = np.linalg.norm(v_over_perp)
    if norm_perp > 1e-12:
        v_over_perp = v_over_perp / norm_perp

    # --- boundary margin for over-refusal samples ---------------------------
    # Project over-refusal hidden states onto v_ref
    # Positive values → on the "harmful/refuse" side of the boundary
    proj_over   = hidden_over   @ v_ref
    proj_safe   = hidden_safe   @ v_ref
    proj_harm   = hidden_harmful @ v_ref

    # Centre at the midpoint between harmful and safe means
    midpoint    = (proj_harm.mean() + proj_safe.mean()) / 2.0
    boundary_margin = float((proj_over - midpoint).mean())

    return LayerGeometry(
        layer=layer,
        v_ref=v_ref,
        v_over=v_over,
        v_over_perp=v_over_perp,
        entanglement=dot,
        boundary_margin=boundary_margin,
        probe_ref_acc=acc_ref,
        probe_over_acc=acc_over,
    )


def compute_checkpoint_geometry(
    checkpoint_tag: str,
    system_prompt_key: str,
    hidden_harmful: dict[int, np.ndarray],
    hidden_safe:    dict[int, np.ndarray],
    hidden_over:    dict[int, np.ndarray],
    method: ProbeMethod = "logistic",
    layers: list[int] | None = None,
) -> CheckpointGeometry:
    """
    Run compute_layer_geometry for all requested layers.

    All three hidden-state dicts must have the same key set (layer indices).
    """
    if layers is None:
        layers = sorted(hidden_harmful.keys())

    geom = CheckpointGeometry(
        checkpoint_tag=checkpoint_tag,
        system_prompt=system_prompt_key,
    )

    for l in tqdm(layers, desc=f"Computing geometry [{checkpoint_tag}]"):
        lg = compute_layer_geometry(
            layer=l,
            hidden_harmful=hidden_harmful[l],
            hidden_safe=hidden_safe[l],
            hidden_over=hidden_over[l],
            method=method,
        )
        geom.layers.append(lg)

    return geom


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_geometry(geom: CheckpointGeometry, path: Path) -> None:
    """Save a CheckpointGeometry to a compressed numpy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    meta_keys = []
    for lg in geom.layers:
        prefix = f"layer{lg.layer:03d}"
        arrays[f"{prefix}_v_ref"]      = lg.v_ref
        arrays[f"{prefix}_v_over"]     = lg.v_over
        arrays[f"{prefix}_v_over_perp"]= lg.v_over_perp
        meta_keys.append(
            f"{lg.layer},{lg.entanglement:.6f},{lg.boundary_margin:.6f},"
            f"{lg.probe_ref_acc:.4f},{lg.probe_over_acc:.4f}"
        )

    arrays["__meta__"] = np.array(meta_keys)
    arrays["__tags__"] = np.array([geom.checkpoint_tag, geom.system_prompt])
    np.savez_compressed(path, **arrays)
    logger.info("Saved geometry to %s", path)


def load_geometry(path: Path) -> CheckpointGeometry:
    """Load a CheckpointGeometry previously saved with save_geometry."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    tags = data["__tags__"]
    geom = CheckpointGeometry(
        checkpoint_tag=str(tags[0]),
        system_prompt=str(tags[1]),
    )
    for row in data["__meta__"]:
        parts = row.split(",")
        l         = int(parts[0])
        entangle  = float(parts[1])
        margin    = float(parts[2])
        acc_ref   = float(parts[3])
        acc_over  = float(parts[4])
        prefix    = f"layer{l:03d}"
        geom.layers.append(LayerGeometry(
            layer=l,
            v_ref=data[f"{prefix}_v_ref"],
            v_over=data[f"{prefix}_v_over"],
            v_over_perp=data[f"{prefix}_v_over_perp"],
            entanglement=entangle,
            boundary_margin=margin,
            probe_ref_acc=acc_ref,
            probe_over_acc=acc_over,
        ))
    return geom


# ---------------------------------------------------------------------------
# Aggregation across checkpoints
# ---------------------------------------------------------------------------

def entanglement_table(
    geometries: list[CheckpointGeometry],
) -> dict[str, np.ndarray]:
    """
    Build a dict  checkpoint_tag -> entanglement_curve (num_layers,)
    for easy plotting / comparison.
    """
    return {
        f"{g.checkpoint_tag}__{g.system_prompt}": g.entanglement_curve
        for g in geometries
    }