"""
run_representation_analysis.py

CLI entry point for Experiments 1, 2, 3:

  Exp 1 — Entanglement as proxy for over-refusal:
        Compute per-layer entanglement (v_ref · v_over) and boundary margin
        for a single model checkpoint and correlate with behavioural metrics.

  Exp 2 — Entanglement evolution during post-training:
        Sweep over all checkpoints listed in a config file (base→SFT→DPO→
        Instruct) and track how entanglement changes at each training stage.

  Exp 3 — System prompt influence:
        For each checkpoint, compare entanglement under different system prompts.

Usage
-----
# Exp 1: single checkpoint
python run_representation_analysis.py \
    --config config_olmo2 \
    --experiment entanglement \
    --checkpoint sft__none \
    --n-samples 200

# Exp 2: all checkpoints, fixed system prompt
python run_representation_analysis.py \
    --config config_olmo2 \
    --experiment evolution \
    --system-prompt none \
    --n-samples 200

# Exp 3: system prompt sweep, all checkpoints
python run_representation_analysis.py \
    --config config_olmo2 \
    --experiment system_prompt \
    --n-samples 200

Results land in results/<model>/geometry/ as .npz files and a summary CSV.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.representation_analysis import (
    CheckpointGeometry,
    compute_checkpoint_geometry,
    entanglement_table,
    extract_hidden_states,
    load_geometry,
    save_geometry,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_name: str):
    """Import a config_*.py module by name and return it."""
    mod = importlib.import_module(config_name)
    return mod


def _sample_prompts(
    dataset_key: str,
    label: int,
    n: int,
    seed: int = 42,
) -> list[str]:
    """
    Sample n prompts with the given label from a registered dataset.

    Datasets expected to have 'prompt' and 'label' columns (same convention
    as the rest of the repo).  Adjust field names below if needed.
    """
    from data.dataset_loader import load_dataset_by_key  # repo-internal
    df = load_dataset_by_key(dataset_key)
    subset = df[df["label"] == label]
    if len(subset) < n:
        logger.warning(
            "Only %d samples with label=%d in %s (requested %d)",
            len(subset), label, dataset_key, n,
        )
    return subset["prompt"].sample(min(n, len(subset)), random_state=seed).tolist()


def _build_prompt_sets(
    n_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Return (harmful_prompts, safe_prompts, over_refusal_prompts).

    harmful      → from harmbench/jailbreakbench (label=1)
    safe         → from or_bench/false_reject    (label=0, clearly safe)
    over_refusal → also label=0 but typically over-refused
                   (we use or_bench which is specifically designed for this)
    """
    n = n_samples // 2  # half from each source for balance

    harmful_prompts = (
        _sample_prompts("harmbench",      label=1, n=n, seed=seed)
        + _sample_prompts("jailbreakbench", label=1, n=n, seed=seed)
    )
    safe_prompts = _sample_prompts("wildguard", label=0, n=n_samples, seed=seed)
    over_refusal_prompts = (
        _sample_prompts("or_bench",    label=0, n=n, seed=seed)
        + _sample_prompts("false_reject", label=0, n=n, seed=seed)
    )
    return harmful_prompts, safe_prompts, over_refusal_prompts


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    logger.info("Loading model %s ...", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _extract_all(
    prompts_harm: list[str],
    prompts_safe: list[str],
    prompts_over: list[str],
    model,
    tokenizer,
    system_prompt: str | None,
    layers: list[int],
    batch_size: int,
    device: str,
    token_position: str,
) -> tuple[dict, dict, dict]:
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        layers=layers,
        batch_size=batch_size,
        device=device,
        token_position=token_position,
    )
    h_harm = extract_hidden_states(prompts_harm, **kwargs)
    h_safe = extract_hidden_states(prompts_safe, **kwargs)
    h_over = extract_hidden_states(prompts_over, **kwargs)
    return h_harm, h_safe, h_over


def _geometry_cache_path(results_dir: Path, checkpoint_tag: str, system_prompt_key: str) -> Path:
    safe_tag = checkpoint_tag.replace("/", "_").replace(" ", "_")
    return results_dir / "geometry" / f"{safe_tag}__{system_prompt_key}.npz"


def _save_summary_csv(geometries: list[CheckpointGeometry], out_path: Path) -> None:
    """Write a flat CSV with one row per (checkpoint, layer)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in geometries:
        for lg in g.layers:
            rows.append({
                "checkpoint":       g.checkpoint_tag,
                "system_prompt":    g.system_prompt,
                "layer":            lg.layer,
                "entanglement":     lg.entanglement,
                "boundary_margin":  lg.boundary_margin,
                "probe_ref_acc":    lg.probe_ref_acc,
                "probe_over_acc":   lg.probe_over_acc,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Summary CSV saved to %s", out_path)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_entanglement(args, cfg) -> None:
    """Exp 1: analyse a single checkpoint, correlate with I/O metrics."""
    checkpoint_tag  = args.checkpoint
    system_prompt_key = args.system_prompt

    # Resolve checkpoint → model path using config
    checkpoints_map: dict[str, str] = {
        c["tag"]: c["model"] for c in cfg.CHECKPOINTS
    }
    if checkpoint_tag not in checkpoints_map:
        raise ValueError(
            f"Checkpoint '{checkpoint_tag}' not found in config. "
            f"Available: {list(checkpoints_map)}"
        )
    model_path = checkpoints_map[checkpoint_tag]

    system_prompts_map: dict[str, str | None] = cfg.SYSTEM_PROMPTS
    system_prompt = system_prompts_map.get(system_prompt_key)

    results_dir = Path("results") / cfg.MODEL_KEY
    cache_path  = _geometry_cache_path(results_dir, checkpoint_tag, system_prompt_key)

    if cache_path.exists() and not args.force:
        logger.info("Loading cached geometry from %s", cache_path)
        geom = load_geometry(cache_path)
    else:
        model, tokenizer = _load_model_and_tokenizer(model_path, args.device)
        num_layers = model.config.num_hidden_layers
        layers     = list(range(0, num_layers + 1, args.layer_stride))

        prompts_harm, prompts_safe, prompts_over = _build_prompt_sets(
            args.n_samples, seed=args.seed
        )

        h_harm, h_safe, h_over = _extract_all(
            prompts_harm, prompts_safe, prompts_over,
            model, tokenizer, system_prompt,
            layers, args.batch_size, args.device, args.token_position,
        )
        del model  # free GPU memory

        geom = compute_checkpoint_geometry(
            checkpoint_tag=checkpoint_tag,
            system_prompt_key=system_prompt_key,
            hidden_harmful=h_harm,
            hidden_safe=h_safe,
            hidden_over=h_over,
            method=args.probe_method,
            layers=layers,
        )
        save_geometry(geom, cache_path)

    # --- correlation with behavioural I/O metrics (if results CSV exists) ---
    io_csv = results_dir / "raw_results.csv"
    if io_csv.exists():
        _correlate_with_io(geom, io_csv, checkpoint_tag, system_prompt_key, results_dir)
    else:
        logger.warning(
            "No raw_results.csv found at %s. "
            "Run run_experiment.py first for I/O correlation.",
            io_csv,
        )

    _save_summary_csv([geom], results_dir / "geometry" / "summary_exp1.csv")
    logger.info("Exp 1 done. Entanglement curve:\n%s", geom.entanglement_curve)


def run_evolution(args, cfg) -> None:
    """Exp 2: sweep all checkpoints and track entanglement over training."""
    system_prompt_key = args.system_prompt
    system_prompt     = cfg.SYSTEM_PROMPTS.get(system_prompt_key)
    results_dir       = Path("results") / cfg.MODEL_KEY

    geometries = []

    for ckpt in cfg.CHECKPOINTS:
        checkpoint_tag = ckpt["tag"]
        model_path     = ckpt["model"]
        cache_path     = _geometry_cache_path(results_dir, checkpoint_tag, system_prompt_key)

        if cache_path.exists() and not args.force:
            logger.info("[%s] Loading cached geometry", checkpoint_tag)
            geom = load_geometry(cache_path)
        else:
            model, tokenizer = _load_model_and_tokenizer(model_path, args.device)
            num_layers = model.config.num_hidden_layers
            layers     = list(range(0, num_layers + 1, args.layer_stride))

            prompts_harm, prompts_safe, prompts_over = _build_prompt_sets(
                args.n_samples, seed=args.seed
            )
            h_harm, h_safe, h_over = _extract_all(
                prompts_harm, prompts_safe, prompts_over,
                model, tokenizer, system_prompt,
                layers, args.batch_size, args.device, args.token_position,
            )
            del model

            geom = compute_checkpoint_geometry(
                checkpoint_tag=checkpoint_tag,
                system_prompt_key=system_prompt_key,
                hidden_harmful=h_harm,
                hidden_safe=h_safe,
                hidden_over=h_over,
                method=args.probe_method,
                layers=layers,
            )
            save_geometry(geom, cache_path)

        geometries.append(geom)

    _save_summary_csv(geometries, results_dir / "geometry" / "summary_exp2.csv")
    logger.info("Exp 2 done. Saved %d checkpoint geometries.", len(geometries))

    # Print mean entanglement per checkpoint (averaged over layers)
    print("\n=== Mean entanglement across layers ===")
    for g in geometries:
        print(f"  {g.checkpoint_tag:30s}  {g.entanglement_curve.mean():.4f}")


def run_system_prompt(args, cfg) -> None:
    """Exp 3: compare entanglement across system prompts for all checkpoints."""
    results_dir = Path("results") / cfg.MODEL_KEY
    geometries  = []

    for ckpt in cfg.CHECKPOINTS:
        checkpoint_tag = ckpt["tag"]
        model_path     = ckpt["model"]

        # Load model once per checkpoint, sweep system prompts
        model, tokenizer = None, None

        for sp_key, system_prompt in cfg.SYSTEM_PROMPTS.items():
            cache_path = _geometry_cache_path(results_dir, checkpoint_tag, sp_key)

            if cache_path.exists() and not args.force:
                logger.info("[%s | %s] Loading cached geometry", checkpoint_tag, sp_key)
                geom = load_geometry(cache_path)
            else:
                if model is None:
                    model, tokenizer = _load_model_and_tokenizer(model_path, args.device)
                num_layers = model.config.num_hidden_layers
                layers     = list(range(0, num_layers + 1, args.layer_stride))

                prompts_harm, prompts_safe, prompts_over = _build_prompt_sets(
                    args.n_samples, seed=args.seed
                )
                h_harm, h_safe, h_over = _extract_all(
                    prompts_harm, prompts_safe, prompts_over,
                    model, tokenizer, system_prompt,
                    layers, args.batch_size, args.device, args.token_position,
                )

                geom = compute_checkpoint_geometry(
                    checkpoint_tag=checkpoint_tag,
                    system_prompt_key=sp_key,
                    hidden_harmful=h_harm,
                    hidden_safe=h_safe,
                    hidden_over=h_over,
                    method=args.probe_method,
                    layers=layers,
                )
                save_geometry(geom, cache_path)

            geometries.append(geom)

        if model is not None:
            del model  # free GPU after all system prompts for this checkpoint

    _save_summary_csv(geometries, results_dir / "geometry" / "summary_exp3.csv")
    logger.info("Exp 3 done. Saved %d (checkpoint × system_prompt) geometries.", len(geometries))


# ---------------------------------------------------------------------------
# Optional: correlate geometry with I/O over-refusal rate
# ---------------------------------------------------------------------------

def _correlate_with_io(
    geom: CheckpointGeometry,
    io_csv: Path,
    checkpoint_tag: str,
    system_prompt_key: str,
    results_dir: Path,
) -> None:
    """
    For each layer, compute Pearson r between the entanglement score and
    the over-refusal rate (FP rate) estimated from run_experiment output.

    The over-refusal rate for this (checkpoint, system_prompt) pair is a
    scalar, so the correlation is across layers within one checkpoint.
    We also emit a joint CSV for downstream multi-checkpoint analysis.
    """
    df = pd.read_csv(io_csv)

    if "checkpoint" not in df.columns:
        subset = df
    else:
        # checkpoint column in raw_results.csv follows the pattern
        # "{stage}__{system_prompt}" (e.g. "sft__none"), matching checkpoint_tag directly.
        subset = df[df["checkpoint"] == checkpoint_tag]
        if subset.empty:
            logger.warning(
                "No rows matching checkpoint='%s' in %s. "
                "Check that checkpoint tags in raw_results.csv match config tags.",
                checkpoint_tag, io_csv,
            )
            return

    # FP rate: safe prompts (label=0) that were refused (predicted_refusal=1)
    safe_df = subset[subset["label"] == 0] if "label" in subset.columns else pd.DataFrame()
    if len(safe_df) == 0:
        logger.warning("No label=0 rows in I/O CSV for this checkpoint/system_prompt.")
        return

    fp_rate = safe_df["predicted_refusal"].mean() if "predicted_refusal" in safe_df else None
    if fp_rate is None:
        return

    out = results_dir / "geometry" / f"{checkpoint_tag}__{system_prompt_key}_io_corr.csv"
    rows = [
        {
            "layer":           lg.layer,
            "entanglement":    lg.entanglement,
            "boundary_margin": lg.boundary_margin,
            "fp_rate":         fp_rate,   # same scalar for all layers
        }
        for lg in geom.layers
    ]
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("I/O correlation table saved to %s  (fp_rate=%.3f)", out, fp_rate)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--config", required=True,
        help="Config module name, e.g. config_olmo2"
    )
    p.add_argument(
        "--experiment", required=True,
        choices=["entanglement", "evolution", "system_prompt"],
        help=(
            "entanglement: Exp 1 — single checkpoint geometry + I/O correlation; "
            "evolution: Exp 2 — track entanglement across training stages; "
            "system_prompt: Exp 3 — compare system prompts."
        ),
    )

    # Exp 1 specific
    p.add_argument("--checkpoint", default=None,
                   help="[Exp 1] Checkpoint tag (e.g. sft__none)")
    p.add_argument("--system-prompt", default="none",
                   help="System prompt key defined in config SYSTEM_PROMPTS")

    # Shared
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of prompts per category")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layer-stride", type=int, default=1,
                   help="Analyse every N-th layer (1 = all layers)")
    p.add_argument("--probe-method", default="logistic",
                   choices=["logistic", "mean_diff"],
                   help="Method for fitting linear probe direction")
    p.add_argument("--token-position", default="last",
                   choices=["last", "first", "mean"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached .npz exists")

    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = _load_config(args.config)

    if args.experiment == "entanglement":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --experiment entanglement")
        run_entanglement(args, cfg)

    elif args.experiment == "evolution":
        run_evolution(args, cfg)

    elif args.experiment == "system_prompt":
        run_system_prompt(args, cfg)


if __name__ == "__main__":
    main()