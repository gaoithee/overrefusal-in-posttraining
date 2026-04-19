"""
run_experiment.py taken from https://github.com/gaoithee/overrefusal-map/blob/main/run_experiment.py

Step 1 of the pipeline: generate responses for every checkpoint × dataset × system prompt.
Keyword-based refusal detection is applied inline. The LLM judge runs separately (run_judge.py).

Usage
-----
    # Full run — all checkpoints, all datasets in config, all system prompts
    python run_experiment.py --config config_olmo2

    # Select specific subsets
    python run_experiment.py --config config_olmo2 --checkpoints base sft --datasets or_bench wildguard

    # Dry-run: load data, print stats, skip generation
    python run_experiment.py --config config_olmo2 --dry-run

    # Recompute metrics from an existing CSV without re-generating
    python run_experiment.py --config config_olmo2 --load-results results/olmo2/raw_results.csv
"""

import argparse
import importlib
import logging
import os
from itertools import islice

import pandas as pd

from data.dataset_loader import load_all_datasets
from evaluation.refusal_detector import detect_refusal_batch
from evaluation.metrics import (
    compute_metrics,
    compute_metrics_by_category,
    print_summary,
    save_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def run_generation(model, prompts: list[str], batch_size: int = 8) -> list[str]:
    responses = []
    total = len(prompts)
    n_batches = (total + batch_size - 1) // batch_size
    for i, batch in enumerate(_batched(prompts, batch_size)):
        logger.info("  [%s] batch %d/%d", model.checkpoint_name, i + 1, n_batches)
        responses.extend(model.generate_batch(batch))
    return responses


def _safe_merge(existing: pd.DataFrame, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Concatenate existing and new rows, deduplicating where possible.

    Returns (merged_df, n_added).

    The original code called drop_duplicates(subset=dedup_cols) where
    dedup_cols could be an empty list if none of the three key columns
    existed in both DataFrames.  pandas raises ValueError on
    drop_duplicates(subset=[]), crashing the run and losing all newly
    generated responses.  This function guards against that case by
    skipping deduplication when no key columns are available.
    """
    before = len(existing)
    dedup_cols = [c for c in ["prompt", "checkpoint", "source"]
                  if c in existing.columns and c in raw_df.columns]
    combined = pd.concat([existing, raw_df], ignore_index=True)
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="first")
    else:
        logger.warning(
            "No dedup columns found in common between existing CSV and new results "
            "(%s). Skipping deduplication — manual review may be needed.",
            list(existing.columns),
        )
    added = len(combined) - before
    return combined, added


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_and_save_metrics(raw_df: pd.DataFrame, results_dir: str) -> None:
    """
    Compute and save metrics, respecting dataset_type:
      - over_refusal datasets → report FP rate only
      - harmful datasets       → report FN rate only
      - mixed datasets         → report both
      - overall                → all rows combined
    """
    os.makedirs(results_dir, exist_ok=True)

    # Overall (all datasets)
    m = compute_metrics(raw_df)
    print_summary(m, title="Overall metrics (all datasets)")
    save_metrics(m, os.path.join(results_dir, "metrics_overall.csv"))

    # Per-category
    m_cat = compute_metrics_by_category(raw_df)
    save_metrics(m_cat, os.path.join(results_dir, "metrics_by_category.csv"))

    # Per-source, with type-aware reporting
    if "source" not in raw_df.columns:
        return

    for source, grp in raw_df.groupby("source"):
        m_src = compute_metrics(grp)
        save_metrics(m_src, os.path.join(results_dir, f"metrics_{source}.csv"))

        n_safe    = (grp["label"] == 0).sum()
        n_harmful = (grp["label"] == 1).sum()
        if n_harmful == 0:
            title = f"{source}  [over-refusal benchmark — FP rate]"
        elif n_safe == 0:
            title = f"{source}  [harmful benchmark — FN rate]"
        else:
            title = f"{source}  [mixed benchmark — FP + FN rates]"

        print_summary(m_src, title=title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = importlib.import_module(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── 1. Select datasets ─────────────────────────────────────────────────
    if args.datasets:
        missing = [k for k in args.datasets if k not in cfg.DATASETS]
        if missing:
            logger.error(
                "Unknown dataset key(s): %s. Available in this config: %s",
                missing, list(cfg.DATASETS.keys())
            )
            return
        selected_datasets = {k: cfg.DATASETS[k] for k in args.datasets}
    else:
        selected_datasets = cfg.DATASETS

    # ── 2. Load data ───────────────────────────────────────────────────────
    logger.info("Loading datasets: %s", list(selected_datasets.keys()))
    data_df = load_all_datasets(selected_datasets)

    safe_count    = (data_df["label"] == 0).sum()
    harmful_count = (data_df["label"] == 1).sum()
    logger.info("Prompts: %d total  (%d safe / %d harmful)",
                len(data_df), safe_count, harmful_count)

    if len(data_df) == 0:
        logger.error("No prompts loaded — check dataset config.")
        return

    # ── 3. Dry run ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDataset breakdown:")
        print(data_df.groupby(["source", "label"]).size().rename("count").to_string())
        print("\nSample prompts:")
        print(data_df[["source", "label", "prompt"]].head(10).to_string())
        return

    # ── 4. Load results from CSV (skip generation) ─────────────────────────
    if args.load_results:
        logger.info("Loading pre-computed results from %s", args.load_results)
        raw_df = pd.read_csv(args.load_results)
        compute_and_save_metrics(raw_df, cfg.RESULTS_DIR)
        return

    # ── 5. Select checkpoints and system prompts ───────────────────────────
    if args.checkpoints:
        missing = [k for k in args.checkpoints if k not in cfg.OLMO_CHECKPOINTS]
        if missing:
            logger.error("Unknown checkpoint(s): %s", missing)
            return
        selected_checkpoints = {k: cfg.OLMO_CHECKPOINTS[k] for k in args.checkpoints}
    else:
        selected_checkpoints = cfg.OLMO_CHECKPOINTS

    if args.system_prompts:
        selected_prompts = {k: cfg.SYSTEM_PROMPTS[k] for k in args.system_prompts}
    else:
        selected_prompts = cfg.SYSTEM_PROMPTS

    # ── 6. Generate ────────────────────────────────────────────────────────
    from models.olmo_loader import iter_checkpoints

    all_rows = []
    prompts  = data_df["prompt"].tolist()

    for model in iter_checkpoints(selected_checkpoints, selected_prompts, cfg.GENERATION):
        logger.info("Running inference: %s  (%d prompts)", model.checkpoint_name, len(prompts))
        responses = run_generation(model, prompts, cfg.GENERATION.batch_size)
        refusals  = detect_refusal_batch(responses)

        for idx, (resp, ref) in enumerate(zip(responses, refusals)):
            row = data_df.iloc[idx].to_dict()
            row["checkpoint"]        = model.checkpoint_name
            row["response"]          = resp
            row["predicted_refusal"] = int(ref)
            all_rows.append(row)

        model.unload()

    raw_df = pd.DataFrame(all_rows)

    # ── 7. Save (merge with existing, never overwrite) ─────────────────────
    raw_path = os.path.join(cfg.RESULTS_DIR, "raw_results.csv")
    if os.path.exists(raw_path):
        existing         = pd.read_csv(raw_path)
        before           = len(existing)
        combined, added  = _safe_merge(existing, raw_df)
        skipped          = len(raw_df) - added
        combined.to_csv(raw_path, index=False)
        logger.info(
            "Merged: %d existing + %d new → %d total (%d duplicates skipped)",
            before, len(raw_df), len(combined), skipped,
        )
        raw_df = combined
    else:
        raw_df.to_csv(raw_path, index=False)
        logger.info("Saved %d rows → %s", len(raw_df), raw_path)

    # ── 8. Metrics ─────────────────────────────────────────────────────────
    compute_and_save_metrics(raw_df, cfg.RESULTS_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses and compute refusal metrics across OLMo checkpoints."
    )
    parser.add_argument(
        "--config", default="config",
        help="Config module: config, config_olmo2, config_olmo3, config_olmo3_think. Default: config.",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Checkpoint keys (e.g. base sft dpo final). Default: all in config.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset keys (e.g. or_bench wildguard harmbench). Default: all in config.",
    )
    parser.add_argument(
        "--system-prompts", nargs="+", default=None, dest="system_prompts",
        help="System prompt keys (e.g. none mistral_safety). Default: all in config.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load data and print stats only — no generation.",
    )
    parser.add_argument(
        "--load-results", default=None, metavar="CSV",
        help="Skip generation; recompute metrics from an existing raw_results.csv.",
    )
    args = parser.parse_args()
    main(args)