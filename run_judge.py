"""
run_judge.py

Post-processing step: run the two-stage LLM judge on one or more
raw_results.csv files produced by run_experiment.py.

Pipeline
--------
  Stage 1 — Coherence
    Filters degenerate base-model outputs (loops, gibberish) before scoring.

  Stage 2 — Two-axis scoring  (francescortu V1)
    GA (Goal Address 0-3): how well the request is satisfied.
    PD (Performative Distancing 0-1): unsolicited hedging / refusal language.

Columns added to raw_results.csv
----------------------------------
  is_coherent      : bool
  judge_ga         : int | None   (0-3, None when incoherent)
  judge_pd         : int | None   (0-1, None when incoherent)
  judge_ga_reason  : str
  judge_pd_reason  : str
  judge_label      : str | None   (backwards-compat 3-class label)

Back-end
--------
  API  : set JUDGE_API_KEY (+ JUDGE_API_BASE_URL) in .env
  Local: falls back to openai/gpt-oss-safeguard-20b when no key is set

Usage
-----
    python run_judge.py --results-dirs results/olmo2
    python run_judge.py --results-dirs results/olmo2 results/olmo3
    python run_judge.py --results-dirs results/olmo2 --backend api
    python run_judge.py --results-dirs results/olmo2 --resume
    python run_judge.py --results-dirs results/olmo2 --dry-run
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from evaluation.llm_judge import build_judge
from evaluation.metrics import (
    compute_judge_metrics,
    print_judge_summary_table,
    save_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_COLUMNS = {
    "is_coherent":     None,
    "judge_ga":        None,
    "judge_pd":        None,
    "judge_ga_reason": None,
    "judge_pd_reason": None,
    "judge_label":     None,
}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in JUDGE_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    return df


def _rows_to_judge(df: pd.DataFrame, resume: bool) -> pd.DataFrame:
    if resume:
        mask = df["is_coherent"].isna()
        logger.info("Resume: %d already judged, %d remaining.", (~mask).sum(), mask.sum())
        return df[mask]
    return df


def run_judge_on_file(csv_path: Path, judge, resume: bool, dry_run: bool) -> None:
    logger.info("=== %s ===", csv_path)
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    todo  = _rows_to_judge(df, resume)
    total = len(todo)
    logger.info("%d rows to judge.", total)

    if dry_run:
        logger.info("Dry-run — first 5 rows:")
        for _, row in todo.head(5).iterrows():
            print(f"\n  checkpoint : {row.get('checkpoint', '?')}")
            print(f"  prompt     : {str(row['prompt'])[:100]}")
            print(f"  response   : {str(row.get('response', ''))[:150]}")
        return

    if total == 0:
        logger.info("Nothing to do.")
        _print_summary(df, csv_path)
        return

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        result = judge.evaluate(str(row["prompt"]), str(row.get("response", "")))
        for col in JUDGE_COLUMNS:
            df.at[idx, col] = result.get(col)

        if i % 50 == 0 or i == total:
            df.to_csv(csv_path, index=False)
            n_done     = int(df["is_coherent"].notna().sum())
            n_coherent = int(df["is_coherent"].fillna(False).sum())
            logger.info(
                "  [%d/%d] saved — judged=%d  coherent=%d  "
                "mean_ga=%.2f  pd_rate=%.1f%%",
                i, total, n_done, n_coherent,
                df["judge_ga"].fillna(0).mean(),
                100 * df["judge_pd"].fillna(0).mean(),
            )

    logger.info("Done: %s", csv_path)
    _print_summary(df, csv_path)

    if "label" in df.columns and "checkpoint" in df.columns:
        try:
            m = compute_judge_metrics(df)
            save_metrics(m, csv_path.parent / "metrics_judge.csv")
            print_judge_summary_table(m)
        except Exception as exc:
            logger.warning("Could not compute judge metrics: %s", exc)


def _print_summary(df: pd.DataFrame, path: Path) -> None:
    total      = len(df)
    n_judged   = int(df["is_coherent"].notna().sum())
    n_coherent = int(df["is_coherent"].fillna(False).sum())
    ga         = df["judge_ga"].dropna()
    pd_col     = df["judge_pd"].dropna()

    print(f"\n{'='*60}")
    print(f"Summary: {path}")
    print(f"{'='*60}")
    print(f"Total rows      : {total}")
    print(f"Judged          : {n_judged}")
    print(f"  Coherent      : {n_coherent}  ({100*n_coherent/total:.1f}%)" if total else "")
    print(f"  Incoherent    : {n_judged - n_coherent}")
    if len(ga):
        print(f"Mean GA         : {ga.mean():.2f}  (0=no answer, 3=full answer)")
        for v in [0, 1, 2, 3]:
            print(f"  GA={v}          : {int((ga==v).sum())}  ({100*(ga==v).mean():.1f}%)")
    if len(pd_col):
        print(f"PD=1 (distancing): {int(pd_col.sum())}  ({100*pd_col.mean():.1f}%)")
    print()

    if "checkpoint" in df.columns:
        print("By checkpoint:")
        for ckpt, grp in df.groupby("checkpoint"):
            n    = len(grp)
            coh  = int(grp["is_coherent"].fillna(False).sum())
            mga  = grp["judge_ga"].dropna().mean()
            pdr  = grp["judge_pd"].dropna().mean()
            print(
                f"  {str(ckpt):<35}  coherent={coh}/{n}  "
                f"mean_ga={mga:.2f}  pd_rate={100*pdr:.1f}%"
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run GA/PD two-axis judge on raw_results.csv files."
    )
    parser.add_argument("--results-dirs", nargs="+", required=True)
    parser.add_argument(
        "--backend", choices=["api", "local"], default=None,
        help="Judge back-end (default: auto-detect from environment).",
    )
    parser.add_argument("--model", default=None, help="Override judge model ID.")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel workers for API back-end.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already judged.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 5 sample rows per file, no judging.")
    args = parser.parse_args()

    csv_paths = []
    for d in args.results_dirs:
        p = Path(d) / "raw_results.csv"
        if not p.exists():
            logger.warning("Not found, skipping: %s", p)
        else:
            csv_paths.append(p)

    if not csv_paths:
        logger.error("No valid raw_results.csv files found.")
        return

    if args.dry_run:
        for p in csv_paths:
            run_judge_on_file(p, judge=None, resume=args.resume, dry_run=True)
        return

    kw = {}
    if args.backend == "api":
        kw["max_workers"] = args.max_workers

    judge = build_judge(backend=args.backend, model=args.model, **kw)
    try:
        for p in csv_paths:
            run_judge_on_file(p, judge, resume=args.resume, dry_run=False)
    finally:
        judge.unload()


if __name__ == "__main__":
    main()
