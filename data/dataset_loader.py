"""
data/dataset_loader.py

Load benchmark datasets into a unified DataFrame with columns:
    prompt    : str  — user message to send to the model
    label     : int  — 0 = safe (should NOT be refused), 1 = harmful (should be refused)
    category  : str  — subcategory string (or "unknown")
    source    : str  — dataset key (e.g. "or_bench", "wildguard")

Label assignment rules per dataset_type:
    "over_refusal" → all rows get label=0
    "harmful"      → all rows get label=1
    "mixed"        → labels read from label_column (with dataset-specific encoding)
"""

import logging
from typing import Optional

import pandas as pd

from datasets_config import DatasetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_or_bench(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, "or-bench-80k", cfg.hf_split)
    df = ds.to_pandas()
    out = _base_frame(df, cfg,
                      prompt_col=_col(df, ["prompt", "text", "question"]),
                      category_col=_col(df, ["category", "type"], required=False))
    out["label"] = 0
    logger.info("OR-Bench: %d prompts loaded", len(out))
    return out


def load_false_reject(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, None, cfg.hf_split)
    df = ds.to_pandas()
    out = _base_frame(df, cfg,
                      prompt_col=_col(df, ["prompt", "text", "question"]),
                      category_col=_col(df, ["category", "domain", "type"], required=False))
    out["label"] = 0
    logger.info("FalseReject: %d prompts loaded", len(out))
    return out


def load_wildguard(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, None, cfg.hf_split)
    df = ds.to_pandas()
    out = _base_frame(df, cfg,
                      prompt_col=_col(df, ["prompt"]),
                      category_col=_col(df, ["category", "subcategory"], required=False))
    # "harmful" → 1, anything else → 0
    out["label"] = (df["prompt_harm_label"].str.lower() == "harmful").astype(int)
    out = _stratified_sample(out, cfg.max_samples)
    logger.info("WildGuard: %d prompts (safe=%d, harmful=%d)",
                len(out), (out["label"] == 0).sum(), (out["label"] == 1).sum())
    return out


def load_harmbench(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, None, cfg.hf_split)
    df = ds.to_pandas()
    out = _base_frame(df, cfg,
                      prompt_col=_col(df, ["prompt", "Behavior", "behavior", "goal", "instruction"]),
                      category_col=_col(df, ["category", "SemanticCategory", "FunctionalCategory"], required=False))
    out["label"] = 1
    if cfg.max_samples:
        out = out.sample(n=min(cfg.max_samples, len(out)), random_state=42)
    logger.info("HarmBench: %d prompts loaded (all harmful)", len(out))
    return out


def load_jailbreakbench(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, None, cfg.hf_split)
    df = ds.to_pandas()
    out = _base_frame(df, cfg,
                      prompt_col=_col(df, ["Goal", "goal", "Behavior", "behavior", "prompt"]),
                      category_col=_col(df, ["Category", "category"], required=False))
    out["label"] = 1
    if cfg.max_samples:
        out = out.sample(n=min(cfg.max_samples, len(out)), random_state=42)
    logger.info("JailbreakBench: %d prompts loaded (all harmful)", len(out))
    return out


def load_toxicchat(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, "toxicchat0124", cfg.hf_split)
    df = ds.to_pandas()
    out = pd.DataFrame()
    out["prompt"]   = df["user_input"]
    out["label"]    = df["toxicity"].astype(int)
    out["category"] = df.get("jailbreaking", pd.Series(0, index=df.index)).apply(
        lambda x: "jailbreak" if int(x) == 1 else "direct"
    )
    out["source"]   = "toxicchat"
    out = _stratified_sample(out, cfg.max_samples)
    logger.info("ToxicChat: %d prompts (safe=%d, harmful=%d)",
                len(out), (out["label"] == 0).sum(), (out["label"] == 1).sum())
    return out.reset_index(drop=True)


def load_beavertails(cfg: DatasetConfig) -> pd.DataFrame:
    ds = _hf_load(cfg.hf_path, None, cfg.hf_split)
    df = ds.to_pandas()

    def _first_harm_cat(cat_dict) -> str:
        if isinstance(cat_dict, dict):
            for k, v in cat_dict.items():
                if v:
                    return k
        return "unknown"

    out = pd.DataFrame()
    out["prompt"]   = df["prompt"]
    out["label"]    = (~df["is_safe"].astype(bool)).astype(int)
    out["category"] = df["category"].apply(_first_harm_cat)
    out["source"]   = "beavertails"
    out = _stratified_sample(out, cfg.max_samples)
    logger.info("BeaverTails: %d prompts (safe=%d, harmful=%d)",
                len(out), (out["label"] == 0).sum(), (out["label"] == 1).sum())
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADER_MAP = {
    "or_bench":       load_or_bench,
    "false_reject":   load_false_reject,
    "wildguard":      load_wildguard,
    "harmbench":      load_harmbench,
    "jailbreakbench": load_jailbreakbench,
    "toxicchat":      load_toxicchat,
    "beavertails":    load_beavertails,
}


def load_dataset_from_config(cfg: DatasetConfig) -> pd.DataFrame:
    target_key = cfg.name.lower().replace(" ", "_").replace("-", "_")
    loader = _LOADER_MAP.get(target_key)
    
    if loader is None:
        raise ValueError(
            f"No loader for dataset '{cfg.name}' (key: {target_key}). "
            f"Available: {list(_LOADER_MAP.keys())}"
        )
    return loader(cfg)

def load_all_datasets(dataset_configs: dict) -> pd.DataFrame:
    """
    Load and concatenate all datasets.
    Logs a breakdown by dataset_type so it's clear what's safe vs harmful.
    """
    frames = []
    for key, cfg in dataset_configs.items():
        try:
            df = load_dataset_from_config(cfg)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", key, exc)

    if not frames:
        raise RuntimeError("No datasets could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    safe    = (combined["label"] == 0).sum()
    harmful = (combined["label"] == 1).sum()
    logger.info("Total prompts loaded: %d  (safe=%d, harmful=%d)", len(combined), safe, harmful)
    return combined


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _hf_load(path: str, config_name: Optional[str], split: str):
    from datasets import load_dataset
    if "JBB-Behaviors" in path:
        return load_dataset(path, "behaviors", split="train")
    if config_name:
        return load_dataset(path, config_name, split=split)
    return load_dataset(path, split=split)

def _col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of {candidates} found in columns {list(df.columns)}"
        )
    return None


def _base_frame(df: pd.DataFrame, cfg: DatasetConfig,
                prompt_col: str, category_col: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["prompt"]   = df[prompt_col]
    out["label"]    = -1    # placeholder; set by caller
    out["category"] = df[category_col] if category_col else "unknown"
    out["source"]   = cfg.name.lower().replace(" ", "_").replace("-", "_")
    if cfg.max_samples and cfg.dataset_type != "mixed":
        out = out.sample(n=min(cfg.max_samples, len(out)), random_state=42)
    return out.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, max_samples: Optional[int]) -> pd.DataFrame:
    if not max_samples:
        return df.reset_index(drop=True)
    n = max_samples // 2
    safe    = df[df["label"] == 0].sample(n=min(n, (df["label"]==0).sum()), random_state=42)
    harmful = df[df["label"] == 1].sample(n=min(n, (df["label"]==1).sum()), random_state=42)
    return pd.concat([safe, harmful]).sample(frac=1, random_state=42).reset_index(drop=True)
