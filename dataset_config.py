"""
datasets_config.py

Single source of truth for all benchmark datasets.
Imported by every model config (config.py, config_olmo2.py, …).

Dataset types
-------------
  "over_refusal"  — all prompts are SAFE (label=0). Measures FP (over-refusal).
                    A good model should answer all of them.
  "harmful"       — all prompts are HARMFUL (label=1). Measures FN (under-refusal).
                    A good model should refuse all of them.
  "mixed"         — contains both safe (0) and harmful (1) prompts.
                    Measures both FP and FN simultaneously.
"""

from dataclasses import dataclass
from typing import Literal, Optional


DatasetType = Literal["over_refusal", "harmful", "mixed"]


@dataclass
class DatasetConfig:
    name: str
    dataset_type: DatasetType       # drives which metrics are computed
    hf_path: str                    # HuggingFace dataset path
    hf_split: str = "test"
    prompt_column: str = "prompt"
    label_column: Optional[str] = None   # None → use dataset_type to assign fixed label
    max_samples: Optional[int] = None    # None = use all available


# ---------------------------------------------------------------------------
# Over-refusal benchmarks  (label = 0, FP measurement)
# ---------------------------------------------------------------------------

OR_BENCH = DatasetConfig(
    name="or_bench", # Uniformato
    dataset_type="over_refusal",
    hf_path="bench-llm/or-bench",
    hf_split="train",
    prompt_column="prompt",
    max_samples=500,
)

FALSE_REJECT = DatasetConfig(
    name="false_reject", # Uniformato
    dataset_type="over_refusal",
    hf_path="AmazonScience/FalseReject",
    hf_split="test",
    prompt_column="prompt",
    max_samples=500,
)

# ---------------------------------------------------------------------------
# Harmful benchmarks  (label = 1, FN measurement)
# ---------------------------------------------------------------------------

HARMBENCH = DatasetConfig(
    name="harmbench",
    dataset_type="harmful",
    hf_path="allenai/tulu-3-harmbench-eval",
    hf_split="test", # CORRETTO (era train)
    prompt_column="prompt",
)

JAILBREAKBENCH = DatasetConfig(
    name="jailbreakbench",
    dataset_type="harmful",
    hf_path="JailbreakBench/JBB-Behaviors",
    hf_split="behaviors", # TECNICA: Molti loader HF usano lo split come config se il secondo parametro è None
    prompt_column="Goal",
)

# ---------------------------------------------------------------------------
# Mixed benchmarks  (label = 0 and 1, FP + FN measurement)
# ---------------------------------------------------------------------------

WILDGUARD = DatasetConfig(
    name="wildguard",
    dataset_type="mixed",
    hf_path="allenai/wildguardmix",
    hf_split="test", # Verificato
    label_column="prompt_harm_label",
    max_samples=1000,
)

TOXICCHAT = DatasetConfig(
    name="toxicchat", # Uniformato
    dataset_type="mixed",
    hf_path="lmsys/toxic-chat",
    hf_split="test",
    prompt_column="user_input",
    label_column="toxicity",
    max_samples=500,
)

BEAVERTAILS = DatasetConfig(
    name="beavertails",
    dataset_type="mixed",
    hf_path="PKU-Alignment/BeaverTails",
    hf_split="30k_test", # CORRETTO (era test)
    label_column="is_safe",
    max_samples=1000,
)

# ---------------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------------

# All datasets, grouped by type — for easy reference in configs and scripts
ALL_DATASETS: dict[str, DatasetConfig] = {
    "or_bench":       OR_BENCH,
    "false_reject":   FALSE_REJECT,
    "harmbench":      HARMBENCH,
    "jailbreakbench": JAILBREAKBENCH,
    "wildguard":      WILDGUARD,
    "toxicchat":      TOXICCHAT,
    "beavertails":    BEAVERTAILS,
}

OVER_REFUSAL_DATASETS: list[str] = ["or_bench", "false_reject"]
HARMFUL_DATASETS:      list[str] = ["harmbench", "jailbreakbench"]
MIXED_DATASETS:        list[str] = ["wildguard", "toxicchat", "beavertails"]
