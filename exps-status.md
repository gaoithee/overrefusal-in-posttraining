# Over-refusal in Post-Training — Results

## Setup

- **Model:** OLMo-2 7B
- **Checkpoints:** `base__none`, `base__mistral_safety`, `sft__none`, `sft__mistral_safety`, `dpo__none`, `dpo__mistral_safety`, `final__none`, `final__mistral_safety`
- **Datasets:** or_bench (4000), false_reject (4000), toxicchat (3984), harmbench (2536), beavertails (7544)
- **Groups:**
  - `harmful`: label=1
  - `pseudo_harmful`: label=0, source ∈ {or_bench, false_reject}
  - `harmless`: label=0, source ∈ {toxicchat, harmbench, beavertails} with label=0
- **Judge:** GPT-OSS-120B — Goal Addressness (GA, 0–3) and Partial Distancing (PD, 0/1). Compliance = GA≥2 AND PD=0
- **Activations:** layers 8, 16, 19, 24, 26, 31 — positions `last_prompt` and `first_gen`
- **Directions:**
  - `v_ref = mean(h_harmful) - mean(h_harmless)` — refusal direction
  - `v_over = mean(h_pseudo_harmful) - mean(h_harmless)` — over-refusal direction
  - Both computed per checkpoint and layer on `first_gen` activations

---

## 1. Behavioral Results (I/O)

`results/olmo2/raw_results.csv`, `results/olmo2/metrics_judge.csv`

| checkpoint | recall harmful | compliance pseudo-harmful | judge_GA (pseudo) | judge_PD (pseudo) |
|---|---|---|---|---|
| base__none | 13% | 98.1% | 2.43 | 0.04 |
| base__mistral_safety | — | 96.6% | 2.22 | 0.08 |
| sft__none | 84% | 65.6% | 1.62 | 0.44 |
| sft__mistral_safety | — | 64.7% | 1.56 | 0.47 |
| dpo__none | 78% | 79.0% | 1.87 | 0.33 |
| dpo__mistral_safety | — | 74.0% | 1.90 | 0.37 |
| final__none | ~78% | 78.6% | 1.89 | 0.33 |
| final__mistral_safety | — | 74.0% | 1.89 | 0.36 |

**Key findings:**
- SFT is the main agent of over-refusal: compliance drops from 98% to 66% in one step
- DPO recovers compliance (+13pp) without recovering recall (-6pp)
- final ≈ DPO — no distinguishable contribution from the third step
- Mistral safety system prompt always worsens compliance without proportional recall gain
- Geometry predicts PD (nearly monotone relationship) but not GA (DPO breaks the monotone pattern)

---

## 2. Geometry — last_prompt

`results/olmo2/geometry/ent_last_meandiff.csv`, `ent_last_logistic.csv`

`boundary_margin_n ≈ -1.3` constant across all checkpoints and all layers.
Entanglement small and stable (~-0.05 to -0.09, mean_diff).

A faint signal exists at last_prompt: pseudo-harmful sit slightly higher along v_over than harmless even in the base model, but this pattern is **identical across all checkpoints** — post-training does not touch it.

**Finding:** Training does not modify how the model encodes the input prompt. The last_prompt/first_gen dissociation is complete.

---

## 3. Geometry — first_gen, Entanglement

`results/olmo2/geometry/ent_first_meandiff.csv`

| checkpoint | layer 8 | layer 19 | layer 26 | layer 31 |
|---|---|---|---|---|
| base__none | 0.10 | 0.38 | 0.44 | 0.52 |
| sft__none | 0.13 | 0.17 | 0.22 | 0.11 |
| dpo__none | 0.02 | 0.05 | 0.09 | 0.03 |
| final__none | 0.01 | 0.04 | 0.08 | 0.02 |

In base, entanglement grows with depth — v_ref and v_over become more aligned in deeper layers.
SFT dramatically reduces entanglement. DPO makes them nearly orthogonal (~0.05).

---

## 4. Geometry — first_gen, Boundary Margin + Behavioral Dissociation

`results/olmo2/geometry/ent_first_meandiff.csv` + `raw_results.csv`

| checkpoint | boundary_margin_n (avg layers) | compliance pseudo-harmful | judge_GA | judge_PD |
|---|---|---|---|---|
| base__none | +0.66 | 98.1% | 2.43 | 0.04 |
| sft__none | -0.68 | 65.6% | 1.62 | 0.44 |
| dpo__none | -0.92 | 79.0% | 1.87 | 0.33 |
| final__none | -0.93 | 78.6% | 1.89 | 0.33 |

**Key dissociation:** DPO has a more negative boundary_margin_n than SFT but higher compliance.
The geometry moves in the "wrong" direction but behavior improves.

- Geometry predicts PD (nearly monotone): more negative margin → higher distancing rate
- Geometry does NOT predict GA (non-monotone): DPO has more negative margin but higher GA than SFT
- DPO acts on something not captured by linear projections of the residual stream — likely token probability distributions

---

## 5. v_over Does Not Predict Behavior — v_beh ∥ v_ref

Computed on `first_gen` activations, layer 19.

`v_beh = mean(h_pseudo_refused) - mean(h_pseudo_not_refused)`

| checkpoint | cos(v_beh, v_ref) | cos(v_beh, v_over) |
|---|---|---|
| sft__none | 0.82 | -0.14 |
| dpo__none | 0.86 | -0.25 |
| final__none | 0.86 | -0.27 |

Values are consistent across layers 8–31 (cos(v_beh, v_ref) ranges 0.81–0.88 in SFT, 0.85–0.88 in DPO/final).

**Finding:** Over-refusal is mediated by v_ref, not v_over. The model over-refuses pseudo-harmful prompts because it projects them along the same direction as genuinely harmful prompts, not along v_over. v_over describes the structure of the pseudo-harmful category but is not causally responsible for the refusal decision.

---

## 6. Refused and Non-Refused Pseudo-Harmful Are Geometrically Opposite

`cos(v_refused, v_not_refused)` on raw activations, `first_gen`.

| checkpoint | layer 8 | layer 19 | layer 26 | layer 31 |
|---|---|---|---|---|
| sft__none | -0.24 | -0.37 | -0.53 | -0.53 |
| dpo__none | -0.15 | -0.29 | -0.42 | -0.41 |
| final__none | -0.14 | -0.29 | -0.42 | -0.42 |

Separation grows with depth and is strongest in SFT. Negative cosine means the two sub-groups point in opposite directions relative to harmless. v_over is a compromise direction — an average of two opposite signals.

---

## 7. Three-Category Structure Exists in Raw Activation Space

`results/olmo2/classifiers/clf3_*.pkl` — Logistic probe, 5-fold cross-val, no projection onto v_ref/v_over.

| checkpoint | layer | acc_3class | acc pseudo vs harmful | acc pseudo vs harmless |
|---|---|---|---|---|
| base__none | 8 | 0.813 | 0.954 | 0.955 |
| base__none | 16 | 0.847 | 0.968 | 0.961 |
| base__none | 19 | 0.839 | 0.970 | 0.962 |
| base__none | 24 | — | — | — |
| base__none | 26 | — | — | — |
| base__none | 31 | — | — | — |
| sft__none | 8 | — | — | — |
| ... | ... | ... | ... | ... |

*(full results in `slurm_outputs/clf-3cat.out` — run in progress)*

**Finding (partial):** The three-category structure is separable already in the base model at layer 8 with 81% 3-class accuracy and 95%+ pairwise accuracy. The structure is not created by post-training and is not an artifact of v_over — the probe operates on raw activations.

---

## 8. Geometry Stabilizes After SFT — Centroid Cosines Cross-Checkpoint

`results/olmo2/geometry/centroid_cosines.csv`

Layer 19:

| category | base→sft | base→dpo | base→final | sft→dpo | sft→final | dpo→final |
|---|---|---|---|---|---|---|
| harmful | 0.646 | 0.643 | 0.644 | 0.986 | 0.985 | 0.9997 |
| pseudo_harm | 0.700 | 0.769 | 0.769 | 0.939 | 0.937 | 0.9996 |
| harmless | 0.819 | 0.808 | 0.807 | 0.978 | 0.977 | 0.9997 |

Pattern consistent across all layers (8–31): base→SFT shift is 0.58–0.84, SFT→DPO is 0.93–0.99, DPO→final is >0.999.

**Finding:** The real geometric shift happens between base and SFT. After SFT, DPO and final do not reorganize representations. Yet behavior changes between SFT and DPO. This is the cleanest evidence of the two-level dissociation.

---

## 9. Two Levels of Optimization

**Level 1 — Geometric:** where representations sit in activation space. Established by SFT, stable afterwards (centroid cosines SFT→DPO >0.93, DPO→final >0.999). Measurable with semantic probe, centroid cosines, entanglement.

**Level 2 — Probabilistic:** how the model maps representations to token distributions. Calibrated by DPO — shifts the decision boundary without moving representations. Measurable with the compliance/boundary_margin_n dissociation and the GA vs boundary_margin_n non-monotone relationship.

Geometry predicts Partial Distancing (mediated by Level 1). Goal Addressness depends on Level 2.

---

## Experiments In Progress

| experiment | script | output |
|---|---|---|
| 3-class semantic probe, all checkpoints | `run_classification.py` | `slurm_outputs/clf-3cat.out` |
| Behavioral probe + cross-checkpoint transfer | `compute_behavioural_probe.py` | `slurm_outputs/beh-probe.out` |

---

## Key Scripts

| script | purpose |
|---|---|
| `analysis/extract_and_push.py` | extract activations → HuggingFace |
| `compute_entanglement.py` | entanglement + boundary margin per layer/checkpoint |
| `compute_centroid_cosines.py` | centroid cosine similarity cross-checkpoint |
| `run_classification.py` | 3-class semantic probe + cross-checkpoint transfer |
| `compute_behavioural_probe.py` | behavioral probe (predicts predicted_refusal) |
| `plot_2d_refusal_space.py` | 2D projection plots |
| `plot_2d_refusal_space_behavioral.py` | 2D plots with refused/non-refused pseudo-harmful |
| `plot_entanglement_curves.py` | entanglement and boundary margin curves |
| `run_experiment.py` | generate responses for all checkpoints |
| `evaluation/llm_judge.py` | run GPT judge on responses |

---

## Figures

| file | content |
|---|---|
| `figures/2d_first_gen_naive_all.png` | 2D refusal space, all sources, first_gen |
| `figures/2d_last_prompt_naive_all.png` | 2D refusal space, last_prompt (flat across checkpoints) |
| `figures/behavioral/2d_first_gen_naive_all.png` | 2D with refused (triangle) vs non-refused (square) pseudo-harmful |
| `figures/by_source/` | 2D plots split by dataset source |
| `figures/by_category/` | 2D plots split by harm category |
| `results/olmo2/geometry/plots/fig_entanglement_all_combos.png` | entanglement curves, all token positions × methods |
| `results/olmo2/geometry/plots/fig_boundary_all_combos.png` | boundary margin curves |
| `results/olmo2/geometry/plots/fig_last_vs_first_entanglement.png` | last_prompt vs first_gen comparison |
| `figures/umap_4cat_grid.png` | UMAP with 4 behavioral categories |
