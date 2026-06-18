#!/bin/bash
# regen_figures.sh
# Rigenera tutte le figure 2D senza beavertails usando plot_2d_refusal_space.py
#
# Usage: bash regen_figures.sh

set -e
cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate
export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

SCRIPT="python analysis/plot_2d_refusal_space_behavioral.py"
EXCLUDE="--exclude-sources beavertails"

echo "=== Start: $(date) ==="

# 1. figura principale — tutti i dati, first_gen
echo "[1/6] Plot principale first_gen ..."
$SCRIPT $EXCLUDE \
    --position first_gen \
    --split-by none \
    --output-dir figures/

# 2. figura principale — last_prompt
echo "[2/6] Plot principale last_prompt ..."
$SCRIPT $EXCLUDE \
    --position last_prompt \
    --split-by none \
    --output-dir figures/

# 3. split per source — first_gen
echo "[3/6] Split per source (first_gen) ..."
$SCRIPT $EXCLUDE \
    --position first_gen \
    --split-by source \
    --output-dir figures/by_source/

# 4. split per source — last_prompt
echo "[4/6] Split per source (last_prompt) ..."
$SCRIPT $EXCLUDE \
    --position last_prompt \
    --split-by source \
    --output-dir figures/by_source/

# 5. split per category — first_gen
echo "[5/6] Split per category (first_gen) ..."
$SCRIPT $EXCLUDE \
    --position first_gen \
    --split-by category \
    --output-dir figures/by_category/

# 6. versione ortogonalizzata — first_gen
echo "[6/6] Plot ortogonalizzato first_gen ..."
$SCRIPT $EXCLUDE \
    --position first_gen \
    --split-by none \
    --orthogonalize \
    --output-dir figures/

echo "=== Done: $(date) ==="