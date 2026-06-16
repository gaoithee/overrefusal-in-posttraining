#!/bin/bash
# run_entanglement.sh
# Calcola entanglement per tutte le combinazioni token_position x method
# e produce i plot di confronto.

set -e
cd /u/scandussio/overrefusal-in-posttraining

GEO="results/olmo2/geometry"
mkdir -p "$GEO"

echo "=== [1/4] last_prompt x mean_diff ==="
python compute_entanglement.py \
    --token-position last_prompt \
    --method mean_diff \
    --out "$GEO/ent_last_meandiff.csv"

echo "=== [2/4] last_prompt x logistic ==="
python compute_entanglement.py \
    --token-position last_prompt \
    --method logistic \
    --out "$GEO/ent_last_logistic.csv"

echo "=== [3/4] first_gen x mean_diff ==="
python compute_entanglement.py \
    --token-position first_gen \
    --method mean_diff \
    --out "$GEO/ent_first_meandiff.csv"

echo "=== [4/4] first_gen x logistic ==="
python compute_entanglement.py \
    --token-position first_gen \
    --method logistic \
    --out "$GEO/ent_first_logistic.csv"

echo "=== Plot singolo (last_prompt x mean_diff) ==="
python plot_entanglement_curves.py \
    --csv "$GEO/ent_last_meandiff.csv" \
    --out "$GEO/"

echo "=== Plot confronto 4 combinazioni ==="
python plot_entanglement_curves.py \
    --csv-last-meandiff  "$GEO/ent_last_meandiff.csv" \
    --csv-last-logistic  "$GEO/ent_last_logistic.csv" \
    --csv-first-meandiff "$GEO/ent_first_meandiff.csv" \
    --csv-first-logistic "$GEO/ent_first_logistic.csv" \
    --out "$GEO/"

echo "=== Tutto fatto. File in $GEO/ ==="
ls -lh "$GEO/"
