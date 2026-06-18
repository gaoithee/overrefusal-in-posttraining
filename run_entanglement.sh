#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="entanglement-no-beaver"
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/entanglement-no-beaver.out

cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

echo "=== Start: $(date) ==="

echo "[1/4] first_gen + mean_diff ..."
python analysis/compute_entanglement.py \
    --token-position first_gen \
    --method mean_diff \
    --exclude-sources beavertails \
    --out results/olmo2/geometry/ent_first_meandiff.csv

echo "[2/4] first_gen + logistic ..."
python analysis/compute_entanglement.py \
    --token-position first_gen \
    --method logistic \
    --exclude-sources beavertails \
    --out results/olmo2/geometry/ent_first_logistic.csv

echo "[3/4] last_prompt + mean_diff ..."
python analysis/compute_entanglement.py \
    --token-position last_prompt \
    --method mean_diff \
    --exclude-sources beavertails \
    --out results/olmo2/geometry/ent_last_meandiff.csv

echo "[4/4] last_prompt + logistic ..."
python analysis/compute_entanglement.py \
    --token-position last_prompt \
    --method logistic \
    --exclude-sources beavertails \
    --out results/olmo2/geometry/ent_last_logistic.csv

echo "=== Done: $(date) ==="