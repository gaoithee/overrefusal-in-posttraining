#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="clf-3cat"
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/clf-3cat.out

cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

echo "=== Start: $(date) ==="
python run_classification.py
echo "=== Done: $(date) ==="
