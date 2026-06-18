#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="beh-probe-no-beaver"
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/beh-probe-no-beaver.out

cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

echo "=== Start: $(date) ==="
python analysis/compute_behavioural_probe.py \
    --exclude-sources beavertails \
    --out results/olmo2/geometry/behavioral_probe_no_beaver.csv
echo "=== Done: $(date) ==="