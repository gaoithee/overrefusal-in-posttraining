#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo3-judge"
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/olmo3-judge-%j.out
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python/3.11.7-gcc-13.2.0-b7gwkjx
cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

echo "=== Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date) ==="

# ---------------------------------------------------------------------------
# Orfeo JWT token
# ---------------------------------------------------------------------------
export ORFEO_API_KEY=$(curl -sk \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password&client_id=hdb32CsgkhxaTXut5WEyqqtUNkwtJcAqI8w1VWIH&username=rit-ollama&password=1kT4OyNqr8QID24Hu120PuVAwJo6b7nYCu8KeCMdfUj794BjevH8DVLFBFGi&scope=profile" \
  https://orfeo-auth.areasciencepark.it/application/o/token/ \
  | jq -r '.access_token')

export ORFEO_LLM_URL="https://orfeo-llm.areasciencepark.it/vllm/v1"

if [[ -z "$ORFEO_API_KEY" || "$ORFEO_API_KEY" == "null" ]]; then
    echo "ERROR: failed to obtain Orfeo JWT token"
    exit 1
fi
echo "=== Token obtained: ${ORFEO_API_KEY:0:20}... ==="

echo "ORFEO_API_KEY=$ORFEO_API_KEY" > .env
echo "ORFEO_LLM_URL=$ORFEO_LLM_URL" >> .env

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p slurm_outputs

python run_judge.py \
    --results-dirs ./results/olmo3 \
    --backend api \
    --resume \
    --max-workers 8

EXIT_CODE=$?
echo "=== Done | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE
