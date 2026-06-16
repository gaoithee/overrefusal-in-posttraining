#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="olmo-act-extract"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/olmo-act-%a.out
#SBATCH --array=0-7
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Checkpoint array — must match CHECKPOINT_TO_HF keys in extract_and_push.py
# ---------------------------------------------------------------------------
CHECKPOINTS=(
    "base__none"
    "base__mistral_safety"
    "sft__none"
    "sft__mistral_safety"
    "dpo__none"
    "dpo__mistral_safety"
    "final__none"
    "final__mistral_safety"
)

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID — checkpoint: $CHECKPOINT ==="
echo "=== Node: $SLURMD_NODENAME | Start: $(date) ==="

# Stagger jobs: each task waits (task_id * 90)s so all 8 commits
# are spread ~12 min apart — well within the 128/hr window.
STAGGER=$(( SLURM_ARRAY_TASK_ID * 90 ))
if [[ $STAGGER -gt 0 ]]; then
    echo "=== Staggering ${STAGGER}s to spread HF API calls ==="
    sleep "$STAGGER"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python/3.11.7-gcc-13.2.0-b7gwkjx

cd /u/scandussio/overrefusal-in-posttraining || exit 1
source .overenv/bin/activate

export HF_HOME=/share/ai-lab/scandussio/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is empty. Check ~/.hf_token"
    exit 1
fi

# Give each job a distinct tmp dir to avoid staging-dir collisions
# (not strictly needed since tempfile.mkdtemp is already unique, but
#  explicit TMPDIR avoids filling up the default /tmp on shared nodes)
export TMPDIR="/scratch/scandussio/tmp/olmo_act_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python analysis/extract_and_push.py \
    --csv results/olmo2/raw_results.csv \
    --hf-repo saracandu/olmo-activations \
    --hf-token "$HF_TOKEN" \
    --checkpoint-filter "$CHECKPOINT" \
    --batch-size 32 \
    --device cuda \

EXIT_CODE=$?

# Clean up tmp dir regardless of exit code
rm -rf "$TMPDIR"

echo "=== Done: $CHECKPOINT | Exit: $EXIT_CODE | End: $(date) ==="
exit $EXIT_CODE