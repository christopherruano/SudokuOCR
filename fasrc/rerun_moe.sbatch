#!/bin/bash
#SBATCH --job-name=sudokuocr_moe
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19%6
#SBATCH --output=fasrc/logs/slurm_moe_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_moe_%A_%a.err

set -euo pipefail

cd ~/SudokuOCR
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source ~/.envs/sudokuocr/bin/activate

set -a
source ~/SudokuOCR/.env
set +a

# Gemini-only MoE: unset other API keys so pipeline auto-detects only Gemini
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

export PYTHONUNBUFFERED=1
export GEMINI_MIN_INTERVAL=2.0
export SUDOKUOCR_MOE_FALLBACK=1
export SUDOKUOCR_MOE_FALLBACK_ANY_FAILURE=0
export SUDOKUOCR_MOE_FAST_MODE=0

IMAGE_LIST=${IMAGE_LIST:-$HOME/census_toc/moe_all_images.txt}
RESULTS_DIR=${RESULTS_DIR:-$HOME/census_toc/ocr_results}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}

if [ -n "${HARVARD_GEMINI_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
    N_KEYS=${#KEYS[@]}
    KEY_IDX=$((SLURM_ARRAY_TASK_ID % N_KEYS))
    export HARVARD_GEMINI_API_KEY="${KEYS[$KEY_IDX]}"
    echo "Using API key $((KEY_IDX + 1))/${N_KEYS} for chunk ${SLURM_ARRAY_TASK_ID}"
fi

echo "=== MoE fallback rerun worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

python3 batch_production.py \
    --image-list "$IMAGE_LIST" \
    --results "$RESULTS_DIR" \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --total-chunks "$SLURM_ARRAY_TASK_COUNT" \
    --workers 1 \
    --timeout "$TIMEOUT_SECONDS" \
    --min-interval 2.0 \
    --max-cost 50 \
    --rerun-failures

echo "=== Worker ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
