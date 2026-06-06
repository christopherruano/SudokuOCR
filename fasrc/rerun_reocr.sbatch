#!/bin/bash
#SBATCH --job-name=sudokuocr_reocr
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-39%4
#SBATCH --output=fasrc/logs/slurm_reocr_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_reocr_%A_%a.err
#
# Re-OCR the 409 review/QA-flagged tables (408 distinct page images)
# with the UPDATED prompt (decimal preservation + reads real sub-columns incl.
# Unmarried/Married/Widowed, multi-table split, rotation). %4 = 4 concurrent
# workers, one per active Harvard Gemini key.
set -euo pipefail
cd ~/SudokuOCR
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source ~/.envs/sudokuocr/bin/activate
set -a; source ~/SudokuOCR/.env; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY
export PYTHONUNBUFFERED=1
export GEMINI_MIN_INTERVAL=2.0
# MoE fallback ON: this is a small set of KNOWN-hard tables, worth the extra readings.
export SUDOKUOCR_MOE_FALLBACK=1

IMAGE_LIST=${IMAGE_LIST:-$HOME/census_toc/reocr_images.txt}
RESULTS_DIR=${RESULTS_DIR:-$HOME/census_toc/ocr_results}

if [ -n "${HARVARD_GEMINI_API_KEYS:-}" ]; then
  IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
  N_KEYS=${#KEYS[@]}
  export HARVARD_GEMINI_API_KEY="${KEYS[$((SLURM_ARRAY_TASK_ID % N_KEYS))]}"
  echo "Using key $((SLURM_ARRAY_TASK_ID % N_KEYS + 1))/${N_KEYS}"
fi

echo "=== reocr worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) at $(date) ==="
python3 batch_production.py \
    --image-list "$IMAGE_LIST" \
    --results "$RESULTS_DIR" \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --total-chunks "$SLURM_ARRAY_TASK_COUNT" \
    --workers 1 --timeout 1800 --min-interval 2.0 --max-api-calls 12 --max-cost 60
echo "=== worker ${SLURM_ARRAY_TASK_ID} done at $(date) ==="
