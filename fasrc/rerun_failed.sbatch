#!/bin/bash
#SBATCH --job-name=sudokuocr_fail
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19%6
#SBATCH --output=fasrc/logs/slurm_fail_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_fail_%A_%a.err

set -euo pipefail

cd ~/SudokuOCR
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source ~/.envs/sudokuocr/bin/activate

set -a
source ~/SudokuOCR/.env
set +a

export PYTHONUNBUFFERED=1
export GEMINI_MIN_INTERVAL=2.0

if [ -n "${HARVARD_GEMINI_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
    N_KEYS=${#KEYS[@]}
    KEY_IDX=$((SLURM_ARRAY_TASK_ID % N_KEYS))
    export HARVARD_GEMINI_API_KEY="${KEYS[$KEY_IDX]}"
    echo "Using API key $((KEY_IDX + 1))/${N_KEYS} for chunk ${SLURM_ARRAY_TASK_ID}"
fi

echo "=== Failed-age rerun worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

python3 batch_production.py \
    --image-list ~/census_toc/rerun_failed_age_images.txt \
    --results ~/census_toc/ocr_results \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --total-chunks "$SLURM_ARRAY_TASK_COUNT" \
    --workers 1 \
    --timeout 900 \
    --min-interval 2.0 \
    --max-cost 50 \
    --rerun-failures

echo "=== Worker ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
