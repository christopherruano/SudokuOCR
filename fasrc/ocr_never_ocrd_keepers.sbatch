#!/bin/bash
#SBATCH --job-name=sudokuocr_keepers
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-39%4
#SBATCH --output=fasrc/logs/slurm_keepers_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_keepers_%A_%a.err

# OCR the never-OCR'd "keeper" pages identified by visual triage
# (age-structure + education + occupation + caste + birthplace; data tables only).
# These images have NO prior OCR results, so they are processed fresh
# (no --rerun-failures needed).
#
# Setup before submitting:
#   scp export/never_ocrd_triage/ocr/keepers_to_ocr.txt \
#       cruano@login.rc.fas.harvard.edu:~/census_toc/keepers_to_ocr.txt
#   scp fasrc/ocr_never_ocrd_keepers.sbatch \
#       cruano@login.rc.fas.harvard.edu:~/SudokuOCR/fasrc/
#   # then on the cluster:  cd ~/SudokuOCR && sbatch fasrc/ocr_never_ocrd_keepers.sbatch
#
# --array=0-39%7 : 40 chunks (~100 images each); %7 = 7 concurrent so each
# running worker gets its own Harvard Gemini key (you have 7) -> no 429s.
# Add more keys to .env and raise %K to go faster.

set -euo pipefail

cd ~/SudokuOCR
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source ~/.envs/sudokuocr/bin/activate

set -a
source ~/SudokuOCR/.env
set +a

# Gemini-only MoE: unset other keys so the pipeline auto-detects only Gemini
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY

export PYTHONUNBUFFERED=1
export GEMINI_MIN_INTERVAL=2.0
# Budget-safe pass: NO MoE fallback. Each page runs the cheap base path
# (schema + extract + light constraint recheck), ~2-3 calls. Pages that fail
# constraints are flagged (not auto-repaired by the expensive 12-reading MoE).
# Collect those failures afterward and run MoE only on that small remnant.
export SUDOKUOCR_MOE_FALLBACK=0

IMAGE_LIST=${IMAGE_LIST:-$HOME/census_toc/keepers_to_ocr.txt}
RESULTS_DIR=${RESULTS_DIR:-$HOME/census_toc/ocr_results}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}

if [ -n "${HARVARD_GEMINI_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
    N_KEYS=${#KEYS[@]}
    KEY_IDX=$((SLURM_ARRAY_TASK_ID % N_KEYS))
    export HARVARD_GEMINI_API_KEY="${KEYS[$KEY_IDX]}"
    echo "Using API key $((KEY_IDX + 1))/${N_KEYS} for chunk ${SLURM_ARRAY_TASK_ID}"
fi

echo "=== keeper OCR worker ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} ==="
echo "Host: $(hostname)"; echo "Image list: $IMAGE_LIST"; echo "Start: $(date)"

python3 batch_production.py \
    --image-list "$IMAGE_LIST" \
    --results "$RESULTS_DIR" \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --total-chunks "$SLURM_ARRAY_TASK_COUNT" \
    --workers 1 \
    --timeout "$TIMEOUT_SECONDS" \
    --min-interval 2.0 \
    --max-api-calls 6 \
    --max-cost 50

echo "=== Worker ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
