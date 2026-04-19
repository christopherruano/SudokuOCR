#!/usr/bin/env bash
#SBATCH --job-name=sudokuocr
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19%4
#SBATCH --output=fasrc/logs/slurm_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# SudokuOCR batch processing via Slurm array jobs.
#
# Each array task processes 1/N of the images. The pipeline is API-bound
# (Gemini/OpenAI calls), not compute-bound, so we use minimal resources
# per task and parallelize via array size.
#
# Usage:
#   # First time: dry run to see how many images
#   python3 batch_production.py age_tables/ --dry-run
#
#   # Then submit (adjust --array=0-N based on desired parallelism):
#   sbatch fasrc/submit.sh
#
#   # Monitor:
#   sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
#   tail -f fasrc/logs/slurm_<jobid>_*.out
#
# Tuning:
#   --array=0-19   -> 20 parallel workers (default)
#   --array=0-49   -> 50 parallel workers (faster, higher API load)
#   --array=0-99   -> 100 workers (check your API rate limits first)
#   --time=08:00:00 -> 8 hours per worker; increase for huge chunks
#
# To limit concurrent workers (e.g., stay under API rate limits):
#   #SBATCH --array=0-49%10    # 50 tasks, max 10 running at once

set -euo pipefail

# ---- Configuration ----
# Adjust these paths for your FASRC setup:
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_DIR/age_tables}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"
IMAGE_LIST="${IMAGE_LIST:-}"
VENV_DIR="$HOME/.envs/sudokuocr"

TOTAL_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-20}
CHUNK_ID=${SLURM_ARRAY_TASK_ID:-0}

# ---- Environment ----
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01

source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1

# Load API keys from .env (dotenv will also handle this, but belt-and-suspenders)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Create log directory
mkdir -p "$PROJECT_DIR/fasrc/logs"

# ---- Multi-key rotation ----
# If HARVARD_GEMINI_API_KEYS has comma-separated keys, pick one per chunk.
# This spreads load across keys so each key sees ~2 concurrent connections.
if [ -n "$HARVARD_GEMINI_API_KEYS" ]; then
    IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
    N_KEYS=${#KEYS[@]}
    KEY_IDX=$((CHUNK_ID % N_KEYS))
    export HARVARD_GEMINI_API_KEY="${KEYS[$KEY_IDX]}"
    echo "Using API key $((KEY_IDX + 1))/${N_KEYS} for chunk ${CHUNK_ID}"
fi

# ---- Rate limiting ----
export GEMINI_MIN_INTERVAL="${GEMINI_MIN_INTERVAL:-2.0}"

# ---- Run ----
echo "=== SudokuOCR Worker ${CHUNK_ID}/${TOTAL_CHUNKS} ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Image root: $IMAGE_ROOT"
if [ -n "$IMAGE_LIST" ]; then
    echo "Image list: $IMAGE_LIST"
fi
echo "Results: $RESULTS_DIR"
echo "Rate limit: ${GEMINI_MIN_INTERVAL}s between API calls"
echo ""

cd "$PROJECT_DIR"

# Per-chunk cost cap: total budget / number of chunks
# Override with MAX_COST_PER_CHUNK env var, or set to 0 for no limit.
MAX_COST="${MAX_COST_PER_CHUNK:-50}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
MAX_API_CALLS_PER_IMAGE="${MAX_API_CALLS_PER_IMAGE:-0}"

CMD=(python3 batch_production.py)
if [ -n "$IMAGE_LIST" ]; then
    CMD+=(--image-list "$IMAGE_LIST")
else
    CMD+=("$IMAGE_ROOT")
fi
CMD+=(
    --results "$RESULTS_DIR"
    --chunk "$CHUNK_ID"
    --total-chunks "$TOTAL_CHUNKS"
    --workers 1
    --timeout "$TIMEOUT_SECONDS"
    --min-interval 2.0
    --max-cost "$MAX_COST"
)
if [ "$MAX_API_CALLS_PER_IMAGE" -gt 0 ]; then
    CMD+=(--max-api-calls "$MAX_API_CALLS_PER_IMAGE")
fi

"${CMD[@]}"

echo ""
echo "=== Worker ${CHUNK_ID} finished at $(date) ==="
