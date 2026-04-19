#!/bin/bash
#SBATCH -J census_digitize_tables
#SBATCH -p serial_requeue
#SBATCH -t 0-03:00
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH --array=0-99%50
#SBATCH -o slurm/logs/digitize_tables_%A_%a.out
#SBATCH -e slurm/logs/digitize_tables_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/.envs/sudokuocr}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-$SCRATCH/$USER/census_toc}"

QUEUE_PATH="${CENSUS_DIGITIZATION_QUEUE:-$SCRATCH_ROOT/merged/digitization_queue.jsonl}"
OUTPUT_DIR="${CENSUS_DIGITIZED_DIR:-$SCRATCH_ROOT/digitized_tables}"

mkdir -p "$PROJECT_DIR/slurm/logs" "$OUTPUT_DIR"

module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

python3 digitize_table_pages_task.py \
  --queue "$QUEUE_PATH" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR" \
  --skip-existing
