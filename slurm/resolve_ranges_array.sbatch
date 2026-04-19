#!/bin/bash
#SBATCH -J census_resolve_ranges
#SBATCH -p serial_requeue
#SBATCH -t 0-02:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --array=0-99%50
#SBATCH -o slurm/logs/resolve_ranges_%A_%a.out
#SBATCH -e slurm/logs/resolve_ranges_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/.envs/sudokuocr}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-$SCRATCH/$USER/census_toc}"

DOCUMENTS_JSONL="${CENSUS_DOCUMENTS_JSONL:-$SCRATCH_ROOT/merged/toc_documents.jsonl}"
TASK_DIR="${CENSUS_RANGE_TASK_DIR:-$SCRATCH_ROOT/range_tasks}"

mkdir -p "$PROJECT_DIR/slurm/logs" "$TASK_DIR"

module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

python3 resolve_table_ranges_task.py \
  --documents-jsonl "$DOCUMENTS_JSONL" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$TASK_DIR" \
  --skip-existing
