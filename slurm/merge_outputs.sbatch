#!/bin/bash
#SBATCH -J census_merge_outputs
#SBATCH -p serial_requeue
#SBATCH -t 0-01:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o slurm/logs/merge_outputs_%j.out
#SBATCH -e slurm/logs/merge_outputs_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/.envs/sudokuocr}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-$SCRATCH/$USER/census_toc}"

MANIFEST_TASK_DIR="${CENSUS_MANIFEST_TASK_DIR:-$SCRATCH_ROOT/manifest_tasks}"
RANGE_TASK_DIR="${CENSUS_RANGE_TASK_DIR:-$SCRATCH_ROOT/range_tasks}"
MERGED_DIR="${CENSUS_MERGED_DIR:-$SCRATCH_ROOT/merged}"

mkdir -p "$PROJECT_DIR/slurm/logs" "$MERGED_DIR"

module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

python3 merge_toc_manifest_outputs.py "$MANIFEST_TASK_DIR" \
  --output "$MERGED_DIR/toc_manifest_merged.json" \
  --documents-jsonl "$MERGED_DIR/toc_documents.jsonl" \
  --tables-output "$MERGED_DIR/toc_tables.jsonl" \
  --tables-csv "$MERGED_DIR/toc_tables.csv"

if [ -d "$RANGE_TASK_DIR" ] && [ "$(find "$RANGE_TASK_DIR" -name '*.json' | wc -l | tr -d ' ')" -gt 0 ]; then
  MERGE_ARGS=(
    "$RANGE_TASK_DIR"
    --output "$MERGED_DIR/ranges_merged.json"
    --tables-output "$MERGED_DIR/range_tables.jsonl"
    --tables-csv "$MERGED_DIR/range_tables.csv"
    --digitization-queue "$MERGED_DIR/digitization_queue.jsonl"
    --manual-review-queue "$MERGED_DIR/manual_review_queue.jsonl"
  )
  if [ -n "${CENSUS_DIGITIZATION_TOPICS:-}" ]; then
    IFS=',' read -ra TOPICS <<< "$CENSUS_DIGITIZATION_TOPICS"
    MERGE_ARGS+=(--allowed-topics "${TOPICS[@]}")
  fi
  python3 merge_range_outputs.py "${MERGE_ARGS[@]}"
fi
