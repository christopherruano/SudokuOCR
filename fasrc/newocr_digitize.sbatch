#!/usr/bin/env bash
#SBATCH --job-name=sudokuocr_newocr
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-715%14
#SBATCH --output=fasrc/logs/slurm_newocr_%A_%a.out
#SBATCH --error=fasrc/logs/slurm_newocr_%A_%a.err

# JOB B: re-render the page windows of AGE multi-page tables with missing interior
# pages (digitize_table_pages_task renders EVERY page in the resolved range, so the
# skipped pages get captured). One array task per queue line. Then the rendered
# crops feed the normal extract->OCR path (batch_production over the output dir).

set -euo pipefail
cd "$HOME/census_toc" 2>/dev/null || cd "$SLURM_SUBMIT_DIR"
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01

QUEUE=${QUEUE:-$HOME/census_toc/newocr_queue.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/census_toc/newocr_pages}

python3 digitize_table_pages_task.py \
    --queue "$QUEUE" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing
