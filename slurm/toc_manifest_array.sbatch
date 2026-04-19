#!/bin/bash
#SBATCH -J census_toc_manifest
#SBATCH -p serial_requeue
#SBATCH -t 0-06:00
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --array=0-99%25
#SBATCH -o slurm/logs/toc_manifest_%A_%a.out
#SBATCH -e slurm/logs/toc_manifest_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/.envs/sudokuocr}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-$SCRATCH/$USER/census_toc}"

FILELIST="${CENSUS_PDF_FILELIST:-$SCRATCH_ROOT/inputs/census_pdfs.txt}"
TASK_DIR="${CENSUS_MANIFEST_TASK_DIR:-$SCRATCH_ROOT/manifest_tasks}"
OCR_CACHE_DIR="${CENSUS_OCR_CACHE_DIR:-$SCRATCH_ROOT/ocr_cache}"
ADJUDICATIONS_DIR="${CENSUS_ADJUDICATIONS_DIR:-$PROJECT_DIR/review_annotations_pdf}"

mkdir -p "$PROJECT_DIR/slurm/logs" "$TASK_DIR" "$OCR_CACHE_DIR"

module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

python3 run_toc_manifest_task.py \
  --filelist "$FILELIST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$TASK_DIR" \
  --front-pages "${TOC_FRONT_PAGES:-80}" \
  --front-content-pages "${TOC_FRONT_CONTENT_PAGES:-25}" \
  --ocr-front "${TOC_OCR_FRONT:-auto}" \
  --ocr-cache-dir "$OCR_CACHE_DIR" \
  --adjudications-dir "$ADJUDICATIONS_DIR" \
  --body-max-pages "${TOC_BODY_MAX_PAGES:-0}" \
  --skip-existing
