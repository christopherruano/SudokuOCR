#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:?Set CENSUS_SCRATCH_ROOT first}"
CENSUS_ROOT="${CENSUS_ROOT:?Set CENSUS_ROOT first}"
FILELIST="${CENSUS_PDF_FILELIST:-$SCRATCH_ROOT/inputs/census_pdfs.txt}"

mkdir -p "$SCRATCH_ROOT"/{inputs,manifest_tasks,ocr_cache,merged} "$PROJECT_DIR/slurm/logs"

cd "$PROJECT_DIR"

if [ ! -f "$FILELIST" ]; then
  python3 build_pdf_filelist.py "$CENSUS_ROOT" --output "$FILELIST"
fi

N_PDFS=$(wc -l < "$FILELIST" | tr -d ' ')
if [ "$N_PDFS" -le 0 ]; then
  echo "No PDFs found in $FILELIST"
  exit 1
fi

export CENSUS_PDF_FILELIST="$FILELIST"
export TOC_FRONT_PAGES="${TOC_FRONT_PAGES:-80}"
export TOC_FRONT_CONTENT_PAGES="${TOC_FRONT_CONTENT_PAGES:-25}"
export TOC_OCR_FRONT="${TOC_OCR_FRONT:-auto}"
export TOC_BODY_MAX_PAGES="${TOC_BODY_MAX_PAGES:-350}"

MANIFEST_CONCURRENCY="${MANIFEST_CONCURRENCY:-20}"

echo "Submitting manifest stage for $N_PDFS PDFs"
echo "Concurrency cap: $MANIFEST_CONCURRENCY"
echo "TOC_BODY_MAX_PAGES=$TOC_BODY_MAX_PAGES"

sbatch --parsable --array="0-$((N_PDFS - 1))%${MANIFEST_CONCURRENCY}" slurm/toc_manifest_array.sbatch
