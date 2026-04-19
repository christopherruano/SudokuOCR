#!/bin/bash
# process_census_batch.sh — One-command pipeline for census PDF processing.
#
# Processes all PDFs in ~/census_pdfs through the full ToC pipeline:
#   manifest → merge → range → merge → digitize → save manual review → delete PDFs
#
# Usage:
#   bash process_census_batch.sh                    # auto-labels batch by date
#   bash process_census_batch.sh 1871_1881_1941     # custom batch label
#
# Prerequisites:
#   - PDFs uploaded to ~/census_pdfs/
#   - Virtual env at ~/.envs/sudokuocr
#   - Slurm scripts in ~/SudokuOCR/slurm/

set -euo pipefail

BATCH_LABEL="${1:-batch_$(date +%Y%m%d_%H%M%S)}"

# ---- Paths ----
export PROJECT_DIR="${PROJECT_DIR:-$HOME/SudokuOCR}"
export CENSUS_ROOT="${CENSUS_ROOT:-$HOME/census_pdfs}"
export CENSUS_SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-$HOME/census_toc}"
export VENV_DIR="${VENV_DIR:-$HOME/.envs/sudokuocr}"
export CENSUS_ADJUDICATIONS_DIR="${CENSUS_ADJUDICATIONS_DIR:-$PROJECT_DIR/review_annotations_pdf}"

# Batch-specific task dirs to avoid mixing
export CENSUS_MANIFEST_TASK_DIR="$CENSUS_SCRATCH_ROOT/manifest_tasks_${BATCH_LABEL}"
export CENSUS_RANGE_TASK_DIR="$CENSUS_SCRATCH_ROOT/range_tasks_${BATCH_LABEL}"
export CENSUS_PDF_FILELIST="$CENSUS_SCRATCH_ROOT/inputs/census_pdfs_${BATCH_LABEL}.txt"

# ---- Pipeline settings ----
export TOC_FRONT_PAGES="${TOC_FRONT_PAGES:-80}"
export TOC_FRONT_CONTENT_PAGES="${TOC_FRONT_CONTENT_PAGES:-25}"
export TOC_OCR_FRONT="${TOC_OCR_FRONT:-auto}"
export TOC_BODY_MAX_PAGES="${TOC_BODY_MAX_PAGES:-350}"
export MANIFEST_CONCURRENCY="${MANIFEST_CONCURRENCY:-20}"
export RANGE_CONCURRENCY="${RANGE_CONCURRENCY:-40}"
export DIGITIZE_CONCURRENCY="${DIGITIZE_CONCURRENCY:-40}"

cd "$PROJECT_DIR"
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01
source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "Census Batch Pipeline: $BATCH_LABEL"
echo "============================================================"
echo "Census root:  $CENSUS_ROOT"
echo "Scratch root: $CENSUS_SCRATCH_ROOT"
echo "Started:      $(date)"
echo ""

# ---- Setup ----
mkdir -p "$CENSUS_MANIFEST_TASK_DIR" "$CENSUS_RANGE_TASK_DIR" \
         "$CENSUS_SCRATCH_ROOT"/{inputs,merged,digitized_tables,ocr_cache} \
         "$PROJECT_DIR/slurm/logs"

# ---- Build filelist ----
echo "=== Building PDF filelist ==="
python3 build_pdf_filelist.py "$CENSUS_ROOT" --output "$CENSUS_PDF_FILELIST"
N_PDFS=$(wc -l < "$CENSUS_PDF_FILELIST" | tr -d ' ')
echo "Found $N_PDFS PDFs"
if [ "$N_PDFS" -le 0 ]; then
    echo "No PDFs found. Upload PDFs to $CENSUS_ROOT first."
    exit 1
fi

# ---- Helper: submit and wait ----
submit_and_wait() {
    local JOB_NAME="$1"
    local JOB_ID="$2"
    echo "  Job $JOB_NAME submitted: $JOB_ID"
    echo "  Waiting for completion..."
    # Poll until job disappears from squeue
    while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 15
    done
    # Check for failures
    local FAILED=$(sacct -j "$JOB_ID" --format=State -n 2>/dev/null | grep -c "FAILED" || true)
    if [ "$FAILED" -gt 0 ]; then
        echo "  WARNING: $FAILED tasks failed in $JOB_NAME"
    fi
    echo "  $JOB_NAME complete."
}

# ---- Stage 1: Manifest ----
echo ""
echo "=== Stage 1: ToC Manifest ($N_PDFS PDFs) ==="
MANIFEST_JOB=$(sbatch --parsable \
    --array="0-$((N_PDFS - 1))%${MANIFEST_CONCURRENCY}" \
    slurm/toc_manifest_array.sbatch)
submit_and_wait "manifest" "$MANIFEST_JOB"
echo "  Output: $(ls "$CENSUS_MANIFEST_TASK_DIR"/*.json 2>/dev/null | wc -l) task files"

# ---- Stage 2: Merge manifest ----
echo ""
echo "=== Stage 2: Merge manifest ==="
MERGE1_JOB=$(sbatch --parsable slurm/merge_outputs.sbatch)
submit_and_wait "merge-manifest" "$MERGE1_JOB"

N_DOCS=$(wc -l < "$CENSUS_SCRATCH_ROOT/merged/toc_documents.jsonl" | tr -d ' ')
echo "  Documents: $N_DOCS"

# ---- Stage 3: Range resolution ----
echo ""
echo "=== Stage 3: Range resolution ($N_DOCS documents) ==="
export CENSUS_DOCUMENTS_JSONL="$CENSUS_SCRATCH_ROOT/merged/toc_documents.jsonl"
RANGE_JOB=$(sbatch --parsable \
    --array="0-$((N_DOCS - 1))%${RANGE_CONCURRENCY}" \
    slurm/resolve_ranges_array.sbatch)
submit_and_wait "range" "$RANGE_JOB"
echo "  Output: $(ls "$CENSUS_RANGE_TASK_DIR"/*.json 2>/dev/null | wc -l) task files"

# ---- Stage 4: Merge ranges ----
echo ""
echo "=== Stage 4: Merge ranges ==="
MERGE2_JOB=$(sbatch --parsable slurm/merge_outputs.sbatch)
submit_and_wait "merge-ranges" "$MERGE2_JOB"

N_QUEUE=$(wc -l < "$CENSUS_SCRATCH_ROOT/merged/digitization_queue.jsonl" 2>/dev/null | tr -d ' ')
N_REVIEW=$(wc -l < "$CENSUS_SCRATCH_ROOT/merged/manual_review_queue.jsonl" 2>/dev/null | tr -d ' ')
echo "  Digitization queue: $N_QUEUE"
echo "  Manual review queue: $N_REVIEW"

# ---- Stage 5: Digitize ----
echo ""
echo "=== Stage 5: Digitize ($N_QUEUE page ranges) ==="
export CENSUS_DIGITIZATION_QUEUE="$CENSUS_SCRATCH_ROOT/merged/digitization_queue.jsonl"
DIGI_JOB=$(sbatch --parsable \
    --array="0-$((N_QUEUE - 1))%${DIGITIZE_CONCURRENCY}" \
    slurm/digitize_table_pages_array.sbatch)
submit_and_wait "digitize" "$DIGI_JOB"

N_PNGS=$(find "$CENSUS_SCRATCH_ROOT/digitized_tables" -name "*.png" | wc -l | tr -d ' ')
echo "  Total PNGs: $N_PNGS"

# ---- Save manual review backlog ----
echo ""
echo "=== Saving manual review backlog ==="
{
    echo "=== $BATCH_LABEL ==="
    date
    echo "$N_REVIEW items deferred"
    cat "$CENSUS_SCRATCH_ROOT/merged/manual_review_queue.jsonl"
    echo ""
} >> "$PROJECT_DIR/manual_review_log.txt"
echo "  Appended $N_REVIEW items to manual_review_log.txt"

# ---- Clean up source PDFs ----
echo ""
echo "=== Cleaning up source PDFs ==="
PDF_SIZE=$(du -sh "$CENSUS_ROOT" 2>/dev/null | cut -f1)
echo "  Deleting $PDF_SIZE from $CENSUS_ROOT"
find "$CENSUS_ROOT" -name "*.pdf" -delete
echo "  Done. Disk free: $(df -h /n/home11 2>/dev/null | tail -1 | awk '{print $4}')"

# ---- Summary ----
echo ""
echo "============================================================"
echo "BATCH COMPLETE: $BATCH_LABEL"
echo "============================================================"
echo "  Manifest job:     $MANIFEST_JOB"
echo "  Range job:        $RANGE_JOB"
echo "  Digitize job:     $DIGI_JOB"
echo "  PDFs processed:   $N_PDFS"
echo "  Documents found:  $N_DOCS"
echo "  Pages digitized:  $N_QUEUE ranges → $N_PNGS PNGs"
echo "  Manual review:    $N_REVIEW (saved to manual_review_log.txt)"
echo "  Source PDFs:      DELETED"
echo "  Finished:         $(date)"
echo "============================================================"
echo ""
echo "Next: upload more PDFs to $CENSUS_ROOT and re-run, or start OCR:"
echo "  export CENSUS_DIGITIZATION_TOPICS=\${CENSUS_DIGITIZATION_TOPICS:-age_sex_population,age_sex_civil_condition,age_sex_marital_status}"
echo "  python3 build_digitized_ocr_manifest.py $CENSUS_SCRATCH_ROOT/digitized_tables --strict-age-only --output $CENSUS_SCRATCH_ROOT/merged/ocr_image_list.txt"
echo "  export IMAGE_LIST=$CENSUS_SCRATCH_ROOT/merged/ocr_image_list.txt"
echo "  export IMAGE_ROOT=$CENSUS_SCRATCH_ROOT/digitized_tables"
echo "  export RESULTS_DIR=$CENSUS_SCRATCH_ROOT/ocr_results"
echo "  sbatch --export=ALL --array=0-29%6 fasrc/submit.sh"
