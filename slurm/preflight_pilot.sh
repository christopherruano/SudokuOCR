#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:-}"
CENSUS_ROOT="${CENSUS_ROOT:-}"
FILELIST="${CENSUS_PDF_FILELIST:-${SCRATCH_ROOT:+$SCRATCH_ROOT/inputs/census_pdfs.txt}}"

echo "=== TOC Pipeline Preflight ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CENSUS_ROOT=${CENSUS_ROOT:-<unset>}"
echo "CENSUS_SCRATCH_ROOT=${SCRATCH_ROOT:-<unset>}"
echo "CENSUS_PDF_FILELIST=${FILELIST:-<unset>}"
echo "TOC_FRONT_PAGES=${TOC_FRONT_PAGES:-80}"
echo "TOC_FRONT_CONTENT_PAGES=${TOC_FRONT_CONTENT_PAGES:-25}"
echo "TOC_OCR_FRONT=${TOC_OCR_FRONT:-auto}"
echo "TOC_BODY_MAX_PAGES=${TOC_BODY_MAX_PAGES:-350}"
echo "MANIFEST_CONCURRENCY=${MANIFEST_CONCURRENCY:-20}"
echo "RANGE_CONCURRENCY=${RANGE_CONCURRENCY:-40}"
echo "DIGITIZE_CONCURRENCY=${DIGITIZE_CONCURRENCY:-40}"
echo

if [ -z "${CENSUS_ROOT}" ] || [ -z "${SCRATCH_ROOT}" ]; then
  echo "Set CENSUS_ROOT and CENSUS_SCRATCH_ROOT before running."
  exit 1
fi

if [ -f "${FILELIST}" ]; then
  N_PDFS=$(wc -l < "${FILELIST}" | tr -d ' ')
  echo "PDF filelist exists: ${N_PDFS} rows"
else
  echo "PDF filelist does not exist yet: ${FILELIST}"
fi

if [ -f "${SCRATCH_ROOT}/merged/toc_documents.jsonl" ]; then
  N_DOCS=$(wc -l < "${SCRATCH_ROOT}/merged/toc_documents.jsonl" | tr -d ' ')
  echo "Merged documents: ${N_DOCS}"
fi

if [ -f "${SCRATCH_ROOT}/merged/digitization_queue.jsonl" ]; then
  N_QUEUE=$(wc -l < "${SCRATCH_ROOT}/merged/digitization_queue.jsonl" | tr -d ' ')
  echo "Digitization queue rows: ${N_QUEUE}"
fi
