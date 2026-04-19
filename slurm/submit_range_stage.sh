#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:?Set CENSUS_SCRATCH_ROOT first}"
DOCUMENTS_JSONL="${CENSUS_DOCUMENTS_JSONL:-$SCRATCH_ROOT/merged/toc_documents.jsonl}"

if [ ! -f "$DOCUMENTS_JSONL" ]; then
  echo "Missing documents JSONL: $DOCUMENTS_JSONL"
  echo "Run merge manifest stage first."
  exit 1
fi

N_DOCS=$(wc -l < "$DOCUMENTS_JSONL" | tr -d ' ')
if [ "$N_DOCS" -le 0 ]; then
  echo "No documents found in $DOCUMENTS_JSONL"
  exit 1
fi

export CENSUS_DOCUMENTS_JSONL="$DOCUMENTS_JSONL"
RANGE_CONCURRENCY="${RANGE_CONCURRENCY:-40}"

cd "$PROJECT_DIR"
echo "Submitting range stage for $N_DOCS documents"
echo "Concurrency cap: $RANGE_CONCURRENCY"
sbatch --parsable --array="0-$((N_DOCS - 1))%${RANGE_CONCURRENCY}" slurm/resolve_ranges_array.sbatch
