#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRATCH_ROOT="${CENSUS_SCRATCH_ROOT:?Set CENSUS_SCRATCH_ROOT first}"
QUEUE_PATH="${CENSUS_DIGITIZATION_QUEUE:-$SCRATCH_ROOT/merged/digitization_queue.jsonl}"

if [ ! -f "$QUEUE_PATH" ]; then
  echo "Missing digitization queue: $QUEUE_PATH"
  echo "Run range merge stage first."
  exit 1
fi

N_QUEUE=$(wc -l < "$QUEUE_PATH" | tr -d ' ')
if [ "$N_QUEUE" -le 0 ]; then
  echo "No queue rows found in $QUEUE_PATH"
  exit 1
fi

export CENSUS_DIGITIZATION_QUEUE="$QUEUE_PATH"
DIGITIZE_CONCURRENCY="${DIGITIZE_CONCURRENCY:-40}"

cd "$PROJECT_DIR"
echo "Submitting digitization stage for $N_QUEUE queue rows"
echo "Concurrency cap: $DIGITIZE_CONCURRENCY"
sbatch --parsable --array="0-$((N_QUEUE - 1))%${DIGITIZE_CONCURRENCY}" slurm/digitize_table_pages_array.sbatch
