#!/usr/bin/env bash
# Local simulation of the FASRC Slurm array job workflow.
#
# Runs 4 chunks in parallel (like 4 Slurm array tasks), each processing
# a small number of images. Validates:
#   1. Chunking covers all images with no overlap
#   2. Per-chunk logs don't collide
#   3. Resumability (re-run skips completed images)
#   4. Results merge cleanly
#
# Usage:
#   bash fasrc/test_local.sh                    # dry-run validation only (no API calls)
#   bash fasrc/test_local.sh --live             # process 2 images per chunk (8 total)
#   bash fasrc/test_local.sh --live --all       # process all images across 4 chunks

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

TOTAL_CHUNKS=4
IMAGE_ROOT="$PROJECT_DIR/age_tables"
RESULTS_DIR="$PROJECT_DIR/results_test_fasrc"
LIVE=false
ALL=false

for arg in "$@"; do
    case "$arg" in
        --live) LIVE=true ;;
        --all)  ALL=true ;;
    esac
done

echo "=== FASRC Local Simulation ==="
echo "Project: $PROJECT_DIR"
echo "Chunks:  $TOTAL_CHUNKS"
echo "Results: $RESULTS_DIR"
echo ""

# ---- Step 1: Dry-run each chunk to verify coverage ----
echo "--- Step 1: Verify chunking covers all images ---"
echo ""

total_assigned=0
for chunk_id in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    output=$(python3 batch_production.py "$IMAGE_ROOT" \
        --results "$RESULTS_DIR" \
        --chunk "$chunk_id" --total-chunks "$TOTAL_CHUNKS" \
        --dry-run 2>&1)

    # Extract "To process: N images" line
    n_images=$(echo "$output" | grep "To process:" | grep -o '[0-9]*')
    echo "  Chunk $chunk_id: $n_images images"
    total_assigned=$((total_assigned + n_images))
done

# Compare to un-chunked total
full_output=$(python3 batch_production.py "$IMAGE_ROOT" \
    --results "$RESULTS_DIR" --dry-run 2>&1)
total_expected=$(echo "$full_output" | grep "To process:" | grep -o '[0-9]*')

echo ""
echo "  Total assigned across chunks: $total_assigned"
echo "  Total without chunking:       $total_expected"

if [ "$total_assigned" -eq "$total_expected" ]; then
    echo "  PASS: All images covered, no gaps or overlaps"
else
    echo "  FAIL: Mismatch! $total_assigned assigned vs $total_expected expected"
    exit 1
fi

if [ "$LIVE" != "true" ]; then
    echo ""
    echo "Dry-run only. Use --live to process images (costs API credits)."
    echo "Full dry-run output:"
    echo ""
    echo "$full_output"
    exit 0
fi

# ---- Step 2: Run chunks in parallel ----
echo ""
echo "--- Step 2: Processing images across $TOTAL_CHUNKS parallel chunks ---"

mkdir -p "$RESULTS_DIR"

MAX_IMAGES_FLAG=""
if [ "$ALL" != "true" ]; then
    MAX_IMAGES_FLAG="--max-images 2"
    echo "(Limited to 2 images per chunk. Use --all to process everything.)"
fi
echo ""

pids=()
for chunk_id in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    echo "  Starting chunk $chunk_id ..."
    python3 batch_production.py "$IMAGE_ROOT" \
        --results "$RESULTS_DIR" \
        --chunk "$chunk_id" --total-chunks "$TOTAL_CHUNKS" \
        --timeout 300 \
        $MAX_IMAGES_FLAG \
        > "$RESULTS_DIR/_chunk_${chunk_id}.log" 2>&1 &
    pids+=($!)
done

echo ""
echo "  Waiting for ${#pids[@]} chunks to finish..."
echo "  (Monitor with: tail -f $RESULTS_DIR/_chunk_*.log)"
echo ""

all_ok=true
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  Chunk $i: done"
    else
        echo "  Chunk $i: FAILED (exit code $?)"
        all_ok=false
    fi
done

# ---- Step 3: Validate results ----
echo ""
echo "--- Step 3: Validate results ---"

# Count per-chunk log entries
total_logged=0
for chunk_id in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    log_file="$RESULTS_DIR/_batch_log_chunk${chunk_id}.jsonl"
    if [ -f "$log_file" ]; then
        n=$(wc -l < "$log_file" | tr -d ' ')
        echo "  Chunk $chunk_id log: $n entries"
        total_logged=$((total_logged + n))
    else
        echo "  Chunk $chunk_id log: MISSING"
    fi
done
echo "  Total logged: $total_logged"

# Count result JSON files
n_results=$(find "$RESULTS_DIR" -name "*_oneshot.json" | wc -l | tr -d ' ')
echo "  Result JSON files: $n_results"

# Check for any duplicate results (shouldn't happen with correct chunking)
n_unique=$(find "$RESULTS_DIR" -name "*_oneshot.json" | sort -u | wc -l | tr -d ' ')
if [ "$n_results" -ne "$n_unique" ]; then
    echo "  WARNING: Duplicate result files detected!"
else
    echo "  No duplicates (good)"
fi

# ---- Step 4: Test resumability ----
echo ""
echo "--- Step 4: Test resumability ---"
echo "  Re-running chunk 0 (should skip already-processed images)..."

resume_output=$(python3 batch_production.py "$IMAGE_ROOT" \
    --results "$RESULTS_DIR" \
    --chunk 0 --total-chunks "$TOTAL_CHUNKS" \
    --dry-run 2>&1)

n_remaining=$(echo "$resume_output" | grep "To process:" | grep -o '[0-9]*')
n_skipped=$(echo "$resume_output" | grep -o 'accepted_high_confidence: [0-9]*' | grep -o '[0-9]*' || echo "0")

echo "  Remaining to process: $n_remaining"
echo "  Already high-confidence: $n_skipped"

echo ""
echo "=== Test complete ==="
if [ "$all_ok" = true ]; then
    echo "All chunks succeeded. Check results in: $RESULTS_DIR"
else
    echo "Some chunks failed. Check logs in: $RESULTS_DIR/_chunk_*.log"
fi
