#!/usr/bin/env bash
# Local Gemini-only re-OCR of imperfect Table VII scans, 17 keys in parallel.
# Writes fresh oneshot results IN PLACE into export/ocr_results_referenced so
# reingest.sh picks them up. GEMINI ONLY — OpenAI/Anthropic unset (cost).
set -uo pipefail
cd "$(dirname "$0")"
set -a; source .env 2>/dev/null; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY            # Gemini only (Mistral kept, for optional MoE)
export SUDOKUOCR_TABLEVII_HINT=1                         # prime wide Table VII (4 col groups, scan far-right)
export SUDOKUOCR_MOE_FALLBACK="${MOE:-0}"                # MOE=1 to enable 3-Gemini(+Mistral) fallback on failures
# SCHEMA=1 -> inject sibling-derived row labels + column schema (forces the full grid
# using known-good labels from earlier pages of the same multi-page group)
if [ "${SCHEMA:-0}" = "1" ]; then
  export SUDOKUOCR_SCHEMA_TEMPLATES="$(pwd)/export_final/sendout_final/sendout_v1/group_schema_templates.json"
fi
export PYTHONUNBUFFERED=1

LIST="${LIST:-export_final/sendout_final/sendout_v1/reocr_imperfect_local.txt}"
RESULTS="${RESULTS:-export/ocr_results_referenced}"
WORKERS="${WORKERS:-6}"          # concurrent images PER key — we're I/O-bound on slow gateway calls
MININT="${MININT:-0.8}"          # seconds between calls per worker (rate-limit headroom is large)
IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
N=${#KEYS[@]}
mkdir -p fasrc/logs
echo "re-OCR $(grep -c . "$LIST") scans across $N keys x $WORKERS workers = $((N*WORKERS)) concurrent (MoE=$SUDOKUOCR_MOE_FALLBACK) -> $RESULTS"

pids=()
for i in $(seq 0 $((N-1))); do
  HARVARD_GEMINI_API_KEY="${KEYS[$i]}" \
  python3 batch_production.py --image-list "$LIST" --results "$RESULTS" \
      --chunk "$i" --total-chunks "$N" \
      --workers "$WORKERS" --timeout 1800 --min-interval "$MININT" --max-api-calls 12 --max-cost 60 \
      > "fasrc/logs/reocr_w${i}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} workers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "all workers done (failed processes: $fail)"
# quick tally
acc=$(grep -h '"decision"' fasrc/logs/reocr_w*.log 2>/dev/null | grep -c accepted || true)
echo "accepted decisions logged: $acc"
