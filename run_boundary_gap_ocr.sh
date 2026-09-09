#!/usr/bin/env bash
# OCR the boundary-gap pages on a chosen key range (default keys 14-16) so it doesn't contend
# with the main re-OCR on keys 0-13. Bash (read -ra works here, unlike zsh).
set -uo pipefail
cd "$(dirname "$0")"
set -a; source .env 2>/dev/null; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY
export SUDOKUOCR_TABLEVII_HINT=1 PYTHONUNBUFFERED=1
IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
LIST="${LIST:-export/boundary_gap_list.txt}"
RESULTS="${RESULTS:-export/boundary_gap_ocr}"
START_KEY="${START_KEY:-14}"
NK="${NK:-3}"
mkdir -p "$RESULTS" fasrc/logs
pids=()
for c in $(seq 0 $((NK-1))); do
  ki=$((START_KEY + c))
  HARVARD_GEMINI_API_KEY="${KEYS[$ki]}" \
  python3 batch_production.py --image-list "$LIST" --results "$RESULTS" \
    --chunk "$c" --total-chunks "$NK" --workers 1 --timeout 1800 --min-interval 1.0 \
    --max-api-calls 8 --max-cost 30 > "fasrc/logs/bgap_w${c}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} boundary-gap workers on keys ${START_KEY}..$((START_KEY+NK-1)): ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done
echo "boundary-gap OCR done: $(find "$RESULTS" -name '*_oneshot.json' | wc -l) results"
