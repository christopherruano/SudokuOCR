#!/usr/bin/env bash
# Memory-safe Gemini OCR for the flash-drive renders: a SUBSET of keys, 1 worker each,
# so we don't spike RAM on a swap-constrained machine. NKEYS / WORKERS overridable.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source .env 2>/dev/null; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY
export SUDOKUOCR_TABLEVII_HINT=1
export PYTHONUNBUFFERED=1

LIST="${LIST:-export/flashdrive_render_list.txt}"
RESULTS="${RESULTS:-export/flashdrive_ocr}"
NKEYS="${NKEYS:-8}"          # how many keys (=processes) to run concurrently — memory lever
WORKERS="${WORKERS:-1}"      # threads per process — keep at 1 for low RAM
MININT="${MININT:-1.0}"
MAXCALLS="${MAXCALLS:-12}"   # per-image API-call cap — lower = faster/lighter (less repair cascade)
mkdir -p fasrc/logs "$RESULTS"
IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
TOTAL=${#KEYS[@]}
N=$(( NKEYS < TOTAL ? NKEYS : TOTAL ))
echo "OCR $(grep -c . "$LIST" 2>/dev/null) pages across $N keys x $WORKERS worker (memory-safe) -> $RESULTS"

pids=()
for i in $(seq 0 $((N-1))); do
  HARVARD_GEMINI_API_KEY="${KEYS[$i]}" \
  python3 batch_production.py --image-list "$LIST" --results "$RESULTS" \
      --chunk "$i" --total-chunks "$N" \
      --workers "$WORKERS" --timeout 1800 --min-interval "$MININT" --max-api-calls "$MAXCALLS" --max-cost 60 \
      > "fasrc/logs/fd_w${i}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} OCR workers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "all workers done (failed processes: $fail)"
echo "results: $(find "$RESULTS" -name '*_oneshot.json' 2>/dev/null | wc -l) artifacts"
