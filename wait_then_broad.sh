#!/usr/bin/env bash
# Wait for pass-1 (202 priority pages) to drain, then launch the BROAD OCR over the
# full flash-drive render list (2,179 pages). Resume skips the ~202 already done.
# All 17 Gemini keys, WORKERS=1 (proven-stable memory footprint), MAXCALLS=6 (lighter repair).
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/wait_then_broad.log
echo "$(date '+%H:%M:%S') waiting for pass-1 to drain..." >> "$LOG"
while pgrep -f batch_production.py >/dev/null 2>&1; do
  sleep 20
done
echo "$(date '+%H:%M:%S') pass-1 drained; launching broad OCR" >> "$LOG"
NKEYS=17 WORKERS=1 MAXCALLS=6 MININT=1.0 \
  LIST=export/flashdrive_render_list.txt RESULTS=export/flashdrive_ocr \
  bash run_flashdrive_ocr.sh >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') broad OCR run returned" >> "$LOG"
