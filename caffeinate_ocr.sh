#!/usr/bin/env bash
# Keep the Mac awake (no idle/system sleep) until the OCR run drains, then release automatically.
# Self-stops when 0 OCR procs remain, or after a 5h safety cap — never holds the machine awake forever.
cd "$(dirname "$0")"
LOG=fasrc/logs/caffeinate_ocr.log
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production|python3 batch_production' | grep -v grep | wc -l | tr -d ' '; }
caffeinate -i -s &
CAFF=$!
echo "$(date '+%H:%M:%S') caffeinate started (pid $CAFF) — no idle/system sleep until OCR drains" >> "$LOG"
for i in $(seq 1 300); do            # 300 x 60s = 5h safety cap
  if [ "$(realprocs)" = "0" ]; then
    echo "$(date '+%H:%M:%S') OCR drained after ~${i} min; releasing caffeinate" >> "$LOG"
    break
  fi
  sleep 60
done
kill "$CAFF" 2>/dev/null
echo "$(date '+%H:%M:%S') caffeinate released" >> "$LOG"
