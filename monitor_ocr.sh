#!/usr/bin/env bash
# Memory + progress monitor for the broad OCR run. File-based (argv = "bash monitor_ocr.sh")
# and matches the REAL python process pattern, so it NEVER matches `pgrep -f batch_production.py`
# itself (the bug that blocked the waiter). Exits when OCR drains or swap goes low.
cd "$(dirname "$0")"
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production|python3 batch_production' | grep -v grep | wc -l | tr -d ' '; }
for i in $(seq 1 240); do
  sw=$(sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/')
  mf=$(memory_pressure 2>/dev/null | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p')
  np=$(realprocs)
  nr=$(find export/flashdrive_ocr -name '*_oneshot.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date '+%H:%M:%S')] ocr_procs=$np results=$nr memfree=${mf}% swapfree=${sw}M"
  # swap is dynamic on this Mac (total can read 0); the real ceiling is memory_pressure free %.
  if [ -n "$mf" ] && [ "$mf" -lt 12 ]; then echo "!! MEM LOW (${mf}% free) — kill some keys to avoid jetsam"; break; fi
  if [ "$np" = "0" ]; then echo ">> BROAD OCR DRAINED ($nr results)"; break; fi
  sleep 120
done
