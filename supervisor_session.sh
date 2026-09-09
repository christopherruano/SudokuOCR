#!/bin/bash
# Session supervisor: keeps the confirmed-range OCR watcher alive (relaunches when it exits stable)
# and periodically builds assembly runs for freshly confirmed docs. Runs ~9 hours then stops.
cd /Users/chrisruano/Research/SudokuOCR
for tick in $(seq 1 110); do
  if ! pgrep -f "ocr_confirmed_watch.py" >/dev/null 2>&1; then
    echo "[$(date +%H:%M)] watcher not running — relaunching"
    nohup python3 ocr_confirmed_watch.py >> "export/logs/confirmed_watch_supervised.log" 2>&1 &
  fi
  # every 4th tick (~20 min): build runs for newly confirmed docs
  if [ $((tick % 4)) -eq 0 ]; then
    python3 build_runs_from_confirmed.py 2>/dev/null | grep -vE "indexing|images indexed" | tail -1
  fi
  sleep 300
done
echo "[$(date +%H:%M)] supervisor session ended"
