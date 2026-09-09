#!/usr/bin/env bash
# Wait for the running Gemini re-OCR batch to finish, then merge ONLY the reads that actually reduce
# constraint failures (merge_reocr's gate; never touches human_corrected), re-check, rebuild the queue +
# assembled view, and log before/after. Robust: polls for the batch, times out after ~75 min.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_finish.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }

say "BEFORE — current broken-table count"
python3 build_broken_queue.py 2>&1 | head -1

say "waiting for Gemini batch_production to finish (poll 30s, cap ~75min)"
for i in $(seq 1 150); do
  pgrep -f "batch_production.py --image-list export/reocr_constraint_list.txt" >/dev/null 2>&1 || { say "batch finished"; break; }
  sleep 30
done

say "MERGE — fold improved Gemini reads (only when strictly better)"
python3 merge_reocr.py 2>&1 | tail -4

say "AFTER — re-check broken-table count + worst remaining"
python3 build_broken_queue.py 2>&1 | head -6

say "rebuild assembled view with corrected reads"
python3 build_assembled_view.py 2>&1 | tail -1
say "DONE"
