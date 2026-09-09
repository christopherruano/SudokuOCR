#!/usr/bin/env bash
# Keep OCR working on exactly what needs it: re-OCR the tables you mark 'needs re-OCR' in the review app
# + the thin/empty extraction failures. Gemini-only, full quality. Folds improved reads back into the
# corpus (the review app reads it live, so they refresh on the next page). Picks up NEW flags each round;
# never re-OCRs a human-CORRECTED table. Loops, idle-waiting when nothing is flagged.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_flagged.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%m-%d %H:%M:%S') ===== $* ====="; }
pgrep -f 'caffeinate -i' >/dev/null || caffeinate -i -s &
for round in $(seq 1 100000); do
  python3 build_reocr_flagged_list.py
  N=$(grep -c . export/reocr_flagged_list.txt 2>/dev/null); N=${N:-0}
  if [ "$N" = "0" ]; then say "nothing newly flagged — idle"; sleep 150; continue; fi
  say "round $round: re-OCR $N flagged/thin tables (Gemini, full quality)"
  NKEYS=10 WORKERS=1 MAXCALLS=8 MININT=1.0 LIST=export/reocr_flagged_list.txt RESULTS=export/reocr_results \
    bash run_flashdrive_ocr.sh >> "$LOG" 2>&1 || true
  # mark these imghashes done so we don't re-OCR them again (unless re-flagged), then fold improvements
  grep -oE '[0-9a-f]{40}' export/reocr_flagged_list.txt >> export/reocr_flagged_done.txt
  say "round $round: fold improved reads into the corpus"
  python3 merge_reocr.py 2>&1 | tail -1
  sleep 30
done
