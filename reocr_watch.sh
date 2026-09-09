#!/usr/bin/env bash
# CONTINUOUS auto re-OCR daemon. Always running: whenever you flag a page ('✗ re-OCR just this page'),
# it picks it up within ~20-40s, re-OCRs from the drive (Gemini), merges any read that reduces
# constraint failures, and runs the frozen-aware merge + dedupe — so by the time you navigate back to a
# page, it's already re-OCR'd. Writes live state to export/reocr_watch_status.json and per-batch
# before/after to export/reocr_batch_result.md. Runs until killed.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_watch.log
STATUS=export/reocr_watch_status.json
SETTLE=${SETTLE:-20}
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
status(){ printf '{"state":"%s","flagged":%s,"last":"%s"}\n' "$1" "${2:-0}" "$(date '+%H:%M:%S')" > "$STATUS"; }
log "===== continuous watcher started (settle ${SETTLE}s) ====="
trap 'log "stopped"; status stopped 0; exit 0' INT TERM

flagcount(){ if [ -s export/reocr_page_flags.txt ]; then tr ' ' '\n' < export/reocr_page_flags.txt | grep -c . ; else echo 0; fi; }

while true; do
  N=$(flagcount)
  if [ "${N:-0}" -ge 1 ]; then
    status settling "$N"; sleep "$SETTLE"; N=$(flagcount)
    if ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1; then
      status processing "$N"; log "re-OCR $N flagged page(s)"
      python3 reocr_snapshot.py before > /tmp/reocr_before.json 2>>"$LOG" || echo '{"pages":[]}' > /tmp/reocr_before.json
      bash run_reocr_flagged.sh
      python3 reocr_snapshot.py after /tmp/reocr_before.json > export/reocr_batch_result.md 2>>"$LOG" || true
      log "batch done"
      status idle 0
    else
      status drive-down "$N"; log "drive down; holding $N flag(s)"; sleep 30; continue
    fi
  else
    status idle 0
  fi
  sleep 12
done
