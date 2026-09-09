#!/usr/bin/env bash
# Autonomous re-OCR loop: re-OCR every page that fails a constraint across the broken Table VII tables
# with Gemini (Harvard keys, free), merge improvements (merge_reocr only overlays when BETTER and never
# touches human_corrected), re-check, repeat. Two rounds, then stop and report what still fails (those
# are the dense ones to consider for Mistral). Run with /Volumes/Expansion mounted.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_loop.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }

ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1 || { say "DRIVE NOT MOUNTED — abort"; exit 1; }
say "LOOP START"

for round in 1 2; do
  say "ROUND $round — build failing-page list (all broken tables)"
  python3 build_reocr_constraint_list.py --top 999 || { say "list build failed"; break; }
  N=$(grep -c . export/reocr_constraint_list.txt 2>/dev/null || echo 0)
  say "ROUND $round — $N failing pages queued"
  [ "${N:-0}" -lt 1 ] && { say "nothing failing — converged"; break; }

  say "ROUND $round — Gemini re-OCR across 12 keys"
  LIST=export/reocr_constraint_list.txt RESULTS=export/reocr_constraint_ocr NKEYS=12 WORKERS=1 MAXCALLS=14 \
    bash run_flashdrive_ocr.sh || say "ocr round returned nonzero (continuing)"

  say "ROUND $round — merge improved reads"
  python3 merge_reocr.py 2>&1 | tail -3

  say "ROUND $round — re-check (rebuild broken queue)"
  python3 build_broken_queue.py 2>&1 | head -1
done

say "MISTRAL FALLBACK — apply cached Mistral reads to the hard tables (free; gated by safe-improvement +"
say "  U+M+W agreement, never overlays a human_corrected table, keeps Gemini as *_oneshot.gemini.json)"
python3 emit_mistral_oneshot.py 2>&1 | tail -3
say "re-check after Mistral overlay"
python3 build_broken_queue.py 2>&1 | head -1

say "LOOP DONE — rebuilding assembled view with corrected reads"
python3 build_assembled_view.py 2>&1 | tail -1
say "ALL DONE — see this log + build_broken_queue output for before/after failure counts"
