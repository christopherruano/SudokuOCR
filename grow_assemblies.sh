#!/usr/bin/env bash
# Continuous assembly growth (the user's strategy): grow each Table VII run in the lacking direction,
# OCR'ing the next page from the DRIVE, until it hits a blank / different table / title page — the
# detector's contiguous-VII grouping makes that the natural boundary, so the group is "totally done".
# Each round: detect -> render the next unread edge pages -> OCR -> incorporate -> rebuild assembled
# view + restart review app (so the growth is visible). Stops when no run can extend (all complete),
# or on low disk. Gemini-only, no MoE. Pulls from /Volumes/Expansion.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/grow_assemblies.log
exec >> "$LOG" 2>&1
# fold reads JSON-only (don't copy source images into the corpus) — the crawl only needs the reads, and
# the image copies were filling the disk. The deploy reingest (separate, blocked) re-copies them later.
export SUDOKUOCR_ADAPTER_NOIMG=1
say(){ echo; echo "$(date '+%m-%d %H:%M:%S') ===== $* ====="; }
pgrep -f 'caffeinate -i' >/dev/null || caffeinate -i -s &

restart_app(){ pkill -f census_review_app.py 2>/dev/null; sleep 1; nohup python3 census_review_app.py > fasrc/logs/review_app.log 2>&1 & }

stable=0
for round in $(seq 1 40); do
  free=$(df -m . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 2000 ]; then say "DISK LOW (${free}MB) — pausing growth"; break; fi
  say "round $round (disk ${free}MB): detect runs + refresh master"
  python3 update_page_master.py >/dev/null 2>&1 || true
  # DIRECTIONAL crawler: grow each run only in the needed direction(s) (from your review notes, or the
  # detector's start/end signal), walking outward until a non-VII boundary. Far fewer wasted reads than
  # the old blind both-sides probe, and it respects 'correct'/'end good' verdicts.
  python3 crawl_assemblies.py --reach 6 >/dev/null 2>&1 || true
  N=$(grep -c . export/boundary_grow_list.txt 2>/dev/null); N=${N:-0}
  echo "  next unread edge pages (directional crawl): $N"
  if [ "$N" = "0" ]; then
    stable=$((stable+1))
    say "no run can extend (all groups hit their natural boundary) — stable x$stable"
    python3 build_assembled_view.py >/dev/null 2>&1 || true; restart_app
    [ "$stable" -ge 2 ] && { say "GROWTH COMPLETE"; break; }
    sleep 300; continue
  fi
  stable=0
  say "round $round: FULL-OCR $N edge pages (Gemini, no MoE, MAXCALLS=6 — full read at 300 DPI)"
  # FULL OCR (repair cascade) at full 300-DPI: the page is being read anyway, so capture complete cell
  # data — pages that join a run are then publication-ready without a second re-OCR pass. If a page does
  # NOT read as Table VII, the detector just stops the run there (the natural boundary).
  NKEYS=17 WORKERS=1 MAXCALLS=6 MININT=1.0 LIST=export/boundary_grow_list.txt RESULTS=export/boundary_grow_ocr \
    bash run_flashdrive_ocr.sh >> "$LOG" 2>&1 || true
  say "round $round: incorporate + rebuild assembled view"
  python3 build_money_corpus_adapter.py >/dev/null 2>&1 || true
  python3 build_assembled_view.py >/dev/null 2>&1 || true
  restart_app
  echo "  runs now: $(python3 -c "import json;r=json.load(open('export/tableVII_runs.json'));print(len(r),'runs',sum(x['n_pages'] for x in r),'pages')" 2>/dev/null)"
done
say "grow loop ended (rounds done or disk/stable). Assembled view + review app are current."
