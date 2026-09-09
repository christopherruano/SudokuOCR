#!/usr/bin/env bash
# Auto-switch OCR streams based on the (flaky) drive:
#   drive ON  -> assembly-GROW loop (renders the next edge page from the drive, grows runs to their
#                natural boundary: blank / different table / title page).
#   drive OFF -> original RE-OCR (local scans, drive-independent) — keep making progress meanwhile.
# Hysteresis (2 consecutive same-state checks) avoids thrashing on rapid toggles. Only one OCR stream
# runs at a time (17 keys each; two would risk the jetsam kill). Gemini-only, no MoE.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/auto_switch.log
exec >> "$LOG" 2>&1
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
pgrep -f 'caffeinate -i' >/dev/null || caffeinate -i -s &
drive_on(){ ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1; }
# re-OCR is "done" when the orchestrator has signalled Gemini-plateau (sentinel) OR the (continuously
# rebuilt) list of remaining imperfect tables is empty. The old >=1450 threshold could never be met
# (only ~981 tables are reachable), so it crash-looped restarting an empty list forever.
reocr_done(){ [ -f export/reocr_gemini_done ] && return 0; local c; c=$(grep -c . export/reocr_image_list.txt 2>/dev/null); [ "${c:-1}" -le 0 ]; }
ocr_procs(){ ps -eo args | grep -E 'MacOS/Python batch_production' | grep -v grep | wc -l | tr -d ' '; }
stop_all(){ pkill -f grow_assemblies.sh 2>/dev/null; pkill -f run_flashdrive_ocr.sh 2>/dev/null; pkill -9 -f 'batch_production.py' 2>/dev/null; sleep 3; }
start_grow(){ : > fasrc/logs/grow_assemblies.log; nohup bash grow_assemblies.sh >/dev/null 2>&1 & }
start_reocr(){ NKEYS=17 WORKERS=1 MAXCALLS=8 MININT=1.0 LIST=export/reocr_image_list.txt RESULTS=export/reocr_results \
                nohup bash run_flashdrive_ocr.sh > fasrc/logs/reocr_resume.log 2>&1 & }

mode=""; on=0; off=0
say "auto-switch START"
while true; do
  if drive_on; then on=$((on+1)); off=0; else off=$((off+1)); on=0; fi
  want="$mode"
  if [ -z "$mode" ]; then want=$(drive_on && echo grow || echo reocr)   # first iteration: set immediately
  elif [ "$on" -ge 2 ]; then want="grow"
  elif [ "$off" -ge 2 ]; then want="reocr"; fi

  if [ "$want" != "$mode" ]; then
    say "SWITCH $mode -> $want (drive $(drive_on && echo ON || echo OFF))"
    stop_all
    if [ "$want" = "grow" ]; then start_grow; else reocr_done || start_reocr; fi
    mode="$want"
  else
    # keep-alive: if the active stream's OCR died but there is still work, restart it
    if [ "$mode" = "reocr" ] && ! reocr_done && [ "$(ocr_procs)" = "0" ] && ! pgrep -f run_flashdrive_ocr.sh >/dev/null; then
      say "reocr keep-alive: restarting"; start_reocr
    fi
    if [ "$mode" = "grow" ] && ! pgrep -f grow_assemblies.sh >/dev/null; then
      say "grow keep-alive: restarting"; start_grow
    fi
  fi
  sleep 90
done
