#!/usr/bin/env bash
# 8-HOUR AUTONOMOUS RUN — fix broken tables + master the assemblies, fully unattended.
#
#   * Gemini re-OCR (drive-independent) finishes the imperfect tables; when it plateaus, escalate the
#     persistent HARD ones to MISTRAL (Gemini reads backed up). [user-approved: "try mistral if that
#     doesn't work i'll manually take a look"]
#   * Assembly GROWTH (when the drive is on): grow each Table VII run in the lacking direction, OCR'ing
#     the next page, stopping at the title-based boundary (blank / different table / non-VII title).
#   * Periodic INCORPORATE: fold re-OCR fixes, refresh the master (precise classifier) + assembled view.
#   * Disk-aware: prune re-renderable boundary renders before the grow loop's 2GB floor bites.
#   * FINAL: fold everything, refresh master + assembled view + provenance workbook, restart review app,
#     write a STATUS report for the morning.
# Drive-aware OCR<->grow switching is delegated to auto_switch.sh (single fleet; Gemini-only, no MoE).
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/overnight_8h.log
STATUS=fasrc/logs/overnight_status.txt
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%m-%d %H:%M:%S') ===== $* ====="; }
START=$(date +%s)
END_AT=$((8*3600))
INCORP_EVERY=1200          # incorporate every 20 min
pgrep -f 'caffeinate -i' >/dev/null || caffeinate -i -s &
rm -f export/reocr_gemini_done

drive_on(){ ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1; }
reocr_done_n(){ find export/reocr_results -name '*_oneshot.json' 2>/dev/null | wc -l | tr -d ' '; }
runs_stat(){ python3 -c "import json;r=json.load(open('export/tableVII_runs.json'));print(len(r),'runs /',sum(x['n_pages'] for x in r),'pages')" 2>/dev/null; }
list_rem(){ local c; c=$(grep -c . export/reocr_image_list.txt 2>/dev/null); echo "${c:-0}"; }

write_status(){   # one-glance state for a morning check-in
  local el=$(( $(date +%s) - START ))
  { echo "OVERNIGHT 8h — updated $(date '+%m-%d %H:%M:%S')  (+$((el/60)) min elapsed)"
    echo "  drive:        $(drive_on && echo ON\ \(growing\ assemblies\) || echo off\ \(re-OCR\))"
    echo "  disk free:    $(df -h / | awk 'NR==2{print $4}')"
    echo "  re-OCR:       $(reocr_done_n)/981 done | $(list_rem) remaining | mistral_done=${MIST}"
    echo "  assemblies:   $(runs_stat)"
    echo "  boundary OCR: $(find export/boundary_grow_ocr -name '*_oneshot.json' 2>/dev/null | wc -l | tr -d ' ') edge pages read"
    echo "  OCR workers:  $(ps -eo args | grep batch_production | grep -v grep | wc -l | tr -d ' ')"
  } > "$STATUS"
}

incorporate(){
  say "incorporate (fold re-OCR fixes, refresh master + assembled view)"
  python3 rebuild_reocr_list.py >/dev/null 2>&1 || true
  python3 merge_reocr.py >/dev/null 2>&1 || true
  python3 update_page_master.py >/dev/null 2>&1 || true
  python3 build_assembled_view.py >/dev/null 2>&1 || true
  echo "  re-OCR $(reocr_done_n)/981 ($(list_rem) left) | assemblies $(runs_stat)"
}

mistral_escalation(){
  say "MISTRAL escalation on persistent HARD tables (Gemini reads -> *_oneshot.gemini.json backup)"
  set -a; source .env 2>/dev/null; set +a
  unset OPENAI_API_KEY ANTHROPIC_API_KEY
  python3 run_mistral_ocr.py 2>&1 | tail -4
  python3 emit_mistral_oneshot.py 2>&1 | tail -4
  python3 update_page_master.py >/dev/null 2>&1 || true
  python3 build_assembled_view.py >/dev/null 2>&1 || true
  MIST=1
}

disk_guard(){    # keep headroom: boundary renders are re-renderable from the drive, safe to prune.
  local free; free=$(df -m . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 3000 ]; then
    say "disk ${free}MB low — pruning re-renderable boundary renders older than 20min (surely OCR'd)"
    # only >20min old: never deletes a render the grow loop just made but hasn't OCR'd yet
    find export/boundary_grow_renders -name '*.png' -type f -mmin +20 2>/dev/null | head -800 | xargs rm -f 2>/dev/null || true
  fi
}

# ---- launch the drive-aware OCR<->grow switcher (single fleet) ----
pkill -f auto_switch.sh 2>/dev/null; pkill -f grow_assemblies.sh 2>/dev/null; pkill -9 -f batch_production.py 2>/dev/null; sleep 2
nohup bash auto_switch.sh >/dev/null 2>&1 &
# PREV_DONE starts high so a re-OCR set that's ALREADY drained registers as plateaued immediately
# (otherwise the first incorporate sees done>0 as a false "advance" and delays Mistral an extra cycle).
MIST=0; PREV_DONE=999999; PLATEAU=0; LAST_INCORP=0
say "OVERNIGHT 8h START — switcher up (drive $(drive_on && echo ON || echo off)), re-OCR $(reocr_done_n)/981"
write_status

while true; do
  now=$(date +%s); el=$((now-START))
  [ "$el" -ge "$END_AT" ] && { say "8h elapsed -> final rebuild"; break; }
  disk_guard
  if [ $((now-LAST_INCORP)) -ge "$INCORP_EVERY" ]; then
    incorporate; LAST_INCORP=$now
    done=$(reocr_done_n); rem=$(list_rem)
    # Plateau = Gemini re-OCR no longer advancing (stuck on unreadable stragglers) OR list fully drained.
    # Only counts while the drive is OFF: re-OCR is the active stream then, and we must NEVER start
    # Mistral during a grow round (2nd concurrent OCR fleet = jetsam risk on this Mac).
    if drive_on; then PLATEAU=0
    elif [ "$done" -le "$PREV_DONE" ] || [ "$rem" -le 0 ]; then PLATEAU=$((PLATEAU+1)); else PLATEAU=0; fi
    PREV_DONE=$done
    if [ "$MIST" = "0" ] && [ "$PLATEAU" -ge 2 ] && ! drive_on; then
      touch export/reocr_gemini_done    # stop the switcher restarting a drained re-OCR list
      mistral_escalation
    fi
  fi
  write_status
  sleep 120
done

# ---------- FINAL REBUILD ----------
say "FINAL: stop OCR, fold everything, refresh master + assembled view + provenance workbook"
pkill -f auto_switch.sh 2>/dev/null; pkill -f grow_assemblies.sh 2>/dev/null; pkill -f run_flashdrive_ocr.sh 2>/dev/null; pkill -9 -f batch_production.py 2>/dev/null; sleep 3
python3 rebuild_reocr_list.py >/dev/null 2>&1 || true
python3 merge_reocr.py 2>&1 | tail -2
[ "$MIST" = "1" ] || mistral_escalation
python3 update_page_master.py 2>&1 | tail -4
python3 build_assembled_view.py >/dev/null 2>&1 || true
# refresh the human-facing provenance workbook from the live CSV master (best-effort)
python3 refresh_provenance_workbook.py 2>&1 | tail -2 || true
# bring the review app back up for the morning
pkill -f census_review_app.py 2>/dev/null; sleep 1
nohup python3 census_review_app.py > fasrc/logs/review_app.log 2>&1 &
MIST=$MIST write_status
say "OVERNIGHT DONE — re-OCR $(reocr_done_n)/981, assemblies $(runs_stat), review app up on :5058"
