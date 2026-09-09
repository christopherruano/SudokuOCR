#!/usr/bin/env bash
# Robust finisher: drive the gap OCR to TRUE completion (auto-restart if sleep/jetsam kills workers),
# THEN rebuild census_site (assembly-free) and emit the deploy command.
# Rebuild triggers ONLY when the remaining-gap count hits 0 — a lid-close drain can't ship incomplete.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/complete_and_build.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production|python3 batch_production' | grep -v grep | wc -l | tr -d ' '; }
remaining(){ python3 - <<'PY'
import os,glob,re
have=set()
for f in glob.glob("export/flashdrive_ocr/**/*_oneshot.json",recursive=True):
    m=re.search(r"(\d+)_p(\d+)",os.path.basename(f))
    if m: have.add(f"{m.group(1)}_p{m.group(2)}")
lst=[l.strip() for l in open("export/flashdrive_render_list.txt") if l.strip()]
print(len([p for p in lst if os.path.basename(p)[:-4] not in have]))
PY
}
launch_ocr(){
  python3 - <<'PY'
import os,glob,re
have=set()
for f in glob.glob("export/flashdrive_ocr/**/*_oneshot.json",recursive=True):
    m=re.search(r"(\d+)_p(\d+)",os.path.basename(f))
    if m: have.add(f"{m.group(1)}_p{m.group(2)}")
lst=[l.strip() for l in open("export/flashdrive_render_list.txt") if l.strip()]
open("export/flashdrive_remaining.txt","w").write("\n".join(p for p in lst if os.path.basename(p)[:-4] not in have)+"\n")
PY
  NKEYS=17 WORKERS=1 MAXCALLS=6 MININT=1.0 LIST=export/flashdrive_remaining.txt RESULTS=export/flashdrive_ocr \
    bash run_flashdrive_ocr.sh > fasrc/logs/ocr_complete.log 2>&1 &
}
fail(){ say "FAILED: $*"; kill "${CAFF:-0}" 2>/dev/null; echo "BUILD_STATUS=FAILED:$*"; exit 1; }

caffeinate -i -s & CAFF=$!
say "caffeinate up (pid $CAFF). NOTE: -i can't stop a LID-CLOSE sleep — keep the lid open to finish fast."

# PHASE 1: OCR to true completion, auto-restarting on sleep/jetsam death
say "PHASE 1: completing OCR ($(remaining) gap pages remain)"
stall=0
while [ "$(remaining)" -gt 0 ]; do
  if [ "$(realprocs)" = "0" ]; then
    say "no OCR running but $(remaining) remain -> (re)launching 17 keys"
    launch_ocr
    sleep 120
  fi
  sleep 60
done
say "OCR COMPLETE (0 gap pages remain, $(find export/flashdrive_ocr -name '*_oneshot.json'|wc -l|tr -d ' ') reads); settling"
while [ "$(realprocs)" != "0" ]; do sleep 30; done

# PHASE 2: assembly-free rebuild
say "PHASE 2: rebuild"
say "1/5 incorporate new reads"; python3 build_money_corpus_adapter.py || fail adapter
say "2/5 NO_ASSEMBLY reingest";  NO_ASSEMBLY=1 bash reingest.sh || fail reingest
say "3/5 prune orphans";         python3 prune_orphan_pages.py --apply || true
say "4/5 render site";           python3 build_static_site.py || fail static_site
say "5/5 highlights";            python3 build_highlights.py || true
kill "$CAFF" 2>/dev/null
N=$(find census_site -type f | wc -l | tr -d ' '); PAGES=$(ls census_site/tables/*.html 2>/dev/null | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $PAGES pages"
if [ "$N" -lt 20000 ]; then echo "BUILD_STATUS=READY files=$N pages=$PAGES"
else echo "BUILD_STATUS=OVER_LIMIT files=$N"; fi
echo "DEPLOY: npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true"
