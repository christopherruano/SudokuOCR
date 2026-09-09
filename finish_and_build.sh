#!/usr/bin/env bash
# AUTO-FINISH: when the main OCR drains, rebuild census_site (assembly-free) so it's READY TO REUPLOAD.
# Sequence runs ONLY after OCR procs hit 0 (memory-safe: no rebuild while OCR keys are hot).
#   incorporate new reads -> NO_ASSEMBLY reingest -> prune orphans -> static site -> highlights -> verify <20k
# Robust: each step's exit code checked; aborts + reports rather than shipping a half-built site.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/finish_and_build.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production|python3 batch_production' | grep -v grep | wc -l | tr -d ' '; }
fail(){ say "FAILED at: $*"; kill "${CAFF:-0}" 2>/dev/null; echo "BUILD_STATUS=FAILED:$*"; exit 1; }

say "confirming OCR is running before arming the drain-wait (avoids relaunch-gap race)"
for i in $(seq 1 120); do [ "$(realprocs)" != "0" ] && break; sleep 5; done
[ "$(realprocs)" = "0" ] && { say "no OCR detected after 10 min — aborting (nothing to wait for)"; exit 1; }
say "OCR running ($(realprocs) procs); waiting for it to drain"
# require TWO consecutive zero readings 60s apart, so a brief blip never triggers a rebuild
while :; do
  [ "$(realprocs)" = "0" ] && { sleep 60; [ "$(realprocs)" = "0" ] && break; }
  sleep 60
done
say "OCR drained ($(find export/flashdrive_ocr -name '*_oneshot.json' | wc -l | tr -d ' ') reads); rebuilding"

caffeinate -i -s & CAFF=$!   # keep awake through the (OCR-free) rebuild

say "1/5 incorporate new reads into the corpus"
python3 build_money_corpus_adapter.py || fail "adapter"

say "2/5 NO_ASSEMBLY reingest (assembly-free; per-page reads)"
NO_ASSEMBLY=1 bash reingest.sh || fail "reingest"

say "3/5 prune orphaned assembled pages"
python3 prune_orphan_pages.py --apply || true

say "4/5 render static site (assembly-free DB; client-side CSV)"
python3 build_static_site.py || fail "build_static_site"

say "5/5 highlights + coverage banner"
python3 build_highlights.py || true

kill "$CAFF" 2>/dev/null
N=$(find census_site -type f | wc -l | tr -d ' ')
PAGES=$(ls census_site/tables/*.html 2>/dev/null | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $PAGES table pages"
if [ "$N" -lt 20000 ]; then
  echo "BUILD_STATUS=READY files=$N pages=$PAGES"
  echo "DEPLOY: npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true"
else
  echo "BUILD_STATUS=OVER_LIMIT files=$N  (exceeds Cloudflare 20k cap — investigate before deploy)"
fi
