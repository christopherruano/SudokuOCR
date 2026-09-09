#!/usr/bin/env bash
# Prepare the deployable static-site upload: waits for the OCR jobs to drain, merges the re-OCR
# improvements + new gap reads into the corpus, finalizes page numbers + boundaries, then does a
# disk-safe NO_ASSEMBLY rebuild and verifies the file count. Leaves census_site ready to deploy.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/prepare_upload.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production|python3 batch_production' | grep -v grep | wc -l | tr -d ' '; }
fail(){ say "FAILED: $*"; kill "${CAFF:-0}" 2>/dev/null; echo "UPLOAD_STATUS=FAILED:$*"; exit 1; }

say "waiting for OCR jobs (re-OCR + boundary-gap) to drain"
while [ "$(realprocs)" != "0" ]; do sleep 60; done
sleep 30; [ "$(realprocs)" != "0" ] && { while [ "$(realprocs)" != "0" ]; do sleep 60; done; }
say "OCR drained: re-OCR=$(find export/reocr_results -name '*_oneshot.json'|wc -l|tr -d ' ') boundary=$(find export/boundary_gap_ocr -name '*_oneshot.json'|wc -l|tr -d ' ')"

caffeinate -i -s & CAFF=$!
# disk-safety: keeper_inference -> Expansion drive, skip the unused 12G plain facts
if [ -d /Volumes/Expansion ]; then
  rm -rf export/keeper_inference 2>/dev/null
  mkdir -p /Volumes/Expansion/census_keeper_inference
  ln -sfn /Volumes/Expansion/census_keeper_inference export/keeper_inference
fi
export SUDOKUOCR_SKIP_PLAIN_FACTS=1 NO_ASSEMBLY=1

say "1/7 merge re-OCR improvements into the corpus"
python3 merge_reocr.py || fail merge_reocr
say "2/7 incorporate new gap reads (boundary + final + flashdrive)"
python3 build_money_corpus_adapter.py || fail adapter
say "3/7 update the page-number master (authoritative)"
python3 update_page_master.py || true
say "4/7 finalize Table VII run boundaries (cont-marker)"
python3 build_tableVII_runs.py || true
say "5/7 NO_ASSEMBLY reingest"
bash reingest.sh || fail reingest
say "6/7 render deployable allowlist + static site + highlights + coverage"
python3 build_render_allowlist.py 11500 || true
rm -rf census_site/tables census_site/data; mkdir -p census_site/tables census_site/data
python3 build_static_site.py || fail static_site
python3 build_highlights.py || true
python3 build_coverage_nav.py || true
say "7/7 verify file count"
kill "$CAFF" 2>/dev/null
N=$(find census_site -type f | wc -l | tr -d ' ')
PAGES=$(ls census_site/tables/*.html 2>/dev/null | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $PAGES pages"
if [ "$N" -lt 20000 ]; then echo "UPLOAD_STATUS=READY files=$N pages=$PAGES"
else echo "UPLOAD_STATUS=OVER_LIMIT files=$N (lower the allowlist cap)"; fi
echo "DEPLOY: npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true"
