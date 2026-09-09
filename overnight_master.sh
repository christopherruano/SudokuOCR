#!/usr/bin/env bash
# OVERNIGHT: master the assemblies + read the pages the boundaries imply + rebuild a better site.
#   1. boundary-grow loop: detect runs -> render the unread pages just before/after each run from the
#      DRIVE -> OCR them -> incorporate -> re-detect (runs grow toward their true title-page start +
#      district extent). Repeat until no new edge pages.
#   2. fold in the banked re-OCR + empty-VII improvements.
#   3. disk-safe NO_ASSEMBLY reingest (keeper_inference -> /Volumes/Expansion, skip plain facts).
#   4. UPDATE THE MASTER DOC: rebuild the provenance workbook (page_provenance_workbook.xlsx).
#   5. rebuild the static site (pages + highlights + coverage + assembled view).
# Emits UPLOAD_STATUS=READY + the deploy command. Gemini-only, no MoE. Pulls from the drive.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/overnight_master.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%m-%d %H:%M:%S') ===== $* ====="; }
realprocs(){ ps -eo args | grep -E 'MacOS/Python batch_production' | grep -v grep | wc -l | tr -d ' '; }

caffeinate -i -s & CAFF=$!
say "START overnight master (caffeinate $CAFF; -i can't stop a LID-CLOSE sleep)"

ocr_list(){  # $1=list $2=results  -> 17 keys, no MoE, wait for completion
  NKEYS=17 WORKERS=1 MAXCALLS=6 MININT=1.0 LIST="$1" RESULTS="$2" bash run_flashdrive_ocr.sh
}

# ---- 1. boundary-grow loop ----
for round in 1 2 3; do
  say "boundary-grow round $round: refresh master + detect runs"
  python3 update_page_master.py || true            # rebuilds master (page_number_status) + runs
  say "round $round: find + render unread edge pages from the drive"
  python3 find_boundary_pages.py --reach 4 || true
  N=$(grep -c . export/boundary_grow_list.txt 2>/dev/null || echo 0)
  echo "  edge pages to read this round: $N"
  [ "$N" = "0" ] && { say "no new edge pages -> boundaries stable"; break; }
  say "round $round: OCR $N edge pages (Gemini, no MoE)"
  ocr_list export/boundary_grow_list.txt export/boundary_grow_ocr || true
  say "round $round: incorporate the new reads"
  python3 build_money_corpus_adapter.py || true
done

# ---- 2. fold in the banked re-OCR + empty-VII improvements ----
say "fold in banked re-OCR + empty-VII improvements"
python3 merge_reocr.py || true
python3 build_money_corpus_adapter.py || true
python3 update_page_master.py || true

# ---- 3. disk-safe reingest ----
say "disk-safe NO_ASSEMBLY reingest"
if [ -d /Volumes/Expansion ]; then
  rm -rf export/keeper_inference 2>/dev/null
  mkdir -p /Volumes/Expansion/census_keeper_inference
  ln -sfn /Volumes/Expansion/census_keeper_inference export/keeper_inference
fi
export SUDOKUOCR_SKIP_PLAIN_FACTS=1 NO_ASSEMBLY=1
bash reingest.sh || { echo "UPLOAD_STATUS=FAILED:reingest"; kill "$CAFF" 2>/dev/null; exit 1; }

# ---- 4. update THE MASTER DOC (provenance workbook) ----
say "update the master doc: provenance workbook"
python3 build_page_provenance_workbook.py --output-dir export_final/page_provenance || true

# ---- 5. rebuild the site (pages + highlights + coverage + assembled view) ----
say "rebuild static site + assembled view"
python3 build_tableVII_runs.py || true
python3 build_render_allowlist.py 11000 || true
rm -rf census_site/tables census_site/data; mkdir -p census_site/tables census_site/data
python3 build_static_site.py || { echo "UPLOAD_STATUS=FAILED:static_site"; kill "$CAFF" 2>/dev/null; exit 1; }
python3 build_highlights.py || true
python3 build_coverage_nav.py || true
python3 build_assembled_view.py || true

kill "$CAFF" 2>/dev/null
N=$(find census_site -type f | wc -l | tr -d ' ')
P=$(ls census_site/tables/*.html 2>/dev/null | wc -l | tr -d ' ')
A=$(ls census_site/assembled/*.html 2>/dev/null | wc -l | tr -d ' ')
say "DONE — census_site: $N files, $P pages, $A assembled tables"
if [ "$N" -lt 20000 ]; then echo "UPLOAD_STATUS=READY files=$N pages=$P assembled=$A"
else echo "UPLOAD_STATUS=OVER_LIMIT files=$N (lower the allowlist cap)"; fi
echo "DEPLOY: npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true"
