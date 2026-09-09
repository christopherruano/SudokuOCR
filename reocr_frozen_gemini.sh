#!/usr/bin/env bash
# Retry EVERY 'frozen column' misfire page with the full Gemini OCR protocol (render 300dpi from drive ->
# schema-discovery + extraction + constraint repair via batch_production), then a frozen-aware keep-best
# merge (only adopts a read that REMOVES frozen columns or fixes constraints) + duplicate dedupe.
# Pages STILL frozen after Gemini are written to export/frozen_still_bad.txt for the Mistral fallback.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_frozen.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%F %T') ===== $* ====="; }

ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1 || { say "DRIVE NOT MOUNTED — abort"; exit 1; }

say "baseline frozen-column tables"
python3 detect_misfires.py | head -1

say "flag all frozen-column pages (fresh)"
: > export/reocr_page_flags.txt
python3 detect_misfires.py --flag | tail -1
N=$(grep -c . export/reocr_page_flags.txt 2>/dev/null || echo 0)
say "$N pages flagged"

say "render full pages from the drive (300 dpi)"
python3 build_reocr_page_list.py --dpi 300 | tail -1

say "Gemini protocol re-OCR (17 Harvard keys, 2 workers for memory safety)"
LIST=export/reocr_page_list.txt RESULTS=export/reocr_constraint_ocr NKEYS=17 WORKERS=2 MAXCALLS=10 \
  bash run_flashdrive_ocr.sh || say "ocr returned nonzero (continuing to merge what landed)"

say "frozen-aware keep-best merge"
python3 merge_reocr.py 2>&1 | tail -3
say "propagate best read across duplicate copies"
python3 dedupe_corpus_best.py --apply 2>&1 | tail -2

say "archive flags"
mv export/reocr_page_flags.txt export/reocr_page_flags.txt.frozen.done 2>/dev/null || true

say "re-detect frozen + rebuild queue + assembled view"
python3 detect_misfires.py | head -1
python3 build_broken_queue.py 2>&1 | head -1
python3 build_assembled_view.py 2>&1 | tail -1

say "remaining frozen -> Mistral fallback list"
python3 -c "import csv; rows=list(csv.DictReader(open('export/misfire_tables.csv'))); open('export/frozen_still_bad.txt','w').write('\n'.join(r['imghash'] for r in rows)); print(len(rows),'still frozen -> export/frozen_still_bad.txt')"
say "DONE"
