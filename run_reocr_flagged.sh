#!/usr/bin/env bash
# Re-OCR exactly the pages flagged in the review app ('✗ re-OCR just this page'), merge only the reads
# that strictly reduce constraint failures (never touching human_corrected), clear the flags, re-check.
# Run with /Volumes/Expansion mounted.
set -uo pipefail
cd "$(dirname "$0")"
LOG=fasrc/logs/reocr_flagged.log
exec >> "$LOG" 2>&1
say(){ echo; echo "$(date '+%H:%M:%S') ===== $* ====="; }

ls -d /Volumes/Expansion/census_pdfs >/dev/null 2>&1 || say "DRIVE NOT MOUNTED — using cached scans (lower-res; reconnect for best quality)"

say "build render list from flagged pages"
python3 build_reocr_page_list.py
N=$(grep -c . export/reocr_page_list.txt 2>/dev/null || echo 0)
say "$N flagged page(s) to re-OCR"
[ "${N:-0}" -lt 1 ] && { say "nothing flagged — done"; exit 0; }

say "Gemini re-OCR (high-res renders, all 17 keys)"
LIST=export/reocr_page_list.txt RESULTS=export/reocr_constraint_ocr NKEYS=17 WORKERS=2 MAXCALLS=14 \
  bash run_flashdrive_ocr.sh || say "ocr returned nonzero (continuing to merge what landed)"

say "merge improvements (only if they reduce failures; human_corrected never overwritten)"
python3 merge_reocr.py 2>&1 | tail -3

say "propagate best existing read across duplicate copies of the same page (same-table only)"
python3 dedupe_corpus_best.py --apply 2>&1 | tail -2

say "clear needs_reocr on processed pages + archive the flag file"
python3 - <<'PY'
import os, glob, json
flags = "export/reocr_page_flags.txt"
ihs = [x for x in (open(flags).read().split() if os.path.exists(flags) else []) if x.strip()]
for ih in set(ihs):
    for jf in glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.json"):
        try:
            d = json.load(open(jf)); m = d.get("metadata", {}) or {}; m.pop("needs_reocr", None)
            d["metadata"] = m; json.dump(d, open(jf, "w"))
        except Exception:
            pass
open("export/reocr_last_batch.txt", "w").write("\n".join(ihs))   # for queue_completed_runs.py
if os.path.exists(flags):
    os.replace(flags, flags + ".done")
print(f"cleared {len(set(ihs))} flags, archived -> reocr_page_flags.txt.done")
PY

say "re-check broken queue + rebuild assembled view"
python3 build_broken_queue.py 2>&1 | head -1
python3 build_assembled_view.py 2>&1 | tail -1

say "pop re-OCR'd runs to the BACK of the queue (review the full run) + record completed"
python3 queue_completed_runs.py 2>&1 | tail -2
say "DONE"
