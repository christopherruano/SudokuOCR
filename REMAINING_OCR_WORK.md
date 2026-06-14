# Remaining OCR work to complete the corpus (cluster hand-off)

Tied to `master_tracking.xlsx` — every image's status is in there. This is the full
list of what's left to OCR/render so the corpus is complete. Big jobs → cluster
(the PDFs and crops are there at `/n/home11/cruano/census_toc/`).

## The four batches (in priority order)

| Queue file | What | Count | How to run |
|---|---|---|---|
| `cluster_ocr_neverocrd_VII.txt` | **Table VII** crops rendered but never OCR'd | ~63* | batch_production over the list, `SUDOKUOCR_TABLEVII_HINT=1` |
| `missing_pages.jsonl` | gap pages NEVER rendered (53 volumes) — render from PDF | 378 pages | `newocr_digitize.sbatch` (renders range) → then OCR the output dir |
| `cluster_ocr_failed_reocr.txt` | OCR'd but failed/low-confidence — re-OCR | ~1,345 | batch_production, `TABLEVII_HINT=1` (+ `MoE` for hard ones) |
| `cluster_ocr_neverocrd_all.txt` | all never-OCR'd crops (incl. non-Table-VII) | ~2,848 | batch_production over the list |

\* The 63 are what remained NOT-OCR'd in the tracker snapshot; ~255 Table VII crops
are being OCR'd locally right now, so this number drops to near zero after that lands.

## Run pattern (Gemini only, as always)
```bash
export SUDOKUOCR_TABLEVII_HINT=1; unset OPENAI_API_KEY ANTHROPIC_API_KEY
# image-list batches:
cp cluster_ocr_<batch>.txt ~/census_toc/todo.txt
sbatch --array=0-N fasrc/rerun_reocr.sbatch   # (IMAGE_LIST=~/census_toc/todo.txt)
# render-missing batch:
cp missing_pages.jsonl ~/census_toc/newocr_queue.jsonl
sbatch --array=0-130 fasrc/newocr_digitize.sbatch
```

## After results land
rsync results into `export/ocr_results_referenced/`, then locally:
```bash
bash reingest.sh                # extracts new pages, re-assembles, re-validates
python3 build_missing_page_queue.py    # recompute what (if anything) is still missing
python3 build_master_tracker.py        # refresh the master sheet
```
Each new read also fills its page number in the tracker. Repeat until the master
sheet shows no ⏳ NOT-OCR'd and no remaining gaps.

## Blocked (need cluster-side recovery first)
~71 volumes have NO resolved source PDF (dochash→PDF map lives only on the cluster).
Their gaps can't be rendered until that path is recovered — see STUCK_VOLUMES_RECOVERY.md.
