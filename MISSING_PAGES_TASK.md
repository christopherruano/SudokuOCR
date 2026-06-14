# Fill the missing Table VII pages (the real path to COMPLETE tables)

## What this is
We only digitized ~18% of each source volume's pages, so multi-page Table VII have
**378 genuinely-missing pages across 83 volumes** (371 fillable — they have a source
PDF + a reliable printed→PDF offset, learned from the page-ordering pass). These are
NOT bad reads — they're pages we never OCR'd. Filling them is how an assembled Table
VII goes from "internally consistent for the rows we have" to **complete**.

This is *new* OCR (pages we never touched), not re-OCR. Run it on the cluster, where
the PDFs live (`/n/home11/cruano/census_pdfs/...`).

## Inputs (already generated)
- `export_final/sendout_final/sendout_v1/missing_pages.jsonl` — 131 page-RANGE blocks,
  one per gappy Table VII block, in the format `newocr_digitize.sbatch` already expects
  (`file`, `resolved_start_page`, `resolved_end_page`, year, province, dochash). The
  sbatch renders EVERY page in the range, so the missing interior pages get captured.
- `missing_page_ocr_queue.csv` — human-readable: per volume, the missing printed pages,
  the mapped PDF pages, and offset confidence.

## Run on the cluster
```bash
# 1) stage the queue
cp missing_pages.jsonl ~/census_toc/newocr_queue.jsonl
N=$(wc -l < ~/census_toc/newocr_queue.jsonl)

# 2) render the page ranges (captures the missing interior pages) — JOB B
sbatch --array=0-$((N-1)) fasrc/newocr_digitize.sbatch     # -> ~/census_toc/newocr_pages/

# 3) OCR the rendered pages with the Table VII prime (Gemini only)
export SUDOKUOCR_TABLEVII_HINT=1; unset OPENAI_API_KEY ANTHROPIC_API_KEY
sbatch --array=0-3 fasrc/rerun_reocr.sbatch \
    --export=ALL,IMAGE_LIST=<(find ~/census_toc/newocr_pages -name '*.png'),RESULTS_DIR=~/census_toc/ocr_results
# (or point batch_production.py --root ~/census_toc/newocr_pages --results ~/census_toc/ocr_results)

# 4) rsync the new results back into export/ocr_results_referenced/, then locally:
bash reingest.sh                 # extracts the new pages, re-assembles the now-complete tables
python3 build_assembled_tables.py  # (reingest already does this) — gaps fill, page order holds
```

## After it lands
The assembled Table VII gain their missing pages → re-run `assembly_integrity_audit.py`
(the 71 page-gap suspects should drop sharply) and `validate_tableVII.py`. Tables that
were "perfect but incomplete" become complete. Then rebuild the static site.

## Note on the 4 no-offset / 71 stuck volumes
4 volumes have a source PDF but no reliable offset (need 1 anchor page read);
~71 documents have no resolved source_pdf at all (dochash→PDF map lives only on the
cluster — see STUCK_VOLUMES_RECOVERY.md). Those can't be filled until their PDF path
is recovered cluster-side.
