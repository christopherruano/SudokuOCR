# Cluster run-book — finish Neil's goal tables (multi-page AGE tables)

Goal: complete the Imperial Table VII/VIII/IX age tables (general population age
structure, age-by-religion, main castes). Two jobs.

## PREREQ — push the prompt fix first (CRITICAL)
The biggest failure mode in the last run was **dropped Persons columns** and
**real numbers read as zeros**. The fixed prompts must be on the cluster before re-OCR:
```
rsync -av oneshot.py schema_discovery.py batch_production.py \
  cruano@login.rc.fas.harvard.edu:/n/home11/cruano/census_toc/
```

## JOB A — RE-OCR 619 flagged AGE tables  (reocr_neil_images.txt)
These are age tables we already have images for, flagged bad (94 Opus-confirmed
misreads, 113 unverified, 412 constraint-fail). Re-run OCR with the fixed prompt.
```
scp export_final/sendout_final/sendout_v1/reocr_neil_images.txt \
  cruano@login.rc.fas.harvard.edu:/n/home11/cruano/census_toc/reocr_neil_images.txt
# on the cluster:
IMAGE_LIST=$HOME/census_toc/reocr_neil_images.txt sbatch fasrc/rerun_reocr.sbatch
```

## JOB B — NEW-OCR missing AGE pages  (newocr_neil_pages.csv)
2,688 interior pages of multi-page age tables that were never extracted (the
table-locator skipped them). For each source PDF, RE-RUN table extraction over
`pdf_page_span` — the gaps are interior to that span, so re-detection captures
the skipped pages. The pipeline dedups existing crops, so only new pages get OCR'd.
- Input: `newocr_neil_pages.csv` (dochash, source_pdf, pdf_page_span, tables, missing_printed_pages)
- 98/139 PDFs have a resolved page span; the other 41 need their PDF filename
  recovered first (see open item below).
```
scp export_final/sendout_final/sendout_v1/newocr_neil_pages.csv \
  cruano@login.rc.fas.harvard.edu:/n/home11/cruano/census_toc/
# re-extract per PDF over pdf_page_span using the digitization/table-detection
# entry point, then OCR the new crops (same batch_production path).
```

## After the runs
1. `rsync` results back into `export/ocr_results_referenced/digitized_tables/`
2. Re-extract cells, re-merge keepers, rebuild explorer (merge_keepers.py ->
   detect_keeper_badreads.py -> build_explorer_db.py); re-stitch to confirm gaps closed.

## Open item
41 of the 139 new-OCR docs lack a source-PDF filename (we hold their pages,
not the label). Recover via the digitization manifest join before Job B on those.
