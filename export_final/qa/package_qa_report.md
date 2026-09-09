# Package QA Report

Status: `pass`
Issues: `0`
Warnings: `3`

## Scope
- Source package audited: `export/final_project_exports`
- Final view audited: `export_final`

## Preview Audit
- index rows: `40`
- page previews: `13`
- assembled previews: `27`

## File Counts
- `.csv`: `365`
- `.json`: `74`
- `.jsonl`: `1`
- `.md`: `15`
- `.txt`: `132`
- `.xlsx`: `3`

## Warnings
- Full raw OCR page JSONs are not bundled in the final release; source_json fields are references, usually to cluster paths.
- Preview matrices are convenience views. For analysis, trust cell_facts_rich.csv and source_catalog.csv first.
- Some assembled section names are OCR-noisy; section CSV content and source_catalog_row.json remain the authoritative shipped artifacts.

## Issues
- none
