# Cluster re-OCR — the 136 genuine targets (364 images)

These are the ONLY Table VII tables that genuinely need re-OCR (single scans that
truly dropped a column / have digit errors). The other imperfect tables were fixed
locally (half-page merge + continuation stitch) and must NOT be re-OCR'd.

Do NOT re-run the old 1,500-image set — it overlapped only 107 of these and didn't fix them.

## Files to copy to the cluster
- `tableVII_genuine_reocr.txt` — 364 image paths (already in cluster-path form
  `/n/home11/cruano/census_toc/digitized_tables/...png`)
- `tableVII_genuine_schema.json` — optional per-image expected columns (254 have a
  sibling template)

## Run
The key is the **Table VII prime** — it tells schema-discovery to expect 4 column
groups (Total/Unmarried/Married/Widowed × M/F) and scan to the far-right edge, which
is exactly the dropped-Widowed/Females failure these tables have.

```bash
# on the cluster, in ~/SudokuOCR
cp /path/to/tableVII_genuine_reocr.txt ~/census_toc/reocr_images.txt
export SUDOKUOCR_TABLEVII_HINT=1                      # primes the wide-table prompt
# optional, for the 254 with a known schema (forces the exact full grid):
# export SUDOKUOCR_SCHEMA_TEMPLATES=~/census_toc/group_schema_templates.json
sbatch --array=0-3 fasrc/rerun_reocr.sbatch          # IMAGE_LIST defaults to reocr_images.txt
```

(`SUDOKUOCR_TABLEVII_HINT` is read by oneshot.py:154; the sbatch already targets
U/M/W sub-columns. `SUDOKUOCR_SCHEMA_TEMPLATES` is read by oneshot.py:122 and forces
the full column grid for crops whose imghash is in the template file.)

## After results land
```bash
# rsync results back into export/ocr_results_referenced/ (re-OCR'd crops overwrite in place), then:
bash reingest.sh        # re-extract → merge → QA → half-page merge → continuation stitch → DB → re-validate
python3 build_static_site.py
```
`reingest.sh` now re-runs the full triage (steps 9-10) automatically, so any tables
the cluster fixes drop out of the imperfect set and the perfect-rate climbs.

## Expectation
Realistically the cluster will fix the digit-error / sum-fail cases and the genuinely
recoverable dropped columns. Some very wide / faint tables may still resist OCR — those
become the small residual for a targeted Opus pass.
