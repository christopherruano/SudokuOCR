# FASRC Slurm Runbook

This runbook is for the TOC-first census PDF workflow on Harvard FASRC.

The workflow is:

1. Build a stable PDF filelist
2. Run TOC/document-manifest jobs as a Slurm array
3. Merge manifest outputs
4. Run page-range resolution as a second Slurm array
5. Merge resolved ranges into:
   - a digitization queue
   - a manual-review queue
6. Render only the selected page windows for the digitization queue

## Why this structure

The manifest stage decides what each volume contains.
The range stage decides which PDF pages belong to which table.
Only after those two stages should any expensive digitization/extraction begin.

That is how you remove irrelevant pages before downstream OCR and numeric extraction.

## FASRC notes

- Use `$SCRATCH` for high-I/O transient data. On Cannon/FASSE it currently points to `/n/netscratch`.
- `serial_requeue` is appropriate for small high-throughput jobs and is recommended by FASRC for small jobs.
- FASRC allows job arrays, but keep limits in mind:
  - max array size `10,000`
  - max jobs per user `10,100`
- Always request memory explicitly. Cannon defaults are too small for this workflow.
- After a pilot, use `jobstats JOBID` to tune memory/time.
- Use `spart` to see which partitions you can access.

Official references:

- Running Jobs: https://docs.rc.fas.harvard.edu/kb/running-jobs/
- Submitting Large Numbers of Jobs: https://docs.rc.fas.harvard.edu/kb/submitting-large-numbers-of-jobs/
- Scratch: https://docs.rc.fas.harvard.edu/kb/policy-scratch/
- jobstats: https://docs.rc.fas.harvard.edu/kb/jobstats/
- Convenient Slurm Commands: https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/

## One-time setup

On the cluster:

```bash
cd /path/to/SudokuOCR
bash fasrc/setup.sh
```

## Environment

Set these once in your shell before submission:

```bash
export PROJECT_DIR=/path/to/SudokuOCR
export CENSUS_ROOT=/path/to/census_pdfs
export CENSUS_SCRATCH_ROOT="$SCRATCH/<lab_name>/<your_name>/census_toc"
export VENV_DIR="$HOME/.envs/sudokuocr"
mkdir -p "$CENSUS_SCRATCH_ROOT"/{inputs,manifest_tasks,range_tasks,merged,digitized_tables,ocr_cache}
cd "$PROJECT_DIR"
```

If your review annotations live in the repo:

```bash
export CENSUS_ADJUDICATIONS_DIR="$PROJECT_DIR/review_annotations_pdf"
```

## Preflight and first-pilot defaults

Use these defaults for the first full pilot:

```bash
export TOC_FRONT_PAGES=80
export TOC_FRONT_CONTENT_PAGES=25
export TOC_OCR_FRONT=auto
export TOC_BODY_MAX_PAGES=350
export MANIFEST_CONCURRENCY=20
export RANGE_CONCURRENCY=40
export DIGITIZE_CONCURRENCY=40
```

Preflight:

```bash
bash slurm/preflight_pilot.sh
```

## Build the filelist

```bash
python3 build_pdf_filelist.py "$CENSUS_ROOT" \
  --output "$CENSUS_SCRATCH_ROOT/inputs/census_pdfs.txt"
```

Count the work:

```bash
N_PDFS=$(wc -l < "$CENSUS_SCRATCH_ROOT/inputs/census_pdfs.txt" | tr -d ' ')
echo "$N_PDFS PDFs"
```

## Tiny pilot: manifest stage

Run a very small sanity pilot first:

```bash
sbatch --array=0-4%2 slurm/toc_manifest_array.sbatch
```

Monitor:

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
jobstats <JOBID>
```

## First full pilot: manifest stage

```bash
bash slurm/submit_manifest_stage.sh
```

The manifest jobs are designed to be idempotent and requeue-safe:

- one JSON result per PDF
- atomic writes
- `--skip-existing` enabled

## Merge manifest outputs

```bash
bash slurm/submit_merge_stage.sh
```

Main artifacts:

- `$CENSUS_SCRATCH_ROOT/merged/toc_manifest_merged.json`
- `$CENSUS_SCRATCH_ROOT/merged/toc_documents.jsonl`
- `$CENSUS_SCRATCH_ROOT/merged/toc_tables.jsonl`
- `$CENSUS_SCRATCH_ROOT/merged/toc_tables.csv`

## Tiny pilot: range-resolution stage

Count merged documents:

```bash
N_DOCS=$(wc -l < "$CENSUS_SCRATCH_ROOT/merged/toc_documents.jsonl" | tr -d ' ')
echo "$N_DOCS documents"
```

Pilot:

```bash
sbatch --array=0-4%2 slurm/resolve_ranges_array.sbatch
```

## First full pilot: range-resolution stage

```bash
bash slurm/submit_range_stage.sh
```

Then merge again:

```bash
bash slurm/submit_merge_stage.sh
```

Main range artifacts:

- `$CENSUS_SCRATCH_ROOT/merged/ranges_merged.json`
- `$CENSUS_SCRATCH_ROOT/merged/range_tables.jsonl`
- `$CENSUS_SCRATCH_ROOT/merged/range_tables.csv`
- `$CENSUS_SCRATCH_ROOT/merged/digitization_queue.jsonl`
- `$CENSUS_SCRATCH_ROOT/merged/manual_review_queue.jsonl`

If you want to restrict digitization to a narrower set of canonical topics,
set a comma-separated filter before the merge step. For example, to keep only
strict age-by-sex style tables:

```bash
export CENSUS_DIGITIZATION_TOPICS="age_sex_population,age_sex_civil_condition,age_sex_marital_status"
```

## Review unresolved core items

Check how many core items still need manual review:

```bash
wc -l "$CENSUS_SCRATCH_ROOT/merged/manual_review_queue.jsonl"
```

Those are the tables where the workflow found a relevant entry but could not safely assign a page window.

## Tiny pilot: digitization page rendering

Count queue items:

```bash
N_QUEUE=$(wc -l < "$CENSUS_SCRATCH_ROOT/merged/digitization_queue.jsonl" | tr -d ' ')
echo "$N_QUEUE queue items"
```

Pilot:

```bash
sbatch --array=0-4%2 slurm/digitize_table_pages_array.sbatch
```

## First full pilot: digitization page rendering

```bash
bash slurm/submit_digitization_stage.sh
```

This stage renders only the selected page windows and writes:

- per-page PNGs
- `metadata.json`
- per-page `.txt` sidecars from `pdftotext` or OCR fallback

under:

```bash
$CENSUS_SCRATCH_ROOT/digitized_tables/
```

Before downstream OCR, inspect what the rendered corpus actually contains:

```bash
python3 inspect_digitized_tables.py \
  "$CENSUS_SCRATCH_ROOT/digitized_tables" \
  --results-root "$CENSUS_SCRATCH_ROOT/ocr_results"
```

Then build an explicit OCR image list from TOC-derived metadata instead of
recursively scanning every PNG:

```bash
python3 build_digitized_ocr_manifest.py \
  "$CENSUS_SCRATCH_ROOT/digitized_tables" \
  --strict-age-only \
  --output "$CENSUS_SCRATCH_ROOT/merged/ocr_image_list.txt" \
  --json-output "$CENSUS_SCRATCH_ROOT/merged/ocr_image_list.jsonl"

export IMAGE_LIST="$CENSUS_SCRATCH_ROOT/merged/ocr_image_list.txt"
export IMAGE_ROOT="$CENSUS_SCRATCH_ROOT/digitized_tables"
export RESULTS_DIR="$CENSUS_SCRATCH_ROOT/ocr_results"
sbatch --export=ALL --array=0-29%6 fasrc/submit.sh
```

## Tuned first-pilot resources

Manifest stage:

- `1` core
- `8G` RAM
- `6` hours
- `20` concurrent array tasks
- body mining capped at `350` pages for pilot throughput

Range stage:

- `1` core
- `4G` RAM
- `2` hours
- `40` concurrent array tasks

Digitization page-render stage:

- `1` core
- `6G` RAM
- `3` hours
- `40` concurrent array tasks

Tune after the pilot with `jobstats`.

## Useful commands

See partitions you can access:

```bash
spart
```

Watch queue:

```bash
squeue -u "$USER"
```

Inspect completed jobs:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
jobstats <JOBID>
```

Cancel a runaway array:

```bash
scancel <JOBID>
```

## Important constraint

Do not start numeric table extraction from the raw PDF tree.
Always start from the merged digitization queue, because that queue is the output of:

1. semantic document classification
2. page-range restriction
3. explicit unresolved-case surfacing

That is the provenance chain you want for later academic analysis.
