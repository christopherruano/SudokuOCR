# TOC-First Census Manifest

## Goal

Build a document manifest that says, for each census PDF:

- what the volume is
- which tables/sections it contains
- which table topics are relevant to the age-table pipeline
- which PDF pages should be considered first for downstream extraction

This manifest is the control layer for later table OCR and numeric extraction.

## Why TOC-First

Historical census volumes are not uniform. A robust pipeline cannot start by reading every page as if it were equally important. The table of contents, list of tables, and section headings define the intended structure of the document. That structure should drive:

- page relevance
- table classification
- extraction priority
- downstream query schema

But TOC OCR is not reliable enough on its own, so the pipeline must use multiple evidence sources.

## Evidence Model

`build_toc_manifest.py` combines three evidence sources:

1. Front-matter text extraction with `pdftotext`
2. OCR fallback on front pages when text extraction is blank or boilerplate
3. Body-header mining when TOC parsing is absent or weak

The result is a document manifest plus flat table records.

## Output Levels

The document manifest stores:

- document metadata: year, province, title, page count
- TOC pages detected and why
- front-page extraction provenance (`pdftotext` vs `ocr`)
- parsed entries with canonical topic tags
- human review decision when available

The flat table records store one row per discovered entry and are intended for:

- filtering by topic
- queue construction
- later linking to extracted tables and derived datasets

## Canonical Topic Logic

The current manifest tags entries into canonical topics such as:

- `age_sex_population`
- `age_sex_civil_condition`
- `age_sex_marital_status`
- `age_by_religion`
- `age_distribution`
- `literacy_tables`
- `language_tables`
- `economic_tables`
- `housing_tables`
- `migration_tables`

This is the right level for the conductor layer: it is semantic enough to drive extraction, but still close to the printed census structure.

## Recommended Workflow

1. Run `build_toc_manifest.py` on the corpus.
2. Review the flat table rows for `relevance_tier=core`.
3. Build extraction queues from `best_pdf_page`, `printed_page_label`, and canonical topic.
4. For documents with no TOC and no body headers, escalate to manual/OCR review.
5. Keep all downstream numeric extraction keyed by:
   - source file
   - discovered table identifier/title
   - canonical topic
   - page provenance

## Slurm Integration

Use the manifest to define extraction jobs, not the raw PDF directory.

Suggested unit of work:

- one manifest table record = one extraction target

That lets you:

- prioritize age-relevant tables first
- skip clearly irrelevant sections
- rerun failed tables without touching the rest of the volume
- preserve provenance for every extracted output

## Academic Robustness

For statistical inference, each extracted dataset should be traceable to:

- the original PDF
- the manifest record that justified extraction
- the page(s) used
- the parser/model version used to read it
- any human override or adjudication

That provenance chain is what makes the later quantitative analysis defensible.

## Current Limits

- Some volumes still have weak or absent TOC signals.
- Body-header mining currently relies on text extraction, so image-only bodies may still need OCR escalation.
- OCR quality on badly degraded scans will vary by decade and print quality.

This is acceptable as long as unresolved documents are explicitly surfaced rather than silently treated as negatives.

## Example Run

```bash
python3 build_toc_manifest.py /Volumes/Expansion/census_pdfs \
  --adjudications-dir review_annotations_pdf \
  --output /tmp/toc_manifest.json \
  --tables-output /tmp/toc_tables.jsonl \
  --tables-csv /tmp/toc_tables.csv
```

For OCR-heavy corpora, add a persistent cache:

```bash
python3 build_toc_manifest.py /Volumes/Expansion/census_pdfs \
  --ocr-cache-dir /tmp/census_toc_ocr_cache \
  --output /tmp/toc_manifest.json
```
