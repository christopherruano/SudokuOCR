# Multi-page stitching + missing-page universe check

Source: full Haiku page-number pass over 11,087 multi-document crops, recovered to
true crop identity (positional + image-verified, 97%), stitched by (document,
table-number) into contiguous runs, then a targeted Haiku read of every un-read
scan we hold in gap-bearing documents.

## Stitching (Neil's "combine under one heading")
- **2,339 logical tables**, **910 multi-page**, **247 complete** (no interior gaps)
- **127 multi-page Table VII instances** (Imperial age/civil-condition); 27 complete
- `stitched_tables.csv` — one row per logical table: page range, ordered pages, missing pages

## Missing pages — do we have them in our universe?
4,583 interior missing page-slots across **183 documents — every one a volume we
physically hold** (we have OCR'd tables from all of them). So **0 are truly absent.**

| Bucket | Pages | Action |
|---|---|---|
| Un-read scan already on disk | **274 (5%)** | Place now — no cluster needed |
| Re-extractable from the held volume | **4,309 (94%)** | Targeted cluster re-OCR of those pages |
| Truly absent (no volume) | **0** | — |

48 of 183 gap documents still need their **source-PDF filename** recovered (we hold
their pages; only the human-readable label is missing).

## Files
- `stitched_tables.csv`, `page_gaps.csv` — stitched tables + per-document gaps
- `universe_coverage.csv` — per (document, missing_page): `have_scan` vs `reextract` (+ imghash when on disk)
- `gap_summary.csv` — per document: missing / have-scan / re-extract counts
- `page_reads_fixed.json`, `universe_reads.json` — recovered page reads
