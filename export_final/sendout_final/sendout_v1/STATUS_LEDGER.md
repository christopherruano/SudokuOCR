# HEB 91r — Status Ledger (ducks-in-a-row audit)

Snapshot before the cluster re-OCR trip. Every number below reconciles across files.

## 1. Table universe — RECONCILED
| | count |
|---|---|
| Existing on-site/queue tables (`catalog_tables.csv`) | 3,647 |
| Keeper tables (`export/keeper_inference/source_catalog.csv`) | 12,258 |
| **Provenance rows (`provenance.csv`)** | **15,905** = 3,647 + 12,258 ✓ |
| Distinct source documents (dochash) | 265 |
| — with source PDF resolved | 194 |
| — needing PDF-filename label | 71 (48 of them are gap docs) |

Existing-catalog status: confirmed_sent 1,735 · recovered 870 · needs_reocr 394 ·
needs_human_review 21 · hold_reocr 15 · excluded_or_unscoped 582 · excluded_rejected 20 ·
excluded_malformed 7 · discarded 3  (= 3,647 ✓)

## 2. Page-number pass — 99.95% coverage
- Haiku read 11,081 / 11,087 multi-doc crops (40 dropped in main pass, 34 recovered by top-up; last 6 are duplicate-image crops covered at page level).
- True crop identity recovered positionally + image-verified (97%), since Haiku mangled echoed IDs. `page_reads_fixed.json`.

## 3. Multi-page stitching (Neil's "one heading")
- 2,357 logical tables · 912 multi-page · 247 complete (no interior gaps)
- 127 multi-page Table VII instances (Imperial age/civil-condition)
- `stitched_tables.csv` (range + ordered pages + missing pages), `page_gaps.csv`

## 4. Missing-page universe check — 0 truly absent
4,593 interior missing page-slots across 183 documents, **all volumes we physically hold**:
| Bucket | Pages | Action |
|---|---|---|
| Un-read scan already on disk | 274 (5%) | Place now |
| Re-extractable from held PDF | 4,319 (94%) | Targeted cluster re-OCR |
| Truly absent | 0 | — |
Confirmed against manifest `resolved_start_page`/`resolved_end_page`: the extractor
pulled PDF pages in chunks and skipped interior pages that are still in the PDF.
`universe_coverage.csv` (per page), `gap_summary.csv` (per doc), `UNIVERSE_REPORT.md`.

## 5. Re-OCR queue (bad existing tables) — built, not yet run
- `reocr_images.txt` (408 images), `reocr_manifest.csv`, `fasrc/rerun_reocr.sbatch`

## 6. OPEN ITEMS (loose ducks)
1. **Keepers not in queryable dataset** — `unified_facts.csv` has 2,912 tables, ALL existing-catalog, **0 keepers**. The 12,258 keepers are catalogued + provenance-linked + in stitch/universe, but NOT folded into the explorer/queryable facts yet. (cell_facts_rich.csv extracted; merge step pending.)
2. **71 docs need PDF-filename recovery** (re-run digitization manifest join).
3. **9.4 GB of deliverables are untracked, local-disk only** (browse 5.1G, sendout_final/data 1.2G, census_site 1.1G) — too big for git; no off-machine backup. Risk if disk lost.
4. 6 crops still unread (negligible, duplicate images).

## 7. Cluster plan (next)
- Build a **per-PDF page-range re-OCR manifest** for the 4,319 re-extractable pages (precise: "OCR pages 21–57 of saoa.crl.25366884.pdf"), using manifest resolved-page ranges — better targeting than crop-based.
- Plus the 408-image bad-table queue.
- Then re-ingest → stitch fills → merge keepers into queryable dataset/site.
