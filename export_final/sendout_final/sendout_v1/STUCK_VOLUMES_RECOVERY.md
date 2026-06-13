# 41 Table VII volumes blocked on cluster PDF paths

These 41 volumes contain Table VII (and related age/civil-condition tables) and are
already partially read — **1,370 keeper tables** across them — but we **cannot compute
their tight OCR page ranges** because their `source_pdf` path and printed→PDF page
offset are unknown locally.

## Why they're stuck
- They are **keeper-only** volumes (came from the flash-drive/keeper recovery, `n_existing=0`).
- They do **not** appear in `export/cluster_sync/organized_manifest.jsonl`, and
  `documents.csv` has no `source_pdf` for them.
- The dochash→PDF mapping for these exists **only on the cluster** (in the
  `census_toc` digitization metadata). All the other 95 volumes resolved to
  `high`-confidence ranges; only these 41 lack the PDF anchor.

## What we already know (local)
- `stuck_volumes_need_cluster_pdf.csv` — dochash, year, region, n_keeper, seq range, tables present.
- `stuck_dochashes.json` — just the 41 dochashes.
- The crops for each live on the cluster at:
  `/n/home11/cruano/census_toc/digitized_tables/<dochash>_t<NNNN>_<TABLE>/<imghash>.png`

## To unblock (cluster-side, user action)
For each dochash in `stuck_dochashes.json`, recover from the cluster:
1. the **source PDF path** that produced `census_toc/digitized_tables/<dochash>_t*`
   (whatever digitization manifest the TOC pipeline wrote — the same record that
   feeds `digitized_source_file`), and
2. the PDF **page count**.

Then we can:
- learn the printed→PDF offset (we have read printed pages for these volumes), and
- emit their tight Table VII OCR windows just like the other 95.

Drop the resulting `dochash,source_pdf,n_pages` table back into the repo and re-run
`validate_page_ranges.py` → `build_tight_ocr_queue.py`.
