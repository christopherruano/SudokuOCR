# Final Submission Review Queue

This queue contains every table currently in `browse/by_quality/needs_review`.

Files:
- `review_decisions.csv`: canonical editable review sheet
- `all_tables.csv`: same rows, for reference
- `core_tables.csv`, `tail_tables.csv`: layer-specific subsets
- `suggested_drop.csv`, `suggested_hold.csv`, `suggested_keep.csv`: assistant-seeded buckets
- `final_submission_review.xlsx`: browse workbook
- `previews/tables/`: one preview folder per table

Review app:
- `python3 final_submission_review_app.py`
- default review sheet: `export_final/browse/final_submission_review/review_decisions.csv`
- default URL: `http://127.0.0.1:8767`

Asset sync for real page images:
- full queue:
  - `bash ./sync_release_review_assets.sh cruano@login.rc.fas.harvard.edu /n/home11/cruano/census_toc export_final/browse/final_submission_review/review_decisions.csv export_final/browse/final_submission_review/review_assets`
- after sync, the review app shows the real page image inline and exposes `Open Source OCR JSON`

Decision semantics:
- `keep`: include in the final project-facing submission
- `drop`: exclude from the final project-facing submission
- `hold`: unresolved / not ready to include

Suggested decisions are pre-seeded from the assistant failing-table triage where available.

Total tables: 370
Core: 72
Tail: 298
