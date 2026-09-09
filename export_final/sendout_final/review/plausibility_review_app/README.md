# Plausibility Review App

Run from the repo root:

```bash
python3 plausibility_review_app.py
```

Then open http://127.0.0.1:8768.

This app reads `review_decisions.csv` and lets you compare the table metadata, local source image, plausibility issues, generated matrix preview, fact rows, and generated Excel workbook paths where available.

Current index summary is in `summary.json`. Decisions are written back into `review_decisions.csv`.

Useful queue files:
- `review_has_entries.csv`: rows with fact entries to inspect
- `review_has_entries_no_preview.csv`: rows that have facts but need preview generation
- `review_no_facts_regenerate_candidates.csv`: rows with no facts that are plausible rerun candidates
- `review_no_facts_reference_only.csv`: rows with no facts kept only as reference/scope material
