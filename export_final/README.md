# export_final

This is the human-browsable final package for exploration and analysis.

## Start Here
- `docs/README.md`: package overview
- `docs/PROJECT_CODEBOOK.md`: practical usage guide
- `indexes/dataset_entrypoints.csv`: direct paths to the three canonical datasets
- `browse/by_type/`: browse tables by measurement semantics
- `browse/by_topic/mortality/`: dedicated mortality/death-rate subcollection
- `browse/preview_pack/`: per-table previews
- `browse/excel_previews/`: Excel workbooks for fast inspection
- `browse/failing_tables/assistant_reviewed/`: assistant-triaged failing-table slice
- `qa/`: package QA reports

## Canonical datasets
- `datasets/core_curated_v2_final/`: primary paper dataset
- `datasets/tail_exploratory_v2/`: broader secondary layer
- `datasets/assembled_strong_v2/`: reviewed multipage assembled layer

## Measurement-semantic browsing
- `browse/by_type/counts/`: additive count tables
- `browse/by_type/distribution_of_1000/`: compositional `out of 1,000` tables
- `browse/by_type/per_1000_rates/`: generic per-1,000 rate tables
- `browse/by_type/other_rates/`: percent/rate tables outside the 1,000-distribution bucket

## Important note on raw OCR page JSON
- The release ships table-level metadata JSON and preview JSON, not the full raw OCR page JSON corpus.
- `source_json` fields are references to the original OCR outputs, typically on the cluster path.
