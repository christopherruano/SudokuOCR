# SudokuOCR Sendout Final

This folder is the collaborator-ready local release. Start with `data/tables_confirmed.csv` for the one-row-per-table catalog and `data/facts_confirmed.csv` for the lean numeric cell facts.

One row in `facts_confirmed.csv` is one confirmed numeric cell from one confirmed table. Table-level context such as title, province, year, source paths, confidence, and workbook location lives in `data/tables_confirmed.csv` so the fact table stays compact and analytical.

Excel workbooks are under `excel_tables/`. The layer folders (`core/`, `tail/`) contain the physical workbooks; `excel_tables/by_family/` and `excel_tables/mortality/` contain symlinks for browsing.

Confidence levels come from the team-ready quality layer. This sendout includes only strict validated, in-scope, zero-failure, non-assembled tables with no unresolved manual-review signal, then removes tables that fail extra format-aware number-sense checks.

Mortality is separated under `mortality/`, with its own confirmed table/fact extracts and mortality-focused index.

Excluded and review-needed tables are documented in `data/exclusions.csv`, `data/review_needed.csv`, `excluded/`, and `review/`. Review-needed rows are not in `facts_confirmed.csv`.

Counts:
- confirmed tables: 1933
- confirmed facts: 734157
- mortality tables: 26
- excluded tables: 582
- review-needed tables: 1132
