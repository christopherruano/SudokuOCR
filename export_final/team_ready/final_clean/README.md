# Final Clean Demo Set

This folder is the no-embarrassment subset for team presentation.

Inclusion rule:
- in shipped release scope
- `quality_tier = strict_validated`
- not still flagged for manual review

Counts:
- total clean tables: 2221
- core: 1575
- tail: 646
- assembled: 0

Use:
- `catalogs/final_clean_tables.csv` for the main demo table list
- `catalogs/final_clean_mortality.csv` for the mortality-only clean subset

Excluded from this clean set:
- all non-strict release tables
- all tables still needing manual review
- all assembled tables for now, because the assembled layer still needs semantic review even when constraint checks pass
