# Failing Tables Review Pack

This directory contains only kept release tables with nonzero shipped `failed_checks`.

Files:
- `index.csv`: all failing kept tables
- `top_50_worst.csv`: highest-failure subset
- `core_failed.csv`, `tail_failed.csv`: layer-specific lists
- one folder per failing table with `facts.csv`, `matrix.csv`, `source_catalog_row.json`, `interpretation.txt`, `summary.txt`

Notes:
- These are not all 'bad tables'. Some are close misses or proportional-tolerance issues.
- Use `effective_quality_status`, `failed_checks`, and `qc_flags` to interpret severity.
- core failing kept tables: 26
- tail failing kept tables: 262
