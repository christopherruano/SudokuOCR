# Assistant-Reviewed Failing Tables

This directory applies the assistant's review to the failing kept-table pack.

Buckets:
- `salvage/`: on-topic tables that likely need family-specific repair, multipage assembly, or better axis interpretation
- `keep_with_caution/`: near-acceptable tables with only minor residual issues
- `drop/`: tables that should not be treated as part of the age-table project deliverable

Within each bucket:
- `<bucket>.csv`: full list
- `<dataset>.csv`: layer-specific list
- `preview_links/<dataset>/`: symlinks to the corresponding preview folders
- `top_examples.csv`: highest-priority examples in that bucket

Counts:
- salvage: 49
- keep_with_caution: 5
- drop: 234
