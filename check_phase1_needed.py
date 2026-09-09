#!/usr/bin/env python3
"""Check how many of the Phase 1 queue entries actually need source PDFs.

For each entry, count pages already extracted in census_toc/digitized_tables/
vs pages needed (resolved_end_page - resolved_start_page + 1).
"""
import json
from pathlib import Path

HOME = Path.home()
DIG = HOME / "census_toc" / "digitized_tables"
QUEUE = HOME / "census_toc" / "tableVII_newocr.jsonl"

with QUEUE.open() as f:
    queue = [json.loads(line) for line in f]

complete = 0
partial_rows = []
missing_rows = []

for entry in queue:
    table_idx = entry.get("table_index", 0)
    start = entry.get("resolved_start_page", 0)
    end = entry.get("resolved_end_page", 0)
    needed = max(0, end - start + 1)

    matches = list(DIG.glob(f"*_t{table_idx:04d}_*"))
    pngs = sum(len(list(m.glob("*.png"))) for m in matches)

    if pngs == 0:
        missing_rows.append((entry.get("filename", ""), table_idx, needed))
    elif pngs >= needed:
        complete += 1
    else:
        partial_rows.append((entry.get("filename", ""), table_idx, pngs, needed))

print(f"Total queue entries: {len(queue)}")
print(f"  Already complete (have all pages): {complete}")
print(f"  Partial (some pages missing):      {len(partial_rows)}")
print(f"  Nothing extracted yet:             {len(missing_rows)}")
print()
print("Sample partials (first 5):")
for row in partial_rows[:5]:
    print(f"  {row}")
print()
print("Sample missing (first 5):")
for row in missing_rows[:5]:
    print(f"  {row}")
