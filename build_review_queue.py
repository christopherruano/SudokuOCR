"""Build a manual review queue from the top gap candidates.

Extracts sample paths for the most plausible primary-gap recovery targets.

Run on FASRC:
    python3 build_review_queue.py
"""

import json
from pathlib import Path

GAP_JSON = Path("~/census_toc/gap_candidates.json").expanduser()
OUTPUT = Path("~/census_toc/review_queue.txt").expanduser()

rows = json.load(open(GAP_JSON))["rows"]

keep = {
    ("Madras", 1931),
    ("Bombay", 1931),
    ("Punjab", 1931),
    ("Baluchistan", 1901),
    ("Kashmir", 1921),
}

print("Review queue for primary-gap recovery targets:")
print("=" * 80)

all_paths = []
for row in rows:
    key = (row["province"], row["year"])
    if key not in keep:
        continue
    print(
        f"\n{row['province']} {row['year']}"
        f"  strong={row['cluster_strong_primary_tables']}"
        f"  secondary={row['cluster_secondary_structured_tables']}"
    )
    for title in row["sample_titles"][:6]:
        print(f"  title: {title}")
    for path in row["sample_paths"][:6]:
        print(f"  path:  {path}")
        all_paths.append(path)

with open(OUTPUT, "w") as f:
    for p in all_paths:
        f.write(p + "\n")

print(f"\nWrote {len(all_paths)} paths to {OUTPUT}")
