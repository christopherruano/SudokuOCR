"""Filter gap candidates to strict shortlist: only real Table VII matches.

Run on FASRC:
    python3 filter_gap_shortlist.py
"""

import json
import re
from pathlib import Path

GAP_JSON = Path("~/census_toc/gap_candidates.json").expanduser()

rows = json.load(open(GAP_JSON))["rows"]

table_pat = re.compile(
    r"(?:imperial\s+)?table\s+(?:vii|vii\.|vil|7)\b", re.I
)
phrase_pat = re.compile(
    r"\bage\s*,?\s*sex\s+and\s+civil\s+condition\b", re.I
)

keep = []
for r in rows:
    if r["cluster_strong_primary_tables"] <= 0:
        continue
    titles = [t or "" for t in r["sample_titles"]]
    if any(table_pat.search(t) or phrase_pat.search(t) for t in titles):
        keep.append(r)

print(f"strict shortlist: {len(keep)}")
for r in keep[:40]:
    print(
        f"\n{r['year']} | {r['province']}"
        f" | guide={r['guide_primary_tables']}p/{r['guide_total_tables']}t"
        f" | strong={r['cluster_strong_primary_tables']}"
        f" | secondary={r['cluster_secondary_structured_tables']}"
    )
    for t in r["sample_titles"][:3]:
        print(f"  title: {t[:120]}")
    for s in r["sample_paths"][:3]:
        print(f"  path: {s}")
