"""QA check: what changed in the core rescue export.

Run on FASRC:
    python3 qa_core_rescue.py
"""

import csv
from pathlib import Path

OLD_CATALOG = Path("~/census_toc/export_inference_rich/source_catalog.csv").expanduser()
NEW_CATALOG = Path("~/census_toc/export_inference_plus_core_rescue_v5/source_catalog.csv").expanduser()
CORE_RESCUE = Path("~/census_toc/non_age_rescue_v5/core_rescue_results.txt").expanduser()

# Check 1: what sources were newly added
print("=" * 60)
print("CHECK 1: Newly added sources")
print("=" * 60)

old = set()
with OLD_CATALOG.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if (row.get("keep_for_inference") or "").lower() in {"1", "true", "yes"}:
            old.add(row["source_json"])

new = set()
with NEW_CATALOG.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if (row.get("keep_for_inference") or "").lower() in {"1", "true", "yes"}:
            new.add(row["source_json"])

added = sorted(new - old)
print(f"Newly added sources: {len(added)}")
for p in added[:60]:
    print(f"  {p}")

# Check 2: which core_rescue items made it through
print()
print("=" * 60)
print("CHECK 2: Core rescue merge coverage")
print("=" * 60)

catalog = set()
with NEW_CATALOG.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        catalog.add(row["source_json"])

core = [line.strip() for line in CORE_RESCUE.open(encoding="utf-8") if line.strip()]
missing = [p for p in core if p not in catalog]

print(f"Core rescue total:      {len(core)}")
print(f"Core rescue merged:     {len(core) - len(missing)}")
print(f"Core rescue not merged: {len(missing)}")
for p in missing[:30]:
    print(f"  {p}")
