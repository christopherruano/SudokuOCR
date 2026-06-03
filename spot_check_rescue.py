"""Spot-check rescued tables from the strict rescue pass.

Run on FASRC:
    python3 spot_check_rescue.py
"""

import csv
import random
from pathlib import Path

for name in ["core_rescue.csv", "core_review.csv", "tail_rescue.csv"]:
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "~/census_toc/non_age_rescue_v3"
    p = Path(base).expanduser() / name
    if not p.exists():
        continue
    rows = list(csv.DictReader(p.open()))
    random.seed(42)
    sample = random.sample(rows, min(20, len(rows)))
    print(f"\n{'='*80}")
    print(f"{name} ({len(rows)} rows) — 20 random samples:")
    print("=" * 80)
    for r in sample:
        print(
            f"  {r['province']} {r['year']} | "
            f"{r['canonical_topic']} | "
            f"{r.get('title_from_payload', '')[:100]} | "
            f"axis={r.get('row_axis_type', '')} | "
            f"label={r.get('row_label_name', '')} | "
            f"family={r['rescue_family']} | "
            f"{r['rescue_reasons']}"
        )
