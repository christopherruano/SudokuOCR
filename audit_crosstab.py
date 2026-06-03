"""Cross-tab the recall audit: primary vs tail, by bucket, topic, province-year.

Run on FASRC:
    python3 audit_crosstab.py
"""

import csv
import collections
from pathlib import Path

p = Path("~/census_toc/age_recall_audit/candidate_inventory.csv").expanduser()
bucket = collections.Counter()
topic = collections.Counter()
provyr = collections.Counter()

with p.open() as fh:
    for row in csv.DictReader(fh):
        if row["classic_primary_candidate"] == "1":
            level = "primary"
        elif row["age_structured_tail_candidate"] == "1":
            level = "tail"
        else:
            continue

        key = (level, row["audit_bucket"])
        bucket[key] += 1

        if row["audit_bucket"] in {"never_ocrd", "ocr_non_age", "dropped_by_inference_qc"}:
            topic[(level, row["audit_bucket"], row["canonical_topic"])] += 1
            provyr[(level, row["audit_bucket"], row["province"], row["year"])] += 1

print("Bucket x level")
print("=" * 60)
for k, v in sorted(bucket.items()):
    print(f"  {v:>5}  {k}")

print("\nTop missing topics")
print("=" * 60)
for k, v in topic.most_common(30):
    print(f"  {v:>5}  {k}")

print("\nTop province-years with misses")
print("=" * 60)
for k, v in provyr.most_common(30):
    print(f"  {v:>5}  {k}")
