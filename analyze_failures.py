"""Analyze constraint failures: what's actually failing and by how much.

Run on FASRC:
    python3 analyze_failures.py
"""

import json
import os
from collections import Counter

RESULTS_ROOT = os.path.expanduser("~/census_toc/ocr_results")

level_counts = Counter()
diff_sizes = {"L1": [], "L2": [], "L3": [], "L4": [], "L5": []}
small_diff = 0  # diff <= 1 (likely rounding or off-by-one)
total_failures_count = 0
files_by_nfail = Counter()
sample_by_level = {}

n_files = 0
for dirpath, _, filenames in os.walk(RESULTS_ROOT):
    for fn in sorted(filenames):
        if not fn.endswith("_oneshot.json"):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            d = json.load(open(fp))
        except Exception:
            continue

        c = d.get("constraints", {})
        n_failed = c.get("failed", 0)
        if n_failed == 0:
            continue

        n_files += 1
        n_checks = c.get("total_checks", 0)
        files_by_nfail[n_failed] += 1

        for f in c.get("failures", []):
            total_failures_count += 1
            lvl = f.get("level", "?")
            level_counts[lvl] += 1
            diff = abs(f.get("diff", 0))
            if lvl in diff_sizes:
                diff_sizes[lvl].append(diff)
            if diff <= 1:
                small_diff += 1

            if lvl not in sample_by_level:
                sample_by_level[lvl] = []
            if len(sample_by_level[lvl]) < 5:
                sample_by_level[lvl].append({
                    "file": fn,
                    "detail": f.get("detail", ""),
                    "section": f.get("section", ""),
                    "age": f.get("age", ""),
                    "diff": f.get("diff", 0),
                })

print(f"Files with constraint failures: {n_files}")
print(f"Total individual check failures: {total_failures_count}")
print()

print("=" * 60)
print("FAILURES BY LEVEL")
print("=" * 60)
for lvl in ["L1", "L2", "L3", "L4", "L5", "?"]:
    if lvl in level_counts:
        n = level_counts[lvl]
        diffs = diff_sizes.get(lvl, [])
        if diffs:
            avg = sum(diffs) / len(diffs)
            med = sorted(diffs)[len(diffs) // 2]
            tiny = sum(1 for d in diffs if d <= 1)
            small = sum(1 for d in diffs if d <= 5)
            desc = {
                "L1": "P != M+F (row horizontal)",
                "L2": "Total != sum(ages) (column vertical)",
                "L3": "Cross-group (e.g. All != Hindu+Muslim+...)",
                "L4": "Cross-section",
                "L5": "Negative values",
            }.get(lvl, lvl)
            print(f"\n{lvl}: {n:,} failures — {desc}")
            print(f"  Avg diff: {avg:.1f}  Median diff: {med}")
            print(f"  Diff <= 1: {tiny}  Diff <= 5: {small}")
        else:
            print(f"\n{lvl}: {n:,} failures")

print()
print("=" * 60)
print("FILES BY NUMBER OF FAILURES")
print("=" * 60)
for nf in sorted(files_by_nfail):
    print(f"  {nf:>3} failures: {files_by_nfail[nf]:>4} files")

just_one = files_by_nfail.get(1, 0)
two_or_less = sum(v for k, v in files_by_nfail.items() if k <= 2)
print(f"\n  1-2 failures (likely format/rounding): {two_or_less} files")
print(f"  3+ failures (likely real errors):      {n_files - two_or_less} files")

print()
print("=" * 60)
print(f"SMALL DIFFS (diff <= 1): {small_diff} / {total_failures_count} "
      f"({100*small_diff/max(total_failures_count,1):.0f}%)")
print("=" * 60)
print("These are likely format issues, not actual misreads.")

print()
print("=" * 60)
print("SAMPLE FAILURES BY LEVEL")
print("=" * 60)
for lvl in ["L1", "L2", "L3", "L4", "L5"]:
    samples = sample_by_level.get(lvl, [])
    if not samples:
        continue
    print(f"\n--- {lvl} samples ---")
    for s in samples:
        print(f"  {s['file'][:50]}")
        print(f"    section={s['section'][:30]}  age={s['age']}  diff={s['diff']}")
        print(f"    {s['detail'][:80]}")
