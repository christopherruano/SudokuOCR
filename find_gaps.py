"""Find civil condition / age tables in OCR results that fill coverage gaps.

Run on FASRC:
    python3 find_gaps.py [--results ~/census_toc/ocr_results]
"""

import json
import os
import sys
from collections import defaultdict

RESULTS_ROOT = os.path.expanduser("~/census_toc/ocr_results")

if "--results" in sys.argv:
    idx = sys.argv.index("--results")
    RESULTS_ROOT = sys.argv[idx + 1]


def is_civil_condition(meta):
    title = (meta.get("title") or "").lower()
    col_groups = meta.get("column_groups", [])
    col_str = " ".join(g.lower() for g in col_groups)
    return ("civil condition" in title or "civil condition" in col_str
            or "marital" in title
            or any(x in col_str for x in ["unmarried", "married", "widowed"]))


def is_age_sex(meta):
    title = (meta.get("title") or "").lower()
    col_groups = meta.get("column_groups", [])
    col_str = " ".join(g.lower() for g in col_groups)
    return ("age" in title and "sex" in title
            or "population" in col_str and ("male" in col_str or "female" in col_str))


def get_status(d):
    c = d.get("constraints", {})
    checks = c.get("total_checks", 0)
    failed = c.get("failed", 0)
    if checks > 0 and failed == 0:
        return "validated"
    elif failed > 0:
        return "constraint_fail"
    else:
        return "unvalidated"


print(f"Scanning {RESULTS_ROOT} ...")
by_province_year = defaultdict(lambda: defaultdict(list))
total = 0

for p, ds, fs in os.walk(RESULTS_ROOT):
    for f in sorted(fs):
        if not f.endswith("_oneshot.json"):
            continue
        fp = os.path.join(p, f)
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        total += 1

        meta = d.get("metadata", {})
        civil = is_civil_condition(meta)
        age = is_age_sex(meta)

        if not civil and not age:
            continue

        rel = os.path.relpath(os.path.dirname(fp), RESULTS_ROOT)
        parts = rel.split("/")
        province = parts[0] if len(parts) >= 1 else "unknown"

        try:
            year = int(meta.get("year", 0))
        except (ValueError, TypeError):
            year = 0
        if year == 0 and len(parts) >= 2:
            try:
                year = int(parts[1])
            except ValueError:
                pass

        status = get_status(d)
        table_type = "civil_condition" if civil else "age_sex"
        region = (meta.get("region") or "")[:40]
        n_rows = sum(len(s.get("rows", [])) for s in d.get("data", []))
        source = d.get("source_image", "")

        by_province_year[province][year].append({
            "type": table_type,
            "status": status,
            "region": region,
            "rows": n_rows,
            "file": os.path.basename(fp),
            "source": source,
        })

print(f"Scanned {total} result JSONs\n")

# Print coverage summary
print("=" * 90)
print(f"{'Province':<30} {'Year':<6} {'Civil':<8} {'AgeSex':<8} {'Valid':<8} {'Fail':<8} {'Unval':<8}")
print("=" * 90)

for prov in sorted(by_province_year):
    for year in sorted(by_province_year[prov]):
        items = by_province_year[prov][year]
        n_civil = sum(1 for i in items if i["type"] == "civil_condition")
        n_age = sum(1 for i in items if i["type"] == "age_sex")
        n_val = sum(1 for i in items if i["status"] == "validated")
        n_fail = sum(1 for i in items if i["status"] == "constraint_fail")
        n_unval = sum(1 for i in items if i["status"] == "unvalidated")
        print(f"{prov:<30} {year:<6} {n_civil:<8} {n_age:<8} {n_val:<8} {n_fail:<8} {n_unval:<8}")

# Print constraint failures that could be recovered with MoE
print("\n" + "=" * 90)
print("CONSTRAINT FAILURES (candidates for MoE recovery):")
print("=" * 90)
failures = []
for prov in sorted(by_province_year):
    for year in sorted(by_province_year[prov]):
        for item in by_province_year[prov][year]:
            if item["status"] == "constraint_fail":
                failures.append((prov, year, item))

print(f"\nTotal: {len(failures)} civil/age tables with constraint failures")
for prov, year, item in failures[:50]:
    print(f"  {prov}/{year} | {item['type']:<18} | {item['region']:<30} | {item['rows']} rows | {item['file']}")

if len(failures) > 50:
    print(f"  ... and {len(failures) - 50} more")

# Write source images for failures to a file for MoE rerun
fail_images = os.path.expanduser("~/census_toc/gap_moe_images.txt")
seen = set()
with open(fail_images, "w") as out:
    for prov, year, item in failures:
        src = item["source"]
        if src and src not in seen:
            seen.add(src)
            out.write(src + "\n")
print(f"\nWrote {len(seen)} unique failure source images to {fail_images}")
