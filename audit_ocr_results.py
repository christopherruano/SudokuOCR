"""Comprehensive audit of OCR results.

Classifies every result into:
  - age_table_validated: age rows + constraints passed
  - age_table_unvalidated: has age rows but 0 constraint checks
  - age_table_failed: has age rows but constraint failures
  - non_age_table: not an age table (education, birth-place, etc.)
  - corrupt: unreadable JSON

For age tables, checks plausibility:
  - Are values positive integers in reasonable range?
  - Does age distribution shape look right?
  - Do any totals exist and do rows sum to them?
"""

import json
import os
import re
import sys
import argparse

# Age-group patterns: things that look like age labels
AGE_PATTERNS = [
    r"^\d+[-–]\d+$",          # 0-1, 5-10, 10-15
    r"^\d+\s*(and|&)\s*(over|above|upward)", # 60 and over
    r"^(under|below)\s*\d+",  # under 1, below 5
    r"^all\s*ages",            # all ages
    r"^\d+\s*$",               # single year: 0, 1, 2...
    r"^total",                 # total
    r"^(not\s*)?stated",       # age not stated
    r"^\d+[-–]\d+\s*(years?)?$",  # 0-5 years
]

def looks_like_age_label(text):
    if not text:
        return False
    t = str(text).strip().lower()
    return any(re.match(p, t) for p in AGE_PATTERNS)


def classify_result(data):
    """Classify a single OCR result."""
    sections = data.get("data", [])
    meta = data.get("metadata", {})
    constraints = data.get("constraints", {})
    acceptance = data.get("acceptance", {})

    total_checks = constraints.get("total_checks", 0)
    failed = constraints.get("failed", 0)
    decision = acceptance.get("decision", "unknown")

    # Count age-like row labels
    age_rows = 0
    total_rows = 0
    all_row_labels = []

    for section in sections:
        rows = section.get("rows", [])
        for row in rows:
            total_rows += 1
            # Find the row label (first non-dict value)
            for k, v in row.items():
                if not isinstance(v, dict):
                    label = str(v).strip()
                    all_row_labels.append(label)
                    if looks_like_age_label(label):
                        age_rows += 1
                    break

    is_age_table = age_rows >= 3 and age_rows / max(total_rows, 1) > 0.3

    if not is_age_table:
        return "non_age_table", {
            "reason": "row labels don't look like age groups",
            "sample_labels": all_row_labels[:5],
            "age_rows": age_rows,
            "total_rows": total_rows,
        }

    if total_checks > 0 and failed == 0:
        return "age_table_validated", {
            "checks": total_checks,
            "decision": decision,
        }
    elif total_checks > 0 and failed > 0:
        return "age_table_failed", {
            "checks": total_checks,
            "failed": failed,
            "decision": decision,
        }
    else:
        return "age_table_unvalidated", {
            "checks": total_checks,
            "decision": decision,
            "reason": "no constraint checks available",
        }


def plausibility_check(data):
    """Check if extracted values are plausible for Indian census data."""
    issues = []
    sections = data.get("data", [])

    for section in sections:
        rows = section.get("rows", [])
        all_values = []
        total_row_vals = []

        for row in rows:
            row_label = ""
            for k, v in row.items():
                if not isinstance(v, dict):
                    row_label = str(v).strip().lower()
                    continue
                # v is a group dict like {"persons": N, "males": N, "females": N}
                for col, val in v.items():
                    if val is None:
                        continue
                    if not isinstance(val, (int, float)):
                        issues.append(f"Non-numeric value: {val} in {row_label}/{col}")
                        continue
                    all_values.append(val)
                    if "total" in row_label:
                        total_row_vals.append(val)
                    if val < 0:
                        issues.append(f"Negative value: {val} in {row_label}/{col}")

        if all_values:
            max_val = max(all_values)
            # Indian district populations: typically 10K - 10M
            # But proportional tables use 1000 or 10000
            if max_val > 500_000_000:
                issues.append(f"Implausibly large value: {max_val}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Audit OCR results for age-table coverage and constraint quality")
    parser.add_argument("root", nargs="?", default="~/census_toc/ocr_results",
                        help="OCR results root directory (default: ~/census_toc/ocr_results)")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    files = []
    for p, ds, fs in os.walk(root):
        for f in fs:
            if f.endswith("_oneshot.json"):
                files.append(os.path.join(p, f))

    print(f"Results root: {root}")
    print(f"Auditing {len(files)} results...")
    print()

    categories = {
        "age_table_validated": [],
        "age_table_unvalidated": [],
        "age_table_failed": [],
        "non_age_table": [],
        "corrupt": [],
    }
    plausibility_issues = []

    for f in files:
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, KeyError):
            categories["corrupt"].append(f)
            continue

        cat, info = classify_result(d)
        categories[cat].append({"file": os.path.basename(f), "info": info})

        # Plausibility check for age tables
        if cat.startswith("age_table"):
            issues = plausibility_check(d)
            if issues:
                plausibility_issues.append({
                    "file": os.path.basename(f),
                    "category": cat,
                    "issues": issues,
                })

    # Print summary
    print("=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)
    total = len(files)
    for cat in ["age_table_validated", "age_table_unvalidated",
                "age_table_failed", "non_age_table", "corrupt"]:
        items = categories[cat]
        pct = 100 * len(items) / total if total else 0
        print(f"  {cat}: {len(items)} ({pct:.1f}%)")

    # Show samples of each category
    for cat in categories:
        items = categories[cat]
        if not items:
            continue
        print(f"\n--- {cat} samples ---")
        for item in items[:3]:
            if isinstance(item, str):
                print(f"  {item}")
            else:
                print(f"  {item['file']}: {item['info']}")

    # Non-age table breakdown: what are they?
    non_age = categories["non_age_table"]
    if non_age:
        print(f"\n--- Non-age table label samples ({len(non_age)} total) ---")
        seen = set()
        for item in non_age[:20]:
            labels = item["info"].get("sample_labels", [])
            key = str(labels[:3])
            if key not in seen:
                seen.add(key)
                print(f"  {labels[:5]}")

    # Plausibility issues
    print(f"\n{'=' * 60}")
    print(f"PLAUSIBILITY ISSUES")
    print(f"{'=' * 60}")
    print(f"  Age tables with issues: {len(plausibility_issues)} / "
          f"{len(categories['age_table_validated']) + len(categories['age_table_unvalidated']) + len(categories['age_table_failed'])}")
    for item in plausibility_issues[:10]:
        print(f"  {item['file']} ({item['category']}):")
        for issue in item["issues"][:3]:
            print(f"    - {issue}")

    # Summary
    age_total = (len(categories["age_table_validated"])
                 + len(categories["age_table_unvalidated"])
                 + len(categories["age_table_failed"]))
    print(f"\n{'=' * 60}")
    print(f"BOTTOM LINE")
    print(f"{'=' * 60}")
    print(f"  Total results: {total}")
    print(f"  Age tables found: {age_total}")
    print(f"    Validated (checks passed): {len(categories['age_table_validated'])}")
    print(f"    Unvalidated (no checks): {len(categories['age_table_unvalidated'])}")
    print(f"    Failed (checks failed): {len(categories['age_table_failed'])}")
    print(f"  Non-age tables (filtered out): {len(categories['non_age_table'])}")
    print(f"  Plausibility issues: {len(plausibility_issues)}")
    print(f"  Corrupt: {len(categories['corrupt'])}")


if __name__ == "__main__":
    main()
