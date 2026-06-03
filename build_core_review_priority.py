"""Prioritize non-age rescue core_review rows for manual promotion.

Inputs:
  - non_age_rescue_v5/core_review.csv

Outputs:
  - core_review_priority.csv
  - core_review_promote_now.txt
  - core_review_reject_now.txt
  - core_review_manual.csv

This is a triage helper, not a final classifier.
"""

import argparse
import csv
import re
from pathlib import Path


PROMOTE_PATTERNS = [
    re.compile(r"\bage[\W_]*sex[\W_]*and[\W_]*civil[\W_]*condition\b", re.I),
    re.compile(r"\bdistribution\s+of\s+the\s+population\s+by\s+(?:civil\s+condition[\W_]*)?age(?:[\W_]*|\s+and\s+)sex\b", re.I),
    re.compile(r"\bage\s+returns?\b", re.I),
    re.compile(r"\b(?:imperial\s+)?table\s+(?:vii|vil|vi1|v1i|7)\b", re.I),
]

REJECT_PATTERNS = [
    re.compile(r"\bbirth[-\s]*place\b", re.I),
    re.compile(r"\bplace[-\s]*contd\b", re.I),
    re.compile(r"\blanguage\b", re.I),
    re.compile(r"\bdeaf\b", re.I),
    re.compile(r"\bleper\b", re.I),
    re.compile(r"\bblind\b", re.I),
]


def normalize_text(value):
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_matches_any(text, patterns):
    text = str(text or "")
    return any(pattern.search(text) for pattern in patterns)


def as_int(value):
    try:
        return int(str(value).strip() or "0")
    except Exception:
        return 0


def decide(row):
    title = normalize_text(row.get("title_from_payload") or row.get("title") or "")
    reasons = set(filter(None, (row.get("rescue_reasons") or "").split(" | ")))
    evidence_score = as_int(row.get("rescue_score"))

    if text_matches_any(title, REJECT_PATTERNS):
        return "reject_now", ["reject_pattern"]

    promote_signal = text_matches_any(title, PROMOTE_PATTERNS)
    has_core_title = "core_title_signal" in reasons
    has_primary_topic = any(reason.startswith("primary_topic:") for reason in reasons)
    has_dir_match = "dir_name_primary_support" in reasons or "dir_name_match" in reasons
    has_age_structure = "age_rows_structure" in reasons or "age_columns_structure" in reasons

    if promote_signal and has_age_structure and (has_primary_topic or has_dir_match or has_core_title) and evidence_score >= 8:
        return "promote_now", ["strong_primary_title", f"evidence_score:{evidence_score}"]

    return "manual_review", [f"evidence_score:{evidence_score}"]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row['result_json']}\n")


def main():
    parser = argparse.ArgumentParser(description="Prioritize core_review rows for manual promotion")
    parser.add_argument("--input-csv", default="export/local_research_bundle/non_age_rescue_v5/core_review.csv")
    parser.add_argument("--output-dir", default="export/local_research_bundle/core_review_priority")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bucket, reasons = decide(row)
            rows.append({
                **row,
                "priority_bucket": bucket,
                "priority_reasons": " | ".join(reasons),
            })

    rows.sort(key=lambda r: (-as_int(r.get("rescue_score")), r.get("province") or "", r.get("year") or "", r.get("title_from_payload") or ""))

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "core_review_priority.csv", rows, fieldnames)

    promote = [row for row in rows if row["priority_bucket"] == "promote_now"]
    reject = [row for row in rows if row["priority_bucket"] == "reject_now"]
    manual = [row for row in rows if row["priority_bucket"] == "manual_review"]

    write_csv(output_dir / "core_review_manual.csv", manual, fieldnames)
    write_list(output_dir / "core_review_promote_now.txt", promote)
    write_list(output_dir / "core_review_reject_now.txt", reject)

    print("Core-review priority summary")
    print("----------------------------")
    print(f"Input rows:      {len(rows)}")
    print(f"Promote now:     {len(promote)}")
    print(f"Reject now:      {len(reject)}")
    print(f"Manual review:   {len(manual)}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
