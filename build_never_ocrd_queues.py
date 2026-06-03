"""Build prioritized OCR queues from never_ocrd age-recall audit candidates.

Inputs:
  - age_recall_audit/never_ocrd_candidates.csv

Outputs:
  - queue_summary.json
  - core_auto.csv / .txt
  - core_review.csv / .txt
  - tail_auto.csv / .txt
  - tail_review.csv / .txt
  - defer.csv / .txt

The goal is to decide what to OCR next from the never-OCR'd bucket:
  - core_auto: explicit classic primary age tables; safe to OCR now
  - core_review: likely classic primary, but ambiguous; inspect before OCR
  - tail_auto: structured secondary age tables worth OCRing next
  - tail_review: broader age tail; inspect before OCR
  - defer: too noisy / too weak right now
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PRIMARY_TOPICS = {
    "age_sex_population",
    "age_sex_civil_condition",
    "age_sex_marital_status",
}

TAIL_AUTO_TOPICS = {
    "age_distribution",
    "age_by_religion",
    "age_civil_condition",
}

STRONG_CORE_TITLE_PATTERNS = [
    re.compile(r"\bage[\W_]*sex[\W_]*and[\W_]*civil[\W_]*condition\b", re.I),
    re.compile(r"\bage[\W_]*sex[\W_]*and[\W_]*marital", re.I),
    re.compile(r"\bdistribution\s+of\s+the\s+population\s+by\s+(?:civil\s+condition[\W_]*)?age(?:[\W_]*|\s+and\s+)sex\b", re.I),
    re.compile(r"\bpopulation\b.*\bage\b.*\bsex\b", re.I),
    re.compile(r"\bage\b.*\bsex\b.*\bpopulation\b", re.I),
    re.compile(r"\bage\s+returns?\b", re.I),
    re.compile(r"\b(?:imperial\s+)?table\s+(?:vii|vil|vi1|v1i|7)\b.*\bage\b.*\bsex\b", re.I),
]

WEAK_CORE_TITLE_PATTERNS = [
    re.compile(r"\b(?:imperial\s+)?table\s+(?:vii|vil|vi1|v1i|7)\b", re.I),
    re.compile(r"\bdistribution\s+of\s+the\s+population\b", re.I),
]

TAIL_AUTO_TITLE_PATTERNS = [
    re.compile(r"\bage[-\s]*distribution\b", re.I),
    re.compile(r"\bage\b.*\breligion\b", re.I),
    re.compile(r"\bcivil\s+condition\b.*\bage\b", re.I),
    re.compile(r"\bage\b.*\bcivil\s+condition\b", re.I),
]

BLOCK_PATTERNS = [
    re.compile(r"\bbirth[-\s]*place\b", re.I),
    re.compile(r"\blanguage\b", re.I),
    re.compile(r"\bvaccination\b", re.I),
    re.compile(r"\bsmall[-\s]*pox\b", re.I),
    re.compile(r"\bdeaf\b", re.I),
    re.compile(r"\bleper\b", re.I),
    re.compile(r"\bblind\b", re.I),
    re.compile(r"\blivelihood\b", re.I),
    re.compile(r"\blabou?r\b", re.I),
    re.compile(r"\boccupations?\b", re.I),
    re.compile(r"\beducation\b", re.I),
    re.compile(r"\bliteracy\b", re.I),
    re.compile(r"\bcaste\b", re.I),
    re.compile(r"\btribe\b", re.I),
    re.compile(r"\brace\b", re.I),
    re.compile(r"\bcorrelation\b", re.I),
    re.compile(r"\bfertility\b", re.I),
    re.compile(r"\bhouses\b", re.I),
]

REVIEW_HINT_PATTERNS = [
    re.compile(r"\bby\s+religion\b", re.I),
    re.compile(r"\ball\s+religions?\b", re.I),
    re.compile(r"\bpart\s+[a-z0-9()\-]+\b", re.I),
    re.compile(r"\bcontinued\b", re.I),
    re.compile(r"\bconcluded\b", re.I),
    re.compile(r"\bstatement\s+no\b", re.I),
]


def normalize_text(value):
    text = str(value or "").strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(value):
    text = normalize_text(value).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def as_int(value):
    try:
        return int(str(value).strip() or "0")
    except Exception:
        return 0


def text_matches_any(text, patterns):
    text = str(text or "")
    return any(pattern.search(text) for pattern in patterns)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path, rows, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row[key]}\n")


def decide_queue(row):
    topic = row.get("canonical_topic") or ""
    title = normalize_text(row.get("title") or row.get("table_dir_name") or "")
    query_tags = normalize_token(row.get("query_tags") or "")
    table_dir_name = normalize_text(row.get("table_dir_name") or "")
    evidence_score = as_int(row.get("evidence_score"))
    dir_name_match = as_int(row.get("dir_name_match"))
    classic_primary = as_int(row.get("classic_primary_candidate"))
    tail_candidate = as_int(row.get("age_structured_tail_candidate"))

    strong_core_title = text_matches_any(title, STRONG_CORE_TITLE_PATTERNS)
    weak_core_title = text_matches_any(title, WEAK_CORE_TITLE_PATTERNS)
    tail_auto_title = text_matches_any(title, TAIL_AUTO_TITLE_PATTERNS)
    blocked = text_matches_any(title, BLOCK_PATTERNS) or text_matches_any(table_dir_name, BLOCK_PATTERNS)
    review_hint = text_matches_any(title, REVIEW_HINT_PATTERNS)

    reasons = []

    if classic_primary:
        if strong_core_title and not blocked and evidence_score >= 7:
            reasons.extend(["strong_core_title", f"evidence_score:{evidence_score}"])
            if dir_name_match:
                reasons.append("dir_name_match")
            if review_hint:
                return "core_review", reasons + ["review_hint"]
            return "core_auto", reasons

        if (
            not blocked
            and evidence_score >= 6
            and (
                weak_core_title
                or dir_name_match
                or topic in PRIMARY_TOPICS
            )
        ):
            reasons.extend([f"evidence_score:{evidence_score}"])
            if weak_core_title:
                reasons.append("weak_core_title")
            if dir_name_match:
                reasons.append("dir_name_match")
            if topic in PRIMARY_TOPICS:
                reasons.append(f"primary_topic:{topic}")
            return "core_review", reasons

    if tail_candidate or topic in TAIL_AUTO_TOPICS:
        if (
            not blocked
            and evidence_score >= 4
            and (
                topic in TAIL_AUTO_TOPICS
                or tail_auto_title
                or "religion" in query_tags
                or "civil condition" in query_tags
            )
        ):
            reasons.extend([f"evidence_score:{evidence_score}"])
            if topic in TAIL_AUTO_TOPICS:
                reasons.append(f"tail_topic:{topic}")
            if tail_auto_title:
                reasons.append("tail_auto_title")
            return "tail_auto", reasons

        if evidence_score >= 3:
            reasons.extend([f"evidence_score:{evidence_score}"])
            if blocked:
                reasons.append("blocked_for_auto")
            return "tail_review", reasons

    return "defer", [f"evidence_score:{evidence_score}"] + (["blocked"] if blocked else [])


def main():
    parser = argparse.ArgumentParser(description="Build OCR queues from never_ocrd age-recall candidates")
    parser.add_argument("--input-csv", default="~/census_toc/age_recall_audit/never_ocrd_candidates.csv")
    parser.add_argument("--output-dir", default="~/census_toc/never_ocrd_queues")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            queue_bucket, reasons = decide_queue(row)
            rows.append({
                **row,
                "queue_bucket": queue_bucket,
                "queue_reasons": " | ".join(reasons),
            })

    rows.sort(key=lambda r: (-as_int(r["evidence_score"]), r.get("province") or "", r.get("year") or "", r.get("image_path") or ""))

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "all.csv", rows, fieldnames)

    buckets = {}
    for bucket in ["core_auto", "core_review", "tail_auto", "tail_review", "defer"]:
        bucket_rows = [row for row in rows if row["queue_bucket"] == bucket]
        buckets[bucket] = bucket_rows
        write_csv(output_dir / f"{bucket}.csv", bucket_rows, fieldnames)
        write_list(output_dir / f"{bucket}.txt", bucket_rows, "image_path")

    summary = {
        "input_rows": len(rows),
        "core_auto": len(buckets["core_auto"]),
        "core_review": len(buckets["core_review"]),
        "tail_auto": len(buckets["tail_auto"]),
        "tail_review": len(buckets["tail_review"]),
        "defer": len(buckets["defer"]),
        "top_topics_core_auto": Counter(row["canonical_topic"] for row in buckets["core_auto"]).most_common(20),
        "top_topics_core_review": Counter(row["canonical_topic"] for row in buckets["core_review"]).most_common(20),
        "top_topics_tail_auto": Counter(row["canonical_topic"] for row in buckets["tail_auto"]).most_common(20),
        "top_topics_tail_review": Counter(row["canonical_topic"] for row in buckets["tail_review"]).most_common(20),
    }
    with (output_dir / "queue_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Never-OCR queue summary")
    print("-----------------------")
    print(f"Input rows:   {len(rows)}")
    print(f"Core auto:    {len(buckets['core_auto'])}")
    print(f"Core review:  {len(buckets['core_review'])}")
    print(f"Tail auto:    {len(buckets['tail_auto'])}")
    print(f"Tail review:  {len(buckets['tail_review'])}")
    print(f"Defer:        {len(buckets['defer'])}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
