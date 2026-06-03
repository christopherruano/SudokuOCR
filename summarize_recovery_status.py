"""Summarize current corpus, rescue buckets, and never-OCR queues.

This is meant to answer:
  - what is already in the inference-ready corpus?
  - what can be recovered from existing OCR without more API spend?
  - what remains worth OCRing when budget returns?

Outputs:
  - recovery_status.json
  - recovery_status.md
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_ocr_results import classify_result


def read_json(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_source_catalog(path):
    path = Path(path).expanduser()
    summary = {
        "catalog_rows": 0,
        "kept_for_inference": 0,
        "table_family_top": [],
        "quality_status_top": [],
        "effective_quality_status_top": [],
    }
    if not path.exists():
        return summary

    family = Counter()
    quality = Counter()
    eff_quality = Counter()
    kept = 0
    rows = 0

    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            keep = str(row.get("keep_for_inference") or "").strip().lower() in {"1", "true", "yes"}
            if keep:
                kept += 1
            family[row.get("table_family") or "unknown"] += 1
            quality[row.get("quality_status") or "unknown"] += 1
            eff_quality[row.get("effective_quality_status") or "unknown"] += 1

    summary["catalog_rows"] = rows
    summary["kept_for_inference"] = kept
    summary["table_family_top"] = family.most_common(20)
    summary["quality_status_top"] = quality.most_common(20)
    summary["effective_quality_status_top"] = eff_quality.most_common(20)
    return summary


def summarize_results_root(root):
    root = Path(root).expanduser()
    summary = {
        "root": str(root),
        "json_files": 0,
        "age_table_validated": 0,
        "age_table_unvalidated": 0,
        "age_table_failed": 0,
        "non_age_table": 0,
        "corrupt": 0,
    }
    if not root.exists():
        return summary

    for path in root.rglob("*_oneshot.json"):
        summary["json_files"] += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            category, _ = classify_result(data)
        except Exception:
            category = "corrupt"
        summary[category] = summary.get(category, 0) + 1
    return summary


def write_markdown(path, payload):
    lines = []
    lines.append("# Recovery Status")
    lines.append("")

    src = payload["source_catalog"]
    lines.append("## Current Corpus")
    lines.append(f"- source catalog rows: {src['catalog_rows']}")
    lines.append(f"- kept for inference: {src['kept_for_inference']}")
    lines.append("- top families:")
    for family, count in src["table_family_top"][:10]:
        lines.append(f"  - {family}: {count}")
    lines.append("")

    rescue = payload["non_age_rescue"]
    if rescue:
        lines.append("## Existing OCR Rescue")
        for key in ["candidates_scanned", "core_rescue", "core_review", "tail_rescue", "still_non_age"]:
            if key in rescue:
                lines.append(f"- {key}: {rescue[key]}")
        lines.append("")

    queues = payload["never_ocrd_queues"]
    if queues:
        lines.append("## Never-OCR Queue")
        for key in ["input_rows", "core_auto", "core_review", "tail_auto", "tail_review", "defer"]:
            if key in queues:
                lines.append(f"- {key}: {queues[key]}")
        lines.append("")

    for result_summary in payload["results_roots"]:
        lines.append(f"## Results Root: {result_summary['root']}")
        for key in ["json_files", "age_table_validated", "age_table_unvalidated", "age_table_failed", "non_age_table", "corrupt"]:
            lines.append(f"- {key}: {result_summary.get(key, 0)}")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("- merge `core_rescue` first")
    lines.append("- review `core_review` before merge")
    lines.append("- treat `tail_rescue` as a separate secondary layer")
    lines.append("- if OCR budget returns, run `core_auto` before any other never-OCR queue")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize current corpus and recovery status")
    parser.add_argument("--source-catalog", default="~/census_toc/export_inference_rich/source_catalog.csv")
    parser.add_argument("--non-age-rescue-json", default="~/census_toc/non_age_rescue_v5/rescue_summary.json")
    parser.add_argument("--never-ocrd-queue-json", default="~/census_toc/never_ocrd_queues/queue_summary.json")
    parser.add_argument("--results-root", action="append", default=[],
                        help="Optional results roots to classify (can be passed multiple times)")
    parser.add_argument("--output-dir", default="~/census_toc/recovery_status")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_catalog": summarize_source_catalog(args.source_catalog),
        "non_age_rescue": read_json(args.non_age_rescue_json),
        "never_ocrd_queues": read_json(args.never_ocrd_queue_json),
        "results_roots": [summarize_results_root(root) for root in args.results_root],
    }

    (output_dir / "recovery_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(output_dir / "recovery_status.md", payload)

    print(f"Wrote: {output_dir / 'recovery_status.json'}")
    print(f"Wrote: {output_dir / 'recovery_status.md'}")


if __name__ == "__main__":
    main()
