"""Create and apply final curation decisions for core_review rows.

This script does two jobs:

1. Initialize an editable decision sheet from core_review_priority.csv.
2. Apply edited decisions to produce final result-json lists for a curated export.

The intended workflow is:

  python3 apply_core_review_decisions.py --init
  # edit core_review_decisions.csv
  python3 apply_core_review_decisions.py --apply

Outputs:
  - core_review_decisions.csv         editable review sheet
  - promote_results.txt               promoted result_json paths
  - reject_results.txt                rejected result_json paths
  - unresolved_results.txt            rows still lacking final decisions
  - final_core_results.txt            core_rescue + promoted core_review result_jsons
  - final_core_summary.json           curation summary
  - run_final_export_on_cluster.sh    exact cluster command for the curated export
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_DECISIONS = {"promote", "reject", "hold"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for value in values:
            fh.write(f"{value}\n")


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def normalize_decision(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"accept", "accepted", "promote_now"}:
        return "promote"
    if text in {"reject_now", "rejected"}:
        return "reject"
    if text in {"review", "manual_review", "manual", "pending", "skip"}:
        return "hold"
    return text


def suggestion_from_priority(bucket: str) -> str:
    bucket = str(bucket or "").strip().lower()
    if bucket == "promote_now":
        return "promote"
    if bucket == "reject_now":
        return "reject"
    return "hold"


def initialize_decision_rows(priority_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    rows = []
    fieldnames = list(priority_rows[0].keys()) if priority_rows else []
    extra_cols = ["suggested_decision", "review_decision", "review_notes", "final_decision"]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in priority_rows:
        suggested = suggestion_from_priority(row.get("priority_bucket", ""))
        review = normalize_decision(row.get("review_decision", ""))
        final_decision = review if review in VALID_DECISIONS else ""
        rows.append(
            {
                **row,
                "suggested_decision": suggested,
                "review_decision": review,
                "review_notes": row.get("review_notes", ""),
                "final_decision": final_decision,
            }
        )
    return rows, fieldnames


def apply_decisions(
    decision_rows: list[dict[str, str]],
    core_rescue_results: list[str],
) -> dict[str, object]:
    promote = []
    reject = []
    hold = []

    decision_counts = {"promote": 0, "reject": 0, "hold": 0, "invalid_or_blank": 0}

    for row in decision_rows:
        explicit = normalize_decision(row.get("review_decision", ""))
        if explicit in VALID_DECISIONS:
            final_decision = explicit
        else:
            final_decision = ""

        row["final_decision"] = final_decision

        result_json = str(row.get("result_json", "")).strip()
        if final_decision == "promote":
            promote.append(result_json)
            decision_counts["promote"] += 1
        elif final_decision == "reject":
            reject.append(result_json)
            decision_counts["reject"] += 1
        elif final_decision == "hold":
            hold.append(result_json)
            decision_counts["hold"] += 1
        else:
            hold.append(result_json)
            decision_counts["invalid_or_blank"] += 1

    promote = dedupe_keep_order(promote)
    reject = dedupe_keep_order(reject)
    hold = dedupe_keep_order(hold)
    final_core = dedupe_keep_order(core_rescue_results + promote)

    return {
        "decision_rows": decision_rows,
        "promote": promote,
        "reject": reject,
        "hold": hold,
        "final_core": final_core,
        "decision_counts": decision_counts,
    }


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def write_cluster_command(path: Path, final_results_path: Path, output_dir_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd ~/SudokuOCR
source ~/.envs/sudokuocr/bin/activate

python3 export_inference_dataset.py \\
  --organized-root ~/census_toc/organized_results \\
  --manifest-jsonl ~/census_toc/organized_manifest.jsonl \\
  --digitized-root ~/census_toc/digitized_tables \\
  --output-dir ~/census_toc/{output_dir_name} \\
  --include-unvalidated \\
  --include-failed \\
  --extra-result-json-list {final_results_path}
"""
    path.write_text(script, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and apply final curation decisions for core_review rows.")
    parser.add_argument("--priority-csv", default="export/local_research_bundle/core_review_priority/core_review_priority.csv")
    parser.add_argument("--decisions-csv", default="export/local_research_bundle/final_core_prep/core_review_decisions.csv")
    parser.add_argument("--core-rescue-results", default="export/local_research_bundle/non_age_rescue_v5/core_rescue_results.txt")
    parser.add_argument("--output-dir", default="export/local_research_bundle/final_core_prep")
    parser.add_argument("--cluster-output-name", default="export_inference_final_core_curated_v1")
    parser.add_argument("--init", action="store_true", help="Write or refresh the editable decisions CSV.")
    parser.add_argument("--apply", action="store_true", help="Apply decisions and emit final result-json lists.")
    args = parser.parse_args()

    if not args.init and not args.apply:
        parser.error("Specify at least one of --init or --apply")

    priority_csv = Path(args.priority_csv).expanduser()
    decisions_csv = Path(args.decisions_csv).expanduser()
    core_rescue_results = Path(args.core_rescue_results).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    priority_rows = read_csv_rows(priority_csv)
    decision_rows, fieldnames = initialize_decision_rows(priority_rows)

    if args.init:
        if decisions_csv.exists():
            existing = {row.get("result_json", ""): row for row in read_csv_rows(decisions_csv)}
            merged = []
            for row in decision_rows:
                previous = existing.get(row.get("result_json", ""))
                if previous:
                    row["review_decision"] = normalize_decision(previous.get("review_decision", ""))
                    row["review_notes"] = previous.get("review_notes", "")
                    row["final_decision"] = row["review_decision"] if row["review_decision"] in VALID_DECISIONS else ""
                merged.append(row)
            decision_rows = merged

        write_csv(decisions_csv, decision_rows, fieldnames)
        print(f"Wrote editable decisions CSV: {decisions_csv}")

    if args.apply:
        if not decisions_csv.exists():
            raise SystemExit(f"Decision CSV not found: {decisions_csv}")

        applied_rows = read_csv_rows(decisions_csv)
        result = apply_decisions(applied_rows, load_lines(core_rescue_results))

        write_csv(decisions_csv, result["decision_rows"], fieldnames)
        write_list(output_dir / "promote_results.txt", result["promote"])
        write_list(output_dir / "reject_results.txt", result["reject"])
        write_list(output_dir / "unresolved_results.txt", result["hold"])
        write_list(output_dir / "final_core_results.txt", result["final_core"])
        write_cluster_command(
            output_dir / "run_final_export_on_cluster.sh",
            output_dir / "final_core_results.txt",
            args.cluster_output_name,
        )

        summary = {
            "priority_csv": str(priority_csv),
            "decisions_csv": str(decisions_csv),
            "core_rescue_results": str(core_rescue_results),
            "counts": {
                "core_rescue_results": len(load_lines(core_rescue_results)),
                "promote": len(result["promote"]),
                "reject": len(result["reject"]),
                "unresolved": len(result["hold"]),
                "final_core_results": len(result["final_core"]),
                **result["decision_counts"],
            },
        }
        (output_dir / "final_core_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("Final core curation summary")
        print("---------------------------")
        print(f"Promote:         {len(result['promote'])}")
        print(f"Reject:          {len(result['reject'])}")
        print(f"Unresolved:      {len(result['hold'])}")
        print(f"Final core list: {len(result['final_core'])}")
        print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
