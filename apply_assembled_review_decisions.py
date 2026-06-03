#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_BASE = Path("export/final_project_exports")

SUGGESTIONS = {
    "assembled::ce6c76e2c04a4cd5": {
        "suggested_decision": "drop",
        "suggested_reason": "Population-classification material, not a clean age/civil-condition assembled table.",
    },
    "assembled::6e33664fc503ca86": {
        "suggested_decision": "hold",
        "suggested_reason": "Could be useful, but title/context are too noisy to trust without checking the actual pages.",
    },
    "assembled::40fad1546810a752": {
        "suggested_decision": "hold",
        "suggested_reason": "Likely valid infirmity-by-age material, but semantically narrower than the main age tables.",
    },
    "assembled::fcd4fa6a59de37e9": {
        "suggested_decision": "drop",
        "suggested_reason": "Very weak improvement and heterogeneous member pages.",
    },
    "assembled::111bdc99fff26056": {
        "suggested_decision": "drop",
        "suggested_reason": "Looks more like literacy/caste material than a clean assembled age-by-religion table.",
    },
    "assembled::89341ee051bc78fe": {
        "suggested_decision": "drop",
        "suggested_reason": "Member pages look like infirmities/literacy material, not a clean age-sex table.",
    },
    "assembled::b3ac5174845a875f": {
        "suggested_decision": "drop",
        "suggested_reason": "Population-density material, not appropriate for the final assembled age layer.",
    },
}


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def init_review_sheet(base: Path) -> Path:
    overlays = base / "overlays"
    suspicious_path = overlays / "assembled_suspicious_groups.csv"
    review_path = overlays / "assembled_review_decisions.csv"
    suspicious_rows = read_csv_rows(suspicious_path)
    existing = {}
    if review_path.exists():
        for row in read_csv_rows(review_path):
            existing[row.get("table_id") or ""] = row

    out_rows = []
    for row in suspicious_rows:
        table_id = row["table_id"]
        prev = existing.get(table_id, {})
        sug = SUGGESTIONS.get(table_id, {})
        out_rows.append(
            {
                "table_id": table_id,
                "table_family": row.get("table_family", ""),
                "title": row.get("title", ""),
                "compiled_member_count": row.get("compiled_member_count", ""),
                "assembled_failure_improvement": row.get("assembled_failure_improvement", ""),
                "suggested_decision": sug.get("suggested_decision", ""),
                "suggested_reason": sug.get("suggested_reason", ""),
                "review_decision": prev.get("review_decision", ""),
                "review_notes": prev.get("review_notes", ""),
                "compiled_member_titles": row.get("compiled_member_titles", ""),
                "compiled_member_queue_titles": row.get("compiled_member_queue_titles", ""),
            }
        )

    write_csv(
        review_path,
        out_rows,
        [
            "table_id",
            "table_family",
            "title",
            "compiled_member_count",
            "assembled_failure_improvement",
            "suggested_decision",
            "suggested_reason",
            "review_decision",
            "review_notes",
            "compiled_member_titles",
            "compiled_member_queue_titles",
        ],
    )
    return review_path


def apply_review_sheet(base: Path) -> dict[str, object]:
    overlays = base / "overlays"
    assembled_catalog = base / "assembled_strong_v2" / "source_catalog.csv"
    decisions_path = overlays / "assembled_review_decisions.csv"

    catalog_rows = [row for row in read_csv_rows(assembled_catalog) if truthy(row.get("keep_for_inference"))]
    decisions = {row["table_id"]: row for row in read_csv_rows(decisions_path)}

    all_rows = []
    keep_ids: list[str] = []
    drop_ids: list[str] = []
    hold_ids: list[str] = []
    unresolved_ids: list[str] = []

    for row in catalog_rows:
        table_id = row.get("table_id") or ""
        dec = decisions.get(table_id)
        if dec:
            review_decision = (dec.get("review_decision") or "").strip().lower()
            if review_decision not in {"keep", "drop", "hold"}:
                final_decision = ""
                unresolved_ids.append(table_id)
                decision_source = "manual_required"
            else:
                final_decision = review_decision
                decision_source = "manual_review"
        else:
            final_decision = "keep"
            decision_source = "default_keep"

        if final_decision == "keep":
            keep_ids.append(table_id)
        elif final_decision == "drop":
            drop_ids.append(table_id)
        elif final_decision == "hold":
            hold_ids.append(table_id)

        all_rows.append(
            {
                "table_id": table_id,
                "table_family": row.get("table_family", ""),
                "title": row.get("title", ""),
                "compiled_member_count": row.get("compiled_member_count", ""),
                "assembled_failure_improvement": row.get("assembled_failure_improvement", ""),
                "final_decision": final_decision,
                "decision_source": decision_source,
                "review_notes": (dec or {}).get("review_notes", ""),
            }
        )

    all_rows.sort(key=lambda r: (r["final_decision"] or "zzzz", r["table_family"], r["title"]))
    write_csv(
        overlays / "assembled_group_decisions_all.csv",
        all_rows,
        [
            "table_id",
            "table_family",
            "title",
            "compiled_member_count",
            "assembled_failure_improvement",
            "final_decision",
            "decision_source",
            "review_notes",
        ],
    )

    for name, values in [
        ("assembled_keep_table_ids.txt", keep_ids),
        ("assembled_drop_table_ids.txt", drop_ids),
        ("assembled_hold_table_ids.txt", hold_ids),
        ("assembled_unresolved_table_ids.txt", unresolved_ids),
    ]:
        (overlays / name).write_text("".join(f"{value}\n" for value in values), encoding="utf-8")

    summary = {
        "assembled_groups_total": len(catalog_rows),
        "default_keep_groups": len([r for r in all_rows if r["decision_source"] == "default_keep"]),
        "manual_review_groups": len([r for r in all_rows if r["decision_source"] == "manual_review"]),
        "keep_groups": len(keep_ids),
        "drop_groups": len(drop_ids),
        "hold_groups": len(hold_ids),
        "unresolved_groups": len(unresolved_ids),
    }
    (overlays / "assembled_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize/apply small manual reviews for suspicious assembled groups.")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Final project export root")
    parser.add_argument("--init", action="store_true", help="Create or refresh the 7-row review sheet")
    parser.add_argument("--apply", action="store_true", help="Apply the review sheet into keep/drop/hold filters")
    args = parser.parse_args()

    base = Path(args.base)
    if not args.init and not args.apply:
        parser.error("pass --init and/or --apply")

    if args.init:
        review_path = init_review_sheet(base)
        print(f"Wrote: {review_path}")
    if args.apply:
        summary = apply_review_sheet(base)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
