#!/usr/bin/env python3
"""Materialize assistant-reviewed failing-table action views."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("export_final/browse/failing_tables")
ASSESSMENT_CSV = ROOT / "assistant_assessment.csv"
OUT_ROOT = ROOT / "assistant_reviewed"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build() -> None:
    rows = load_rows(ASSESSMENT_CSV)
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    actions = ["salvage", "keep_with_caution", "drop"]
    summary: dict[str, object] = {
        "total": len(rows),
        "by_action": Counter(r["recommended_action"] for r in rows),
        "by_dataset": defaultdict(Counter),
        "by_action_dataset": defaultdict(Counter),
        "top_examples": {},
    }

    for row in rows:
        summary["by_dataset"][row["dataset"]][row["effective_quality_status"]] += 1
        summary["by_action_dataset"][row["recommended_action"]][row["dataset"]] += 1

    for action in actions:
        action_rows = [r for r in rows if r["recommended_action"] == action]
        action_dir = OUT_ROOT / action
        action_dir.mkdir(parents=True, exist_ok=True)
        write_csv(action_dir / f"{action}.csv", action_rows, fieldnames)

        for dataset in sorted({r["dataset"] for r in action_rows}):
            ds_rows = [r for r in action_rows if r["dataset"] == dataset]
            write_csv(action_dir / f"{dataset}.csv", ds_rows, fieldnames)
            links_dir = action_dir / "preview_links" / dataset
            links_dir.mkdir(parents=True, exist_ok=True)
            for row in ds_rows:
                src = ROOT / row["folder"]
                link_name = Path(row["folder"]).name
                safe_symlink(src, links_dir / link_name)
            (action_dir / f"{dataset}_table_ids.txt").write_text(
                "\n".join(r["table_id"] for r in ds_rows) + ("\n" if ds_rows else ""),
                encoding="utf-8",
            )

        top_rows = sorted(
            action_rows,
            key=lambda r: (
                {"high": 0, "medium": 1, "low": 2}[r["severity"]],
                -int(float(r["failed_checks"] or 0)),
                r["dataset"],
                r["province"],
                r["title"],
            ),
        )[:25]
        write_csv(action_dir / "top_examples.csv", top_rows, fieldnames)
        summary["top_examples"][action] = [
            {
                "table_id": r["table_id"],
                "dataset": r["dataset"],
                "province": r["province"],
                "year": r["year"],
                "failed_checks": r["failed_checks"],
                "title": r["title"],
                "primary_issue": r["primary_issue"],
            }
            for r in top_rows[:10]
        ]

    summary["by_action"] = dict(summary["by_action"])
    summary["by_dataset"] = {k: dict(v) for k, v in summary["by_dataset"].items()}
    summary["by_action_dataset"] = {
        k: dict(v) for k, v in summary["by_action_dataset"].items()
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_lines = [
        "# Assistant-Reviewed Failing Tables",
        "",
        "This directory applies the assistant's review to the failing kept-table pack.",
        "",
        "Buckets:",
        "- `salvage/`: on-topic tables that likely need family-specific repair, multipage assembly, or better axis interpretation",
        "- `keep_with_caution/`: near-acceptable tables with only minor residual issues",
        "- `drop/`: tables that should not be treated as part of the age-table project deliverable",
        "",
        "Within each bucket:",
        "- `<bucket>.csv`: full list",
        "- `<dataset>.csv`: layer-specific list",
        "- `preview_links/<dataset>/`: symlinks to the corresponding preview folders",
        "- `top_examples.csv`: highest-priority examples in that bucket",
        "",
        "Counts:",
        f"- salvage: {sum(1 for r in rows if r['recommended_action'] == 'salvage')}",
        f"- keep_with_caution: {sum(1 for r in rows if r['recommended_action'] == 'keep_with_caution')}",
        f"- drop: {sum(1 for r in rows if r['recommended_action'] == 'drop')}",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
