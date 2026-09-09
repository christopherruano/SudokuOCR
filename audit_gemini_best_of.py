#!/usr/bin/env python3
"""Audit the best available Gemini OCR result for each priority image.

The gap-fill runs intentionally write into separate result roots so retries do
not overwrite earlier outputs. This script compares those roots image-by-image,
selects the best available result, and writes an auditable CSV plus a manifest
of remaining non-accepted images.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_gemini_prime_gapfill import (
    CLASS_MANIFESTS,
    PROJECT_ROOT,
    WORKLIST_DIR,
    audit_image,
    page_sort_key,
    read_manifest,
)


STATUS_RANK = {
    "accepted_high_confidence": 0,
    "accepted_with_risk": 1,
    "has_failures": 2,
    "corrupt": 3,
    "missing": 4,
}


DEFAULT_ROOT_GLOB = "gemini_no_gap_ocr_2026-06-16_v2*"


def discover_roots(pattern: str) -> list[Path]:
    roots = [
        p.resolve()
        for p in (PROJECT_ROOT / "export_final").glob(pattern)
        if p.is_dir()
    ]
    roots.sort(key=lambda p: (p.name != "gemini_no_gap_ocr_2026-06-16_v2_best", p.name))
    return roots


def parse_roots(raw: str, pattern: str) -> list[Path]:
    if raw.strip():
        roots = [Path(x.strip()).expanduser().resolve() for x in raw.split(",") if x.strip()]
    else:
        roots = discover_roots(pattern)
    if not roots:
        raise SystemExit("No result roots found")
    return roots


def read_json_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    constraints = data.get("constraints") or {}
    acceptance = data.get("acceptance") or {}
    signals = acceptance.get("signals") or {}
    failures = constraints.get("failed")
    if failures in (None, 0) and isinstance(constraints.get("failures"), list):
        failures = len(constraints.get("failures") or [])
    return {
        "decision": acceptance.get("decision", ""),
        "trust_score": acceptance.get("trust_score", ""),
        "reasons": "|".join(str(x) for x in acceptance.get("reasons", []) or []),
        "rows_extracted": signals.get("rows_extracted", ""),
        "total_checks": constraints.get("total_checks", ""),
        "failures": failures if failures is not None else "",
    }


def best_candidate(image: Path, roots: list[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for root in roots:
        status, failures, checks, result_json = audit_image(image, root)
        info = read_json_info(result_json)
        candidates.append(
            {
                "root": root,
                "root_name": root.name,
                "status": status,
                "failures": failures,
                "checks": checks,
                "result_json": result_json,
                "rows_extracted": info.get("rows_extracted", ""),
                "trust_score": info.get("trust_score", ""),
                "reasons": info.get("reasons", ""),
            }
        )
    candidates.sort(
        key=lambda x: (
            STATUS_RANK.get(str(x["status"]), 99),
            int(x["failures"] or 0),
            -int(x["checks"] or 0),
            str(x["root_name"]),
        )
    )
    best = dict(candidates[0])
    best["candidate_statuses"] = ";".join(
        f"{c['root_name']}:{c['status']}:{c['failures']}:{c['checks']}"
        for c in candidates
    )
    return best


def build_rows(classes: list[str], roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for cls in classes:
        if cls not in CLASS_MANIFESTS:
            raise SystemExit(f"Unknown class: {cls}")
        for image in read_manifest(CLASS_MANIFESTS[cls]):
            if image in seen:
                continue
            seen.add(image)
            best = best_candidate(image, roots)
            rows.append(
                {
                    "priority_class": cls,
                    "image_path": str(image),
                    "source": page_sort_key(image)[0],
                    "page": page_sort_key(image)[1],
                    "best_status": best["status"],
                    "best_failures": best["failures"],
                    "best_checks": best["checks"],
                    "best_root": best["root_name"],
                    "best_result_json": str(best["result_json"]),
                    "rows_extracted": best["rows_extracted"],
                    "trust_score": best["trust_score"],
                    "reasons": best["reasons"],
                    "candidate_statuses": best["candidate_statuses"],
                }
            )
    rows.sort(
        key=lambda r: (
            STATUS_RANK.get(str(r["best_status"]), 99),
            int(r["best_failures"] or 0),
            r["priority_class"],
            r["source"],
            int(r["page"] or -1),
            r["image_path"],
        )
    )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    audit_csv = output_prefix.with_suffix(".csv")
    remaining_csv = output_prefix.with_name(output_prefix.name + "_remaining_failures.csv")
    remaining_manifest = output_prefix.with_name(output_prefix.name + "_remaining_failures.txt")
    accepted_manifest = output_prefix.with_name(output_prefix.name + "_accepted.txt")
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")

    fieldnames = [
        "priority_class",
        "image_path",
        "source",
        "page",
        "best_status",
        "best_failures",
        "best_checks",
        "best_root",
        "best_result_json",
        "rows_extracted",
        "trust_score",
        "reasons",
        "candidate_statuses",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    remaining = [
        r for r in rows if r["best_status"] not in ("accepted_high_confidence", "accepted_with_risk")
    ]
    accepted = [
        r for r in rows if r["best_status"] in ("accepted_high_confidence", "accepted_with_risk")
    ]
    with remaining_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remaining)
    with remaining_manifest.open("w", encoding="utf-8") as fh:
        for row in remaining:
            fh.write(row["image_path"] + "\n")
    with accepted_manifest.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(row["image_path"] + "\n")

    by_status = Counter(str(r["best_status"]) for r in rows)
    by_class_status: dict[str, dict[str, int]] = defaultdict(dict)
    for cls in sorted({str(r["priority_class"]) for r in rows}):
        c = Counter(str(r["best_status"]) for r in rows if r["priority_class"] == cls)
        by_class_status[cls] = dict(c)
    summary = {
        "total": len(rows),
        "accepted": len(accepted),
        "remaining": len(remaining),
        "by_status": dict(by_status),
        "by_class_status": dict(by_class_status),
        "audit_csv": str(audit_csv),
        "remaining_csv": str(remaining_csv),
        "remaining_manifest": str(remaining_manifest),
        "accepted_manifest": str(accepted_manifest),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default="main,derivative")
    parser.add_argument("--roots", default="", help="Comma-separated result roots; auto-discovers by default")
    parser.add_argument("--root-glob", default=DEFAULT_ROOT_GLOB)
    parser.add_argument(
        "--output-prefix",
        default=str(WORKLIST_DIR / "gemini_best_of_main_derivative"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    roots = parse_roots(args.roots, args.root_glob)
    print("result_roots=" + ",".join(root.name for root in roots))
    rows = build_rows(classes, roots)
    write_outputs(rows, Path(args.output_prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
