#!/usr/bin/env python3
"""Report whether newly arrived OCR artifacts unlock assembly blockers.

This performs no OCR/API calls and does not modify assembly outputs. It indexes
local OCR JSON/MD artifacts, compares them to the latest first-principles
missing-artifact queues, and writes a compact versioned intake report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSEMBLY_GLOB = "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v*"
DEFAULT_OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_v1"

TARGET_FIELDS = [
    "target_scope",
    "target_key",
    "priority_bucket",
    "blocking_context",
    "source_pdf",
    "source_pdf_id",
    "pdf_page",
    "v5_source_key",
    "page_pair",
    "pair_gap",
    "v25_component_id",
    "position_in_component",
    "physical_page_id",
    "image_path",
    "image_path_exists",
    "existing_ocr_artifact",
    "resolved_ocr_artifact",
    "resolution_status",
    "expected_source_page_globs",
    "expected_image_hash_globs",
    "matched_render_count",
    "best_render_path",
    "render_status",
    "matched_artifact_count",
    "best_artifact_path",
    "best_match_method",
    "best_artifact_mtime",
    "best_artifact_age_hours",
    "new_artifact_within_window",
    "unlock_status",
    "recommended_next_step",
]

ARTIFACT_FIELDS = [
    "artifact_path",
    "artifact_kind",
    "source_pdf_id",
    "pdf_page",
    "image_hash_or_stem",
    "mtime",
    "age_hours",
    "size_bytes",
    "target_match_count",
    "target_scopes",
    "target_keys",
]

SUMMARY_FIELDS = ["metric", "name", "count"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    source_pdf_id: str
    pdf_page: str
    image_hash_or_stem: str
    mtime: float
    size_bytes: int


@dataclass(frozen=True)
class Render:
    path: Path
    source_pdf_id: str
    pdf_page: str
    image_hash_or_stem: str
    mtime: float
    size_bytes: int


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def latest_assembly_dir() -> Path:
    candidates = [p for p in ROOT.glob(DEFAULT_ASSEMBLY_GLOB) if p.is_dir() and (p / "summary.json").exists()]
    if not candidates:
        raise SystemExit(f"No assembly output matching {DEFAULT_ASSEMBLY_GLOB}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def split_globs(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def source_page_from_name(path: Path) -> tuple[str, str]:
    match = re.search(r"(\d{7,8})_p(\d+)", path.name)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def image_hash_or_stem_from_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".mistral"):
        stem = stem.removesuffix(".mistral")
    hash_match = re.search(r"([0-9a-f]{40})", stem, flags=re.IGNORECASE)
    if hash_match:
        return hash_match.group(1).lower()
    source_pdf_id, pdf_page = source_page_from_name(path)
    if source_pdf_id and pdf_page:
        return f"{source_pdf_id}_p{pdf_page}"
    return stem


def artifact_kind(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".mistral.md"):
        return "mistral_md"
    if path.suffix.lower() == ".json":
        return "json"
    if path.suffix.lower() == ".md":
        return "md"
    return path.suffix.lower().lstrip(".")


def artifact_priority(path: Path) -> tuple[int, float, str]:
    text = str(path).lower()
    if "/corrected/" in text:
        tier = 0
    elif "v2_best" in text:
        tier = 1
    elif "money_gapfill_ocr" in text:
        tier = 2
    elif "gemini_no_gap_ocr_2026-06-16_v2" in text:
        tier = 3
    elif "gemini_no_gap_ocr_2026-06-16" in text:
        tier = 4
    elif "/results/" in text:
        tier = 5
    elif "mistral_md" in text:
        tier = 6
    else:
        tier = 9
    return tier, -path.stat().st_mtime, str(path)


def artifact_roots(extra_roots: Iterable[str]) -> list[Path]:
    roots: list[Path] = [
        ROOT / "export/money_gapfill_ocr",
        ROOT / "results",
        ROOT / "export_final/sendout_final/sendout_v1/mistral_md",
    ]
    roots.extend(sorted((ROOT / "export_final").glob("gemini_no_gap_ocr_2026-06-16*")))
    roots.extend(Path(root).expanduser() for root in extra_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def index_artifacts(roots: list[Path]) -> tuple[list[Artifact], dict[tuple[str, str], list[Artifact]], dict[str, list[Artifact]]]:
    artifacts: list[Artifact] = []
    by_source_page: dict[tuple[str, str], list[Artifact]] = defaultdict(list)
    by_stem: dict[str, list[Artifact]] = defaultdict(list)
    for root in roots:
        for pattern in ("*.json", "*.md"):
            for path in root.rglob(pattern):
                if path.name.startswith("_"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                source_pdf_id, pdf_page = source_page_from_name(path)
                stem = image_hash_or_stem_from_name(path)
                artifact = Artifact(
                    path=path,
                    kind=artifact_kind(path),
                    source_pdf_id=source_pdf_id,
                    pdf_page=pdf_page,
                    image_hash_or_stem=stem,
                    mtime=stat.st_mtime,
                    size_bytes=stat.st_size,
                )
                artifacts.append(artifact)
                if source_pdf_id and pdf_page:
                    by_source_page[(source_pdf_id, pdf_page)].append(artifact)
                if stem:
                    by_stem[stem].append(artifact)
    for bucket in list(by_source_page.values()) + list(by_stem.values()):
        bucket.sort(key=lambda item: artifact_priority(item.path))
    artifacts.sort(key=lambda item: (-item.mtime, output_path(item.path)))
    return artifacts, by_source_page, by_stem


def index_renders(roots: list[Path]) -> tuple[list[Render], dict[tuple[str, str], list[Render]], dict[str, list[Render]]]:
    renders: list[Render] = []
    by_source_page: dict[tuple[str, str], list[Render]] = defaultdict(list)
    by_stem: dict[str, list[Render]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            source_pdf_id, pdf_page = source_page_from_name(path)
            stem = image_hash_or_stem_from_name(path)
            render = Render(
                path=path,
                source_pdf_id=source_pdf_id,
                pdf_page=pdf_page,
                image_hash_or_stem=stem,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
            )
            renders.append(render)
            if source_pdf_id and pdf_page:
                by_source_page[(source_pdf_id, pdf_page)].append(render)
            if stem:
                by_stem[stem.lower()].append(render)
    for bucket in list(by_source_page.values()) + list(by_stem.values()):
        bucket.sort(key=lambda item: (-item.mtime, output_path(item.path)))
    renders.sort(key=lambda item: (-item.mtime, output_path(item.path)))
    return renders, by_source_page, by_stem


def target_rows(assembly_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename, scope in [
        ("missing_artifact_targets.csv", "physical"),
        ("candidate_missing_artifact_targets.csv", "candidate_review"),
    ]:
        for row in read_csv(assembly_dir / filename):
            rows.append({"target_scope": scope, **row})
    return rows


def expected_stems(row: dict[str, str]) -> set[str]:
    stems: set[str] = set()
    image_path = str(row.get("image_path") or "")
    if image_path:
        stems.add(Path(image_path).stem)
    for field in ["expected_image_hash_globs", "expected_source_page_globs"]:
        for pattern in split_globs(str(row.get(field) or "")):
            base = pattern.split("*", 1)[0].rstrip("_-.")
            stem = Path(base).stem
            if stem:
                stems.add(stem.removesuffix(".mistral"))
    source_pdf_id = str(row.get("source_pdf_id") or "")
    pdf_page = str(row.get("pdf_page") or "")
    if source_pdf_id and pdf_page:
        stems.add(f"{source_pdf_id}_p{pdf_page}")
    return {stem.lower() for stem in stems if stem}


def match_target(
    row: dict[str, str],
    by_source_page: dict[tuple[str, str], list[Artifact]],
    by_stem: dict[str, list[Artifact]],
) -> tuple[list[Artifact], str]:
    candidates: list[Artifact] = []
    methods: list[str] = []
    source_pdf_id = str(row.get("source_pdf_id") or "")
    pdf_page = str(row.get("pdf_page") or "")
    if source_pdf_id and pdf_page:
        hits = by_source_page.get((source_pdf_id, pdf_page), [])
        if hits:
            candidates.extend(hits)
            methods.append("source_pdf_id_pdf_page")
    for stem in expected_stems(row):
        hits = by_stem.get(stem, [])
        if hits:
            candidates.extend(hits)
            methods.append("image_hash_or_expected_stem")
    unique = {artifact.path.resolve(): artifact for artifact in candidates}
    matched = sorted(unique.values(), key=lambda item: artifact_priority(item.path))
    return matched, "+".join(sorted(set(methods)))


def match_target_renders(
    row: dict[str, str],
    by_source_page: dict[tuple[str, str], list[Render]],
    by_stem: dict[str, list[Render]],
) -> list[Render]:
    candidates: list[Render] = []
    source_pdf_id = str(row.get("source_pdf_id") or "")
    pdf_page = str(row.get("pdf_page") or "")
    if source_pdf_id and pdf_page:
        candidates.extend(by_source_page.get((source_pdf_id, pdf_page), []))
    for stem in expected_stems(row):
        candidates.extend(by_stem.get(stem.lower(), []))
    unique = {render.path.resolve(): render for render in candidates}
    return sorted(unique.values(), key=lambda item: (-item.mtime, output_path(item.path)))


def unlock_status(row: dict[str, str], matched: list[Artifact], matched_renders: list[Render]) -> tuple[str, str]:
    if not matched:
        if matched_renders:
            return "render_available_no_ocr_artifact", "OCR the matched local render, then rerun first-principles assembly."
        return "still_blocked_no_local_artifact", "Wait for OCR or attach an OCR artifact matching the expected source-page/image-hash globs."
    if row.get("target_scope") == "physical":
        return "physical_blocker_artifact_available", "Rerun first-principles assembly, then promote only if row-label/column-order constraints pass."
    return "candidate_artifact_available_review_only", "Rerun candidate tests; keep review-only until visual/table-number context confirms the join."


def fingerprint_rows(rows: Iterable[object]) -> str:
    payload = "\n".join(str(row) for row in sorted(rows))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_report(assembly_dir: Path, out_dir: Path, extra_roots: list[str], recent_hours: float) -> dict[str, object]:
    roots = artifact_roots(extra_roots)
    artifacts, by_source_page, by_stem = index_artifacts(roots)
    render_roots = [Path(root).expanduser() for root in extra_roots]
    renders, renders_by_source_page, renders_by_stem = index_renders(render_roots)
    targets = target_rows(assembly_dir)
    now = time.time()
    recent_cutoff = now - recent_hours * 3600

    target_report: list[dict[str, object]] = []
    artifact_matches: dict[Path, list[dict[str, str]]] = defaultdict(list)
    match_fingerprint_parts: list[str] = []
    target_state_fingerprint_parts: list[str] = []
    for row in targets:
        matched, method = match_target(row, by_source_page, by_stem)
        matched_renders = match_target_renders(row, renders_by_source_page, renders_by_stem)
        for artifact in matched:
            artifact_matches[artifact.path.resolve()].append(row)
        best = matched[0] if matched else None
        best_render = matched_renders[0] if matched_renders else None
        status, next_step = unlock_status(row, matched, matched_renders)
        target_id = str(row.get("target_key") or f"{row.get('target_scope', '')}:{row.get('source_pdf_id', '')}:{row.get('pdf_page', '')}:{row.get('image_path', '')}")
        best_artifact_token = ""
        if best:
            best_artifact_token = f"{output_path(best.path)}|{best.mtime:.6f}|{best.size_bytes}|{best.kind}|{method}"
            match_fingerprint_parts.append(f"{target_id}|{best_artifact_token}|matches={len(matched)}")
        target_state_fingerprint_parts.append(
            f"{target_id}|status={status}|renders={len(matched_renders)}|artifacts={len(matched)}|best={best_artifact_token}"
        )
        target_report.append(
            {
                **row,
                "matched_render_count": len(matched_renders),
                "best_render_path": output_path(best_render.path) if best_render else "",
                "render_status": "render_available" if best_render else "no_extra_render_match",
                "matched_artifact_count": len(matched),
                "best_artifact_path": output_path(best.path) if best else "",
                "best_match_method": method,
                "best_artifact_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(best.mtime)) if best else "",
                "best_artifact_age_hours": f"{(now - best.mtime) / 3600:.2f}" if best else "",
                "new_artifact_within_window": "yes" if best and best.mtime >= recent_cutoff else "no",
                "unlock_status": status,
                "recommended_next_step": next_step,
            }
        )

    artifact_report: list[dict[str, object]] = []
    for artifact in artifacts:
        matched_targets = artifact_matches.get(artifact.path.resolve(), [])
        artifact_report.append(
            {
                "artifact_path": output_path(artifact.path),
                "artifact_kind": artifact.kind,
                "source_pdf_id": artifact.source_pdf_id,
                "pdf_page": artifact.pdf_page,
                "image_hash_or_stem": artifact.image_hash_or_stem,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(artifact.mtime)),
                "age_hours": f"{(now - artifact.mtime) / 3600:.2f}",
                "size_bytes": artifact.size_bytes,
                "target_match_count": len(matched_targets),
                "target_scopes": "; ".join(sorted({row.get("target_scope", "") for row in matched_targets if row.get("target_scope")})),
                "target_keys": "; ".join(sorted({row.get("target_key", "") for row in matched_targets if row.get("target_key")})),
            }
        )

    summary_rows: list[dict[str, object]] = [
        {"metric": "paths", "name": "assembly_dir", "count": output_path(assembly_dir)},
        {"metric": "paths", "name": "out_dir", "count": output_path(out_dir)},
        {"metric": "counts", "name": "artifact_roots_scanned", "count": len(roots)},
        {"metric": "counts", "name": "render_roots_scanned", "count": len(render_roots)},
        {"metric": "counts", "name": "artifacts_indexed", "count": len(artifacts)},
        {"metric": "counts", "name": "renders_indexed", "count": len(renders)},
        {"metric": "counts", "name": f"artifacts_recent_{recent_hours:g}h", "count": sum(1 for artifact in artifacts if artifact.mtime >= recent_cutoff)},
        {"metric": "counts", "name": "targets_total", "count": len(targets)},
        {"metric": "counts", "name": "targets_with_render_match", "count": sum(1 for row in target_report if row.get("matched_render_count", 0))},
        {"metric": "counts", "name": "targets_with_render_but_no_ocr_match", "count": sum(1 for row in target_report if row.get("matched_render_count", 0) and not row.get("matched_artifact_count", 0))},
        {"metric": "counts", "name": "targets_with_artifact_match", "count": sum(1 for row in target_report if row.get("matched_artifact_count", 0))},
        {"metric": "counts", "name": "targets_with_recent_artifact_match", "count": sum(1 for row in target_report if row.get("new_artifact_within_window") == "yes")},
        {"metric": "counts", "name": "physical_blockers_with_artifact_match", "count": sum(1 for row in target_report if row.get("target_scope") == "physical" and row.get("matched_artifact_count", 0))},
        {"metric": "counts", "name": "candidate_review_targets_with_artifact_match", "count": sum(1 for row in target_report if row.get("target_scope") == "candidate_review" and row.get("matched_artifact_count", 0))},
        {"metric": "fingerprints", "name": "artifact_match_fingerprint", "count": fingerprint_rows(match_fingerprint_parts)},
        {"metric": "fingerprints", "name": "target_state_fingerprint", "count": fingerprint_rows(target_state_fingerprint_parts)},
    ]
    for metric, counter in [
        ("targets_by_scope", Counter(str(row.get("target_scope") or "") for row in target_report)),
        ("targets_by_unlock_status", Counter(str(row.get("unlock_status") or "") for row in target_report)),
        ("targets_by_render_status", Counter(str(row.get("render_status") or "") for row in target_report)),
        ("targets_by_priority", Counter(str(row.get("priority_bucket") or "") for row in target_report)),
        ("artifacts_by_kind", Counter(str(row.get("artifact_kind") or "") for row in artifact_report)),
        ("artifacts_by_target_match_count", Counter("matched" if row.get("target_match_count") else "unmatched" for row in artifact_report)),
    ]:
        for name, count in sorted(counter.items()):
            summary_rows.append({"metric": metric, "name": name, "count": count})

    out_dir.mkdir(parents=True, exist_ok=True)
    render_needs_ocr = [row for row in target_report if row.get("unlock_status") == "render_available_no_ocr_artifact"]
    still_blocked = [row for row in target_report if row.get("unlock_status") == "still_blocked_no_local_artifact"]
    write_csv(out_dir / "ocr_target_unlock_status.csv", target_report, TARGET_FIELDS)
    write_csv(out_dir / "render_available_no_ocr_queue.csv", render_needs_ocr, TARGET_FIELDS)
    write_csv(out_dir / "still_blocked_no_local_artifact_queue.csv", still_blocked, TARGET_FIELDS)
    write_csv(out_dir / "ocr_artifact_intake_index.csv", artifact_report, ARTIFACT_FIELDS)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    workbook_path = out_dir / "ocr_intake_report.xlsx"
    write_workbook(workbook_path, summary_rows, target_report, artifact_report)
    workbook_bad_member = validate_workbook(workbook_path)
    summary = {
        "assembly_dir": output_path(assembly_dir),
        "out_dir": output_path(out_dir),
        "recent_hours": recent_hours,
        "artifact_roots_scanned": [output_path(root) for root in roots],
        "render_roots_scanned": [output_path(root) for root in render_roots],
        "artifact_count": len(artifacts),
        "render_count": len(renders),
        "recent_artifact_count": sum(1 for artifact in artifacts if artifact.mtime >= recent_cutoff),
        "target_count": len(targets),
        "targets_with_render_match": sum(1 for row in target_report if row.get("matched_render_count", 0)),
        "targets_with_render_but_no_ocr_match": sum(1 for row in target_report if row.get("matched_render_count", 0) and not row.get("matched_artifact_count", 0)),
        "targets_with_artifact_match": sum(1 for row in target_report if row.get("matched_artifact_count", 0)),
        "targets_with_recent_artifact_match": sum(1 for row in target_report if row.get("new_artifact_within_window") == "yes"),
        "physical_blockers_with_artifact_match": sum(1 for row in target_report if row.get("target_scope") == "physical" and row.get("matched_artifact_count", 0)),
        "candidate_review_targets_with_artifact_match": sum(1 for row in target_report if row.get("target_scope") == "candidate_review" and row.get("matched_artifact_count", 0)),
        "artifact_match_fingerprint": fingerprint_rows(match_fingerprint_parts),
        "target_state_fingerprint": fingerprint_rows(target_state_fingerprint_parts),
        "targets_by_unlock_status": dict(Counter(str(row.get("unlock_status") or "") for row in target_report)),
        "workbook_zip_bad_member": workbook_bad_member,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_workbook(
    path: Path,
    summary_rows: list[dict[str, object]],
    target_report: list[dict[str, object]],
    artifact_report: list[dict[str, object]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for name, rows, fields in [
        ("Summary", summary_rows, SUMMARY_FIELDS),
        ("Target Unlock Status", target_report, TARGET_FIELDS),
        ("Render Needs OCR", [row for row in target_report if row.get("unlock_status") == "render_available_no_ocr_artifact"], TARGET_FIELDS),
        ("Still Blocked", [row for row in target_report if row.get("unlock_status") == "still_blocked_no_local_artifact"], TARGET_FIELDS),
        ("Artifact Intake Index", artifact_report, ARTIFACT_FIELDS),
    ]:
        ws = workbook.add_worksheet(name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col, field in enumerate(fields):
                value = row.get(field, "")
                fmt = wrap_fmt if field in {"recommended_next_step", "target_keys", "artifact_path", "best_artifact_path"} else None
                ws.write(row_idx, col, value, fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = min(max(len(field) + 2, 12), 48)
            ws.set_column(col, col, width)
    workbook.close()


def validate_workbook(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir", default="", help="Existing first-principles assembly output directory. Defaults to latest v* output.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Versioned output directory for the intake report.")
    parser.add_argument("--extra-root", action="append", default=[], help="Additional local OCR/artifact root to scan.")
    parser.add_argument("--recent-hours", type=float, default=6.0, help="Mark artifacts newer than this many hours as recent.")
    args = parser.parse_args()

    assembly_dir = Path(args.assembly_dir).expanduser() if args.assembly_dir else latest_assembly_dir()
    if not assembly_dir.is_absolute():
        assembly_dir = ROOT / assembly_dir
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    summary = build_report(assembly_dir, out_dir, args.extra_root, args.recent_hours)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
