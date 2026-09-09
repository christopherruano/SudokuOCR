#!/usr/bin/env python3
"""Assemble strict seed pairs into row-level tables when constraints allow it.

This consumes the v25 first-principles seed components. It does not perform OCR
or change earlier outputs. The goal is to test the actual row-label inheritance
needed for right-half pages: if page B has no row labels, inherit labels from the
adjacent page A only when row/section structure makes that safe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V25_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_seed_assembly_v25"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_row_assembly_v26"

MEMBERS_CSV = V25_DIR / "v25_seed_component_members.csv"
EDGES_CSV = V25_DIR / "v25_pairwise_seed_edges.csv"
ASSEMBLIES_CSV = V25_DIR / "v25_ironclad_physical_assemblies.csv"

BAD_PHYSICAL_PAGE_KINDS = {
    "context_or_blank",
    "offtarget_table",
    "derivative_selected_caste_race_tribe",
}


@dataclass
class ParsedRow:
    label: str
    values: dict[str, Any]
    raw: dict[str, Any] | list[str]


@dataclass
class ParsedBlock:
    block_id: str
    section_name: str
    headers: list[str]
    rows: list[ParsedRow]


@dataclass(frozen=True)
class ArtifactCandidate:
    path: Path
    priority: int
    method: str


BASE_ARTIFACT_SEARCH_ROOTS = [
    ROOT / "export/money_gapfill_ocr",
    ROOT / "results",
]

ARTIFACT_RESOLUTION_FIELDS = [
    "v25_component_id",
    "position_in_component",
    "physical_page_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "pdf_page",
    "image_path",
    "existing_ocr_artifact",
    "resolved_ocr_artifact",
    "resolution_status",
    "resolution_method",
    "candidate_count",
    "candidate_paths",
]

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


def stable_id(prefix: str, *parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact_priority(path: Path) -> int:
    text = str(path).lower()
    if "/corrected/" in text:
        return 0
    if "v2_best" in text:
        return 1
    if "money_gapfill_ocr" in text:
        return 2
    if "gemini_no_gap_ocr_2026-06-16_v2" in text:
        return 3
    if "gemini_no_gap_ocr_2026-06-16" in text:
        return 4
    if "/results/" in text or text.endswith("/results"):
        return 5
    if "mistral_md" in text:
        return 6
    return 9


def artifact_search_roots() -> list[Path]:
    roots = list(BASE_ARTIFACT_SEARCH_ROOTS)
    roots.extend(sorted((ROOT / "export_final").glob("gemini_no_gap_ocr_2026-06-16*")))
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(root)
    return unique_roots


def build_artifact_indexes() -> tuple[dict[tuple[str, str], list[ArtifactCandidate]], dict[str, list[ArtifactCandidate]]]:
    by_source_page: dict[tuple[str, str], list[ArtifactCandidate]] = defaultdict(list)
    by_image_stem: dict[str, list[ArtifactCandidate]] = defaultdict(list)
    for root in artifact_search_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name.startswith("_"):
                continue
            match = re.search(r"(\d{7,8})_p(\d+)", path.name)
            if match:
                by_source_page[(match.group(1), match.group(2))].append(
                    ArtifactCandidate(path=path, priority=artifact_priority(path), method="source_pdf_id_pdf_page_json")
                )
    mistral_root = ROOT / "export_final/sendout_final/sendout_v1/mistral_md"
    if mistral_root.exists():
        for path in mistral_root.glob("*.mistral.md"):
            by_image_stem[path.name.replace(".mistral.md", "")].append(
                ArtifactCandidate(path=path, priority=artifact_priority(path), method="image_hash_mistral_md")
            )
    return by_source_page, by_image_stem


def choose_artifact(candidates: list[ArtifactCandidate]) -> tuple[ArtifactCandidate | None, str]:
    if not candidates:
        return None, "no_match"
    unique = {candidate.path.resolve(): candidate for candidate in candidates}
    candidates = list(unique.values())
    best_priority = min(candidate.priority for candidate in candidates)
    best = sorted((candidate for candidate in candidates if candidate.priority == best_priority), key=lambda c: str(c.path))
    if len(best) == 1:
        return best[0], "resolved"
    return None, "ambiguous_top_priority"


def resolve_member_artifact(
    member: dict[str, str],
    by_source_page: dict[tuple[str, str], list[ArtifactCandidate]],
    by_image_stem: dict[str, list[ArtifactCandidate]],
) -> tuple[dict[str, str], dict[str, object]]:
    existing = member.get("ocr_artifact", "")
    report = {
        "v25_component_id": member.get("v25_component_id", ""),
        "position_in_component": member.get("position_in_component", ""),
        "physical_page_id": member.get("physical_page_id", ""),
        "v5_source_key": member.get("v5_source_key", ""),
        "source_pdf": member.get("source_pdf", ""),
        "source_pdf_id": member.get("source_pdf_id", ""),
        "pdf_page": member.get("pdf_page", ""),
        "image_path": member.get("image_path", ""),
        "existing_ocr_artifact": existing,
        "resolved_ocr_artifact": existing,
        "resolution_status": "existing_artifact" if existing else "unresolved_no_match",
        "resolution_method": "existing" if existing else "",
        "candidate_count": 0,
        "candidate_paths": "",
    }
    if existing:
        return member, report

    candidates: list[ArtifactCandidate] = []
    source_pdf_id = member.get("source_pdf_id", "")
    pdf_page = member.get("pdf_page", "")
    if source_pdf_id and pdf_page:
        candidates = by_source_page.get((source_pdf_id, pdf_page), [])
    method = "source_pdf_id_pdf_page_json"

    if not candidates:
        image_stem = Path(member.get("image_path", "")).stem
        if image_stem:
            candidates = by_image_stem.get(image_stem, [])
            method = "image_hash_mistral_md"

    candidate_paths = [output_path(candidate.path) for candidate in sorted(candidates, key=lambda c: (c.priority, str(c.path)))]
    report["candidate_count"] = len(candidate_paths)
    report["candidate_paths"] = "; ".join(candidate_paths[:12])
    chosen, status = choose_artifact(candidates)
    if chosen is None:
        report["resolution_status"] = f"unresolved_{status}"
        report["resolution_method"] = method if candidates else ""
        return member, report

    updated = dict(member)
    updated["ocr_artifact"] = output_path(chosen.path)
    report["resolved_ocr_artifact"] = updated["ocr_artifact"]
    report["resolution_status"] = "resolved"
    report["resolution_method"] = chosen.method
    return updated, report


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def float_or_zero(value: object) -> float:
    try:
        return float(str(value or "").strip() or 0)
    except ValueError:
        return 0.0


def parse_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    s = compact(value)
    if not s or s in {"...", "..", ".", "-", "--", "—", "–"}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"^\(([-+]?\d+(?:\.\d+)?)\)$", r"-\1", s)
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?\d+\.\d+", s):
        return float(s)
    return None


def is_numericish(value: object) -> bool:
    return parse_number(value) is not None


def is_generic_or_missing_label(value: object) -> bool:
    s = clean_text(value)
    if not s:
        return True
    if s in {"unknown", "na", "none", "blank", "row", "age"}:
        return True
    if re.fullmatch(r"row\s*\d+", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def candidate_text(row: dict[str, object]) -> str:
    return " ".join(
        str(row.get(field, ""))
        for field in [
            "prev_title",
            "title",
            "prev_table_number_key",
            "table_number_key",
            "prev_table_family_key",
            "table_family_key",
        ]
    ).lower()


def has_age_civil_signal(row: dict[str, object]) -> bool:
    text = candidate_text(row)
    return "age" in text and "civil" in text


def is_row_label_dependency_candidate(edge: dict[str, str]) -> bool:
    if edge.get("page_gap") != "1" or edge.get("missing_pages"):
        return False
    if edge.get("v25_decision") == "seed_physical_join":
        return False
    prev_ratio = float_or_zero(edge.get("prev_row_label_ratio"))
    row_ratio = float_or_zero(edge.get("row_label_ratio"))
    if prev_ratio < 0.50 or row_ratio > 0.25:
        return False
    if edge.get("prev_page_kind") in BAD_PHYSICAL_PAGE_KINDS or edge.get("page_kind") in BAD_PHYSICAL_PAGE_KINDS:
        return False
    return True


def flatten_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, subvalue in value.items():
            child = f"{prefix}.{compact(key)}" if prefix else compact(key)
            out.update(flatten_values(subvalue, child))
        return out
    return {prefix: value}


def numeric_cell_count(rows: Iterable[ParsedRow]) -> int:
    total = 0
    for row in rows:
        total += sum(1 for value in row.values.values() if parse_number(value) is not None)
    return total


def row_label_ratio(rows: list[ParsedRow]) -> float:
    if not rows:
        return 0.0
    good = sum(1 for row in rows if not is_generic_or_missing_label(row.label))
    return good / len(rows)


def is_placeholder_value(value: object) -> bool:
    s = compact(value)
    return not s or s in {"...", "..", ".", "-", "--", "—", "–"}


def row_all_placeholders(row: ParsedRow) -> bool:
    return bool(row.values) and all(is_placeholder_value(value) for value in row.values.values())


def age_zero_to_five_prefix(rows: list[ParsedRow]) -> bool:
    labels = [clean_text(row.label) for row in rows[:6]]
    expected = ["0 1", "1 2", "2 3", "3 4", "4 5", "total 0 5"]
    return labels == expected


def maybe_insert_missing_total_0_5_placeholder(left_block: ParsedBlock, right_block: ParsedBlock) -> tuple[ParsedBlock, list[str]]:
    """Repair a narrow, auditable markdown OCR omission.

    Several right-half pages omit the TOTAL 0-5 row. The left page carries the
    exact row labels, while the right page is label-deficient and proceeds from
    the five child-age rows directly into 5-10. Inserting a nonnumeric
    placeholder at row index 5 restores row alignment without inventing totals.
    """
    if len(right_block.rows) != len(left_block.rows) - 1:
        return right_block, []
    if not age_zero_to_five_prefix(left_block.rows):
        return right_block, []
    if len(right_block.rows) < 6:
        return right_block, []
    if row_label_ratio(right_block.rows) > 0.10:
        return right_block, []
    if row_all_placeholders(right_block.rows[5]):
        return right_block, []
    if all(row_all_placeholders(row) for row in right_block.rows[:5]):
        note = "imputed_missing_right_total_0_5_all_ellipsis_row"
    else:
        note = "imputed_missing_right_total_0_5_unread_row"
    placeholder_values = {header: "..." for header in right_block.headers}
    placeholder = ParsedRow(label="", values=placeholder_values, raw=["..."] * len(right_block.headers))
    rows = [*right_block.rows[:5], placeholder, *right_block.rows[5:]]
    repaired = ParsedBlock(
        block_id=f"{right_block.block_id}_pad_total_0_5",
        section_name=right_block.section_name,
        headers=right_block.headers,
        rows=rows,
    )
    return repaired, [note]


def make_unique(headers: list[str]) -> list[str]:
    out: list[str] = []
    counts: Counter[str] = Counter()
    for idx, header in enumerate(headers):
        base = compact(header) or f"col_{idx + 1}"
        counts[base] += 1
        out.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return out


def choose_json_label_key(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    preferred = [metadata.get("row_label_name"), "district", "District", "age", "Age", "AGE.", "Caste", "Caste, &c., and sub-divisions.", "Unknown"]
    for key in preferred:
        if key and key in row and not isinstance(row[key], dict):
            return str(key)
    scalar_keys = [str(key) for key, value in row.items() if not isinstance(value, dict)]
    nonnumeric = [key for key in scalar_keys if not is_numericish(row.get(key))]
    if nonnumeric:
        return nonnumeric[0]
    return scalar_keys[0] if scalar_keys else None


def parse_json_ocr(path: Path) -> tuple[list[ParsedBlock], list[str]]:
    warnings: list[str] = []
    obj = json.loads(path.read_text(encoding="utf-8"))
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    data = obj.get("data")
    if not isinstance(data, list):
        return [], ["json_data_not_list"]
    blocks: list[ParsedBlock] = []
    for sec_idx, section in enumerate(data):
        if not isinstance(section, dict):
            warnings.append(f"section_{sec_idx}_not_dict")
            continue
        raw_rows = section.get("rows")
        if not isinstance(raw_rows, list):
            warnings.append(f"section_{sec_idx}_rows_not_list")
            continue
        parsed_rows: list[ParsedRow] = []
        headers_order: list[str] = []
        headers_seen: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            label_key = choose_json_label_key(raw, metadata)
            label = compact(raw.get(label_key, "")) if label_key else ""
            values: dict[str, Any] = {}
            for key, value in raw.items():
                if key == label_key:
                    continue
                if not isinstance(value, dict) and compact(value) == label and key != label_key:
                    continue
                flat = flatten_values(value, compact(key))
                for flat_key, flat_value in flat.items():
                    if flat_key:
                        values[flat_key] = flat_value
                        if flat_key not in headers_seen:
                            headers_seen.add(flat_key)
                            headers_order.append(flat_key)
            parsed_rows.append(ParsedRow(label=label, values=values, raw=raw))
        blocks.append(
            ParsedBlock(
                block_id=f"json_section_{sec_idx + 1}",
                section_name=compact(section.get("name") or metadata.get("title") or f"section_{sec_idx + 1}"),
                headers=headers_order,
                rows=parsed_rows,
            )
        )
    return blocks, warnings


def split_md_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [compact(cell) for cell in line.split("|")]


def is_md_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def is_column_number_row(cells: list[str]) -> bool:
    filled = [clean_text(cell) for cell in cells if clean_text(cell)]
    if len(filled) < 2:
        return False
    return all(re.fullmatch(r"\d+", cell) for cell in filled)


def is_probable_data_row(cells: list[str]) -> bool:
    if not cells:
        return False
    if is_md_separator(cells) or is_column_number_row(cells):
        return False
    nums = sum(1 for cell in cells if parse_number(cell) is not None)
    blanks = sum(1 for cell in cells if not clean_text(cell))
    if nums >= 2:
        return True
    if cells and not is_generic_or_missing_label(cells[0]) and nums >= 1:
        return True
    if blanks < len(cells) and any(cell in {"...", "..", "."} for cell in cells):
        return True
    return False


def preceding_section_name(lines: list[str], table_start: int) -> str:
    headings: list[str] = []
    idx = table_start - 1
    while idx >= 0 and len(headings) < 4:
        line = compact(lines[idx].lstrip("#").strip())
        if not line or line.startswith("|"):
            idx -= 1
            continue
        if re.fullmatch(r"\d+", line):
            idx -= 1
            continue
        headings.append(line)
        if re.search(r"^[A-Z]\s*[-—]", line) or re.search(r"\bTOTAL POPULATION\b", line, re.I):
            break
        idx -= 1
    return " / ".join(reversed(headings))


def parse_md_ocr(path: Path) -> tuple[list[ParsedBlock], list[str]]:
    warnings: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: list[ParsedBlock] = []
    idx = 0
    table_idx = 0
    while idx < len(lines):
        if not lines[idx].lstrip().startswith("|"):
            idx += 1
            continue
        start = idx
        table_lines: list[str] = []
        while idx < len(lines) and lines[idx].lstrip().startswith("|"):
            table_lines.append(lines[idx])
            idx += 1
        cell_rows = [split_md_cells(line) for line in table_lines]
        if len(cell_rows) < 3 or not any(is_md_separator(cells) for cells in cell_rows):
            continue
        table_idx += 1
        section_name = preceding_section_name(lines, start) or f"markdown_table_{table_idx}"
        sep_pos = next((i for i, cells in enumerate(cell_rows) if is_md_separator(cells)), 1)
        data_start = sep_pos + 1
        scan_limit = min(len(cell_rows), sep_pos + 6)
        for pos in range(sep_pos + 1, scan_limit):
            if is_column_number_row(cell_rows[pos]):
                data_start = pos + 1
                break
        else:
            # Common markdown shape: header row, separator, subheader row, then data.
            data_start = min(len(cell_rows), sep_pos + 2)

        header_candidates = cell_rows[:sep_pos] + cell_rows[sep_pos + 1 : data_start]
        width = max((len(cells) for cells in cell_rows), default=0)
        headers: list[str] = []
        for col in range(width):
            parts = []
            for cells in header_candidates:
                if col < len(cells):
                    part = compact(cells[col])
                    if part and not re.fullmatch(r"\d+", clean_text(part)):
                        parts.append(part)
            headers.append(" / ".join(parts) or f"col_{col + 1}")
        headers = make_unique(headers)

        data_rows = [
            cells
            for cells in cell_rows[data_start:]
            if cells and not is_md_separator(cells) and any(clean_text(cell) or cell.strip() in {"...", "..", "."} for cell in cells)
        ]
        parsed_rows: list[ParsedRow] = []
        for cells in data_rows:
            cells = cells + [""] * (len(headers) - len(cells))
            first = cells[0] if cells else ""
            left_has_label = not is_generic_or_missing_label(first) and not is_numericish(first) and first not in {"...", "..", "."}
            label = first if left_has_label else ""
            start_col = 1 if left_has_label else 0
            values = {headers[col]: cells[col] for col in range(start_col, len(headers)) if col < len(cells)}
            parsed_rows.append(ParsedRow(label=label, values=values, raw=cells))
        blocks.append(ParsedBlock(block_id=f"markdown_table_{table_idx}", section_name=section_name, headers=headers, rows=parsed_rows))
    if not blocks:
        warnings.append("no_markdown_tables_parsed")
    return blocks, warnings


def parse_artifact(path_value: str) -> tuple[list[ParsedBlock], list[str], str]:
    if not path_value:
        return [], ["missing_ocr_artifact"], "missing"
    path = (ROOT / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value)
    if not path.exists():
        return [], [f"artifact_not_found:{path_value}"], "missing"
    suffixes = "".join(path.suffixes).lower()
    try:
        if suffixes.endswith(".json"):
            blocks, warnings = parse_json_ocr(path)
            return blocks, warnings, "json"
        if suffixes.endswith(".md"):
            blocks, warnings = parse_md_ocr(path)
            return blocks, warnings, "markdown"
    except Exception as exc:  # keep the run alive and make parser failures auditable
        return [], [f"parse_error:{type(exc).__name__}:{exc}"], "error"
    return [], [f"unsupported_artifact_type:{path.suffix}"], "unsupported"


def block_summary(blocks: list[ParsedBlock]) -> dict[str, object]:
    rows = [row for block in blocks for row in block.rows]
    return {
        "block_count": len(blocks),
        "row_count_total": len(rows),
        "row_label_ratio": f"{row_label_ratio(rows):.3f}" if rows else "",
        "numeric_cell_count": numeric_cell_count(rows),
        "section_names": "; ".join(block.section_name for block in blocks[:8]),
    }


def align_blocks(left_blocks: list[ParsedBlock], right_blocks: list[ParsedBlock]) -> tuple[list[tuple[ParsedBlock, ParsedBlock]], list[str]]:
    flags: list[str] = []
    if not left_blocks or not right_blocks:
        return [], ["missing_parsed_blocks"]
    if len(left_blocks) == len(right_blocks):
        return list(zip(left_blocks, right_blocks)), flags
    left_total = sum(len(block.rows) for block in left_blocks)
    right_total = sum(len(block.rows) for block in right_blocks)
    if len(left_blocks) == 1 and left_total == right_total:
        # Rare case: page B was split into sub-blocks but total rows still align.
        merged = ParsedBlock(
            block_id="right_blocks_flattened",
            section_name=" / ".join(block.section_name for block in right_blocks[:8]),
            headers=sorted({header for block in right_blocks for header in block.headers}),
            rows=[row for block in right_blocks for row in block.rows],
        )
        return [(left_blocks[0], merged)], ["right_blocks_flattened_for_row_alignment"]
    flags.append(f"section_count_mismatch:left_{len(left_blocks)}_right_{len(right_blocks)}")
    return [], flags


def value_constraint_summary(assembled_rows: list[dict[str, object]]) -> tuple[str, str]:
    """Small, conservative cross-page sanity check.

    For common age/civil-condition pages, left columns hold population/unmarried
    and right columns hold married/widowed. When all three civil-status person
    counts exist, they should sum to population persons. If the names are not
    recognizable, this is marked not_applicable rather than failed.
    """
    applicable = 0
    failures: list[str] = []
    for row in assembled_rows:
        values = row.get("values")
        if not isinstance(values, dict):
            continue
        pop = unmarried = married = widowed = None
        for key, value in values.items():
            ck = clean_text(key)
            num = parse_number(value)
            if num is None:
                continue
            if "population" in ck and "person" in ck and pop is None:
                pop = num
            elif "unmarried" in ck and "person" in ck and unmarried is None:
                unmarried = num
            elif "married" in ck and "person" in ck and "unmarried" not in ck and married is None:
                married = num
            elif ("widowed" in ck or "widower" in ck or "widow" in ck) and "person" in ck and widowed is None:
                widowed = num
        if None not in {pop, unmarried, married, widowed}:
            applicable += 1
            if abs(float(pop) - float(unmarried + married + widowed)) > 1:
                failures.append(str(row.get("row_label") or row.get("row_index")))
    if not applicable:
        return "not_applicable", ""
    if failures:
        return "fail", f"{len(failures)} civil-status sum failures: {', '.join(failures[:12])}"
    return "pass", f"{applicable} civil-status sum checks"


def ordered_value_items(headers: list[str], values: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return values in visual header order, preserving any parser extras at end."""
    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for header in headers:
        if header in values:
            out.append((header, values[header]))
            seen.add(header)
    for key, value in values.items():
        if key not in seen:
            out.append((key, value))
    return out


def assemble_pair(component_id: str, left: dict[str, str], right: dict[str, str], edge: dict[str, str]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    left_blocks, left_warnings, left_type = parse_artifact(left.get("ocr_artifact", ""))
    right_blocks, right_warnings, right_type = parse_artifact(right.get("ocr_artifact", ""))
    left_summary = block_summary(left_blocks)
    right_summary = block_summary(right_blocks)
    aligned, align_flags = align_blocks(left_blocks, right_blocks)

    assembled_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    flags: list[str] = []
    flags.extend(left_warnings)
    flags.extend(f"right_{warning}" for warning in right_warnings)
    flags.extend(align_flags)

    left_ratio = float(left_summary["row_label_ratio"] or 0)
    right_ratio = float(right_summary["row_label_ratio"] or 0)
    right_missing_labels = right_ratio <= 0.10
    if not right_missing_labels:
        flags.append(f"right_page_not_label_deficient:{right_ratio:.3f}")
    if left_ratio < 0.75:
        flags.append(f"left_page_label_ratio_too_low:{left_ratio:.3f}")

    for block_idx, (left_block, original_right_block) in enumerate(aligned, start=1):
        right_block, block_notes = maybe_insert_missing_total_0_5_placeholder(left_block, original_right_block)
        row_count_match = len(left_block.rows) == len(right_block.rows)
        right_numeric = numeric_cell_count(right_block.rows)
        left_labels = sum(1 for row in left_block.rows if not is_generic_or_missing_label(row.label))
        block_flags: list[str] = []
        if not row_count_match:
            block_flags.append(f"row_count_mismatch:left_{len(left_block.rows)}_right_{len(right_block.rows)}")
        if left_labels < len(left_block.rows):
            block_flags.append(f"left_block_missing_labels:{len(left_block.rows)-left_labels}")
        if right_numeric == 0:
            block_flags.append("right_block_has_no_numeric_cells")
        if row_label_ratio(right_block.rows) > 0.10:
            block_flags.append(f"right_block_has_own_labels:{row_label_ratio(right_block.rows):.3f}")
        left_column_order = [f"left.{header}" for header in left_block.headers]
        right_column_order = [f"right.{header}" for header in right_block.headers]
        assembled_column_order = [*left_column_order, *right_column_order]
        column_order_flags: list[str] = []
        if not left_column_order:
            column_order_flags.append("left_columns_missing")
        if not right_column_order:
            column_order_flags.append("right_columns_missing")
        if any(not key.startswith("left.") for key in left_column_order):
            column_order_flags.append("left_prefix_error")
        if any(not key.startswith("right.") for key in right_column_order):
            column_order_flags.append("right_prefix_error")
        constraint_rows.append(
            {
                "v26_constraint_id": stable_id("v26constraint", component_id, block_idx),
                "v25_component_id": component_id,
                "block_index": block_idx,
                "left_block_id": left_block.block_id,
                "right_block_id": right_block.block_id,
                "left_section_name": left_block.section_name,
                "right_section_name": right_block.section_name,
                "left_row_count": len(left_block.rows),
                "right_row_count": len(right_block.rows),
                "row_count_match": "yes" if row_count_match else "no",
                "left_label_ratio": f"{row_label_ratio(left_block.rows):.3f}",
                "right_label_ratio": f"{row_label_ratio(right_block.rows):.3f}",
                "right_numeric_cell_count": right_numeric,
                "left_column_count": len(left_column_order),
                "right_column_count": len(right_column_order),
                "assembled_column_count": len(assembled_column_order),
                "column_order_status": "pass" if not column_order_flags else "fail",
                "column_order_rule": "visual_header_order:left_page_columns_then_right_page_columns",
                "left_column_order": "; ".join(left_column_order),
                "right_column_order": "; ".join(right_column_order),
                "assembled_column_order": "; ".join(assembled_column_order),
                "constraint_status": "pass" if not block_flags else "fail",
                "constraint_flags": "; ".join([*block_notes, *block_flags, *column_order_flags]),
            }
        )
        flags.extend(block_notes)
        if block_flags:
            flags.extend(block_flags)
            continue
        for row_idx, (left_row, right_row) in enumerate(zip(left_block.rows, right_block.rows), start=1):
            values: dict[str, Any] = {}
            for key, value in ordered_value_items(left_block.headers, left_row.values):
                values[f"left.{key}"] = value
            for key, value in ordered_value_items(right_block.headers, right_row.values):
                values[f"right.{key}"] = value
            column_order = list(values.keys())
            assembled_rows.append(
                {
                    "v26_row_id": stable_id("v26row", component_id, block_idx, row_idx, left_row.label),
                    "v25_component_id": component_id,
                    "block_index": block_idx,
                    "row_index": row_idx,
                    "row_label": left_row.label,
                    "left_pdf_page": left.get("pdf_page", ""),
                    "right_pdf_page": right.get("pdf_page", ""),
                    "left_section_name": left_block.section_name,
                    "right_section_name": right_block.section_name,
                    "column_order_rule": "visual_header_order:left_page_columns_then_right_page_columns",
                    "column_order": column_order,
                    "values": values,
                }
            )

    failed_block_count = sum(1 for row in constraint_rows if row.get("constraint_status") == "fail")
    passed_block_count = sum(1 for row in constraint_rows if row.get("constraint_status") == "pass")
    value_status, value_reason = value_constraint_summary(assembled_rows)
    structural_status = "not_attempted"
    if aligned and constraint_rows:
        structural_status = "pass" if failed_block_count == 0 else ("partial" if passed_block_count > 0 else "fail")
    value_flags: list[str] = []
    if value_status == "fail":
        value_flags.append(value_reason)

    if not left.get("ocr_artifact") or not right.get("ocr_artifact"):
        status = "blocked_missing_ocr_artifact"
    elif left_type in {"unsupported", "error"} or right_type in {"unsupported", "error"}:
        status = "blocked_parser"
    elif not aligned:
        status = "review_no_safe_block_alignment"
    elif structural_status == "pass" and value_status != "fail" and assembled_rows:
        status = "assembled_pass"
    elif structural_status == "pass" and value_status == "fail" and assembled_rows:
        status = "assembled_complete_value_review"
    elif assembled_rows:
        status = "partial_assembled_review"
    elif flags:
        status = "review_failed_constraints"
    elif assembled_rows:
        status = "assembled_pass"
    else:
        status = "review_no_rows_assembled"

    row = {
        "v26_attempt_id": stable_id("v26attempt", component_id),
        "v25_component_id": component_id,
        "v26_status": status,
        "recommended_action": {
            "assembled_pass": "use_as_seed_row_assembly",
            "assembled_complete_value_review": "review_values_before_static_site_use",
            "partial_assembled_review": "use_only_passing_blocks_after_manual_review",
            "review_failed_constraints": "manual_review_or_reocr_before_using",
            "review_no_safe_block_alignment": "manual_review_or_reocr_before_using",
            "blocked_missing_ocr_artifact": "OCR_existing_images_before_row_assembly",
            "blocked_parser": "add_parser_or_normalize_artifact_before_row_assembly",
            "review_no_rows_assembled": "manual_review_before_using",
        }.get(status, "manual_review_before_using"),
        "v5_source_key": left.get("v5_source_key", ""),
        "source_pdf": left.get("source_pdf", "") or right.get("source_pdf", ""),
        "source_pdf_id": left.get("source_pdf_id", "") or right.get("source_pdf_id", ""),
        "year": left.get("year", "") or right.get("year", ""),
        "province": left.get("province", "") or right.get("province", ""),
        "page_pair": f"{left.get('pdf_page','')}-{right.get('pdf_page','')}",
        "edge_layer": edge.get("v25_decision_layer", ""),
        "edge_reason": edge.get("v25_reason", ""),
        "left_artifact_type": left_type,
        "right_artifact_type": right_type,
        "left_ocr_artifact": left.get("ocr_artifact", ""),
        "right_ocr_artifact": right.get("ocr_artifact", ""),
        "left_block_count": left_summary["block_count"],
        "right_block_count": right_summary["block_count"],
        "passed_block_count": passed_block_count,
        "failed_block_count": failed_block_count,
        "structural_status": structural_status,
        "left_row_count_total": left_summary["row_count_total"],
        "right_row_count_total": right_summary["row_count_total"],
        "left_label_ratio": left_summary["row_label_ratio"],
        "right_label_ratio": right_summary["row_label_ratio"],
        "left_numeric_cell_count": left_summary["numeric_cell_count"],
        "right_numeric_cell_count": right_summary["numeric_cell_count"],
        "assembled_row_count": len(assembled_rows),
        "value_constraint_status": value_status,
        "value_constraint_reason": value_reason,
        "constraint_flags": "; ".join(dict.fromkeys(flag for flag in [*flags, *value_flags] if flag)),
        "left_title": left.get("title", ""),
        "right_title": right.get("title", ""),
        "left_image_path": left.get("image_path", ""),
        "right_image_path": right.get("image_path", ""),
    }
    return row, assembled_rows, constraint_rows


def write_workbook(path: Path, sheets: dict[str, tuple[list[dict[str, object]], list[str]]]) -> None:
    workbook = xlsxwriter.Workbook(str(path))
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, (rows, fields) in sheets.items():
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header_fmt)
        for row_idx, row in enumerate(rows, start=1):
            for col_idx, field in enumerate(fields):
                fmt = wrap_fmt if any(token in field for token in ["reason", "flags", "title", "artifact", "image", "section"]) else None
                ws.write(row_idx, col_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            width = 16
            if field in {"constraint_flags", "edge_reason", "left_title", "right_title", "left_image_path", "right_image_path"}:
                width = 48
            elif "artifact" in field or "section" in field:
                width = 34
            ws.set_column(col, col, width)
    workbook.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    members = read_csv(MEMBERS_CSV)
    edges = {row.get("v25_edge_id", ""): row for row in read_csv(EDGES_CSV)}
    assemblies = {row.get("v25_component_id", ""): row for row in read_csv(ASSEMBLIES_CSV)}
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in members:
        by_component[member.get("v25_component_id", "")].append(member)

    attempts: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    assembled_jsonl_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    artifact_resolution_rows: list[dict[str, object]] = []
    candidate_attempts: list[dict[str, object]] = []
    candidate_constraint_rows: list[dict[str, object]] = []
    candidate_assembled_jsonl_rows: list[dict[str, object]] = []
    candidate_artifact_resolution_rows: list[dict[str, object]] = []
    by_source_page, by_image_stem = build_artifact_indexes()
    member_by_id = {member.get("physical_page_id", ""): member for member in members}

    target_components = [
        comp_id
        for comp_id, comp_members in by_component.items()
        if comp_id in assemblies and len(comp_members) == 2
    ]

    for comp_id in sorted(target_components):
        comp_members = sorted(by_component[comp_id], key=lambda row: int(row.get("position_in_component") or 0))
        left, left_report = resolve_member_artifact(comp_members[0], by_source_page, by_image_stem)
        right, right_report = resolve_member_artifact(comp_members[1], by_source_page, by_image_stem)
        artifact_resolution_rows.extend([left_report, right_report])
        edge = edges.get(right.get("join_from_prev_edge_id", ""), {})
        attempt, assembled_rows, constraints = assemble_pair(comp_id, left, right, edge)
        attempts.append(attempt)
        constraint_rows.extend(constraints)
        for row in assembled_rows:
            assembled_jsonl_rows.append(row)
        if attempt.get("v26_status") != "assembled_pass":
            review_rows.append(attempt)

    candidate_edges = [
        edge
        for edge in edges.values()
        if is_row_label_dependency_candidate(edge)
    ]
    for edge in sorted(
        candidate_edges,
        key=lambda row: (
            str(row.get("v5_source_key") or ""),
            int_or_none(row.get("prev_pdf_page")) or 10**9,
            int_or_none(row.get("pdf_page")) or 10**9,
            str(row.get("v25_edge_id") or ""),
        ),
    ):
        left_member = member_by_id.get(edge.get("prev_physical_page_id", ""))
        right_member = member_by_id.get(edge.get("physical_page_id", ""))
        component_id = stable_id(
            "v26cand",
            edge.get("v25_edge_id"),
            edge.get("prev_physical_page_id"),
            edge.get("physical_page_id"),
            edge.get("prev_pdf_page"),
            edge.get("pdf_page"),
        )
        if not left_member or not right_member:
            candidate_attempts.append(
                {
                    "candidate_scope": "row_label_dependency_review_candidate",
                    "candidate_promotion_policy": "review_only_never_auto_promote",
                    "source_v25_edge_id": edge.get("v25_edge_id", ""),
                    "v26_attempt_id": stable_id("v26attempt", component_id),
                    "v25_component_id": component_id,
                    "v26_status": "blocked_missing_member_record",
                    "recommended_action": "repair_seed_member_index_before_testing",
                    "v5_source_key": edge.get("v5_source_key", ""),
                    "page_pair": f"{edge.get('prev_pdf_page','')}-{edge.get('pdf_page','')}",
                    "edge_layer": edge.get("v25_decision_layer", ""),
                    "edge_reason": edge.get("v25_reason", ""),
                    "constraint_flags": "missing_left_or_right_member_record",
                    "left_title": edge.get("prev_title", ""),
                    "right_title": edge.get("title", ""),
                    "left_image_path": edge.get("prev_image_path", ""),
                    "right_image_path": edge.get("image_path", ""),
                }
            )
            continue
        left, left_report = resolve_member_artifact(left_member, by_source_page, by_image_stem)
        right, right_report = resolve_member_artifact(right_member, by_source_page, by_image_stem)
        left_report = {**left_report, "v25_component_id": component_id}
        right_report = {**right_report, "v25_component_id": component_id}
        candidate_artifact_resolution_rows.extend([left_report, right_report])
        attempt, assembled_rows, constraints = assemble_pair(component_id, left, right, edge)
        attempt = {
            "candidate_scope": "row_label_dependency_review_candidate",
            "candidate_promotion_policy": "review_only_never_auto_promote",
            "source_v25_edge_id": edge.get("v25_edge_id", ""),
            **attempt,
        }
        if attempt.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}:
            attempt["recommended_action"] = "manual_review_candidate_pass_before_promoting"
        candidate_attempts.append(attempt)
        for constraint in constraints:
            candidate_constraint_rows.append(
                {
                    "candidate_scope": "row_label_dependency_review_candidate",
                    "source_v25_edge_id": edge.get("v25_edge_id", ""),
                    **constraint,
                }
            )
        for row in assembled_rows:
            candidate_assembled_jsonl_rows.append(
                {
                    "candidate_scope": "row_label_dependency_review_candidate",
                    "source_v25_edge_id": edge.get("v25_edge_id", ""),
                    **row,
                }
            )

    attempts.sort(key=lambda row: (str(row.get("v26_status")), str(row.get("source_pdf") or row.get("v5_source_key")), int_or_none(str(row.get("page_pair", "")).split("-", 1)[0]) or 10**9))
    review_rows.sort(key=lambda row: (str(row.get("recommended_action")), str(row.get("source_pdf") or row.get("v5_source_key")), int_or_none(str(row.get("page_pair", "")).split("-", 1)[0]) or 10**9))
    constraint_rows.sort(key=lambda row: (str(row.get("v25_component_id")), int(row.get("block_index") or 0)))
    candidate_attempts.sort(key=lambda row: (str(row.get("v26_status")), str(row.get("source_pdf") or row.get("v5_source_key")), int_or_none(str(row.get("page_pair", "")).split("-", 1)[0]) or 10**9))
    candidate_constraint_rows.sort(key=lambda row: (str(row.get("v25_component_id")), int(row.get("block_index") or 0)))
    artifact_resolution_rows.sort(
        key=lambda row: (
            str(row.get("resolution_status", "")),
            str(row.get("source_pdf") or row.get("v5_source_key")),
            int_or_none(row.get("pdf_page")) or 10**9,
            str(row.get("physical_page_id", "")),
        )
    )
    candidate_artifact_resolution_rows.sort(
        key=lambda row: (
            str(row.get("resolution_status", "")),
            str(row.get("source_pdf") or row.get("v5_source_key")),
            int_or_none(row.get("pdf_page")) or 10**9,
            str(row.get("physical_page_id", "")),
        )
    )

    attempt_fields = [
        "v26_attempt_id",
        "v25_component_id",
        "v26_status",
        "recommended_action",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "year",
        "province",
        "page_pair",
        "edge_layer",
        "edge_reason",
        "left_artifact_type",
        "right_artifact_type",
        "left_block_count",
        "right_block_count",
        "passed_block_count",
        "failed_block_count",
        "structural_status",
        "left_row_count_total",
        "right_row_count_total",
        "left_label_ratio",
        "right_label_ratio",
        "left_numeric_cell_count",
        "right_numeric_cell_count",
        "assembled_row_count",
        "value_constraint_status",
        "value_constraint_reason",
        "constraint_flags",
        "left_ocr_artifact",
        "right_ocr_artifact",
        "left_title",
        "right_title",
        "left_image_path",
        "right_image_path",
    ]
    constraint_fields = [
        "v26_constraint_id",
        "v25_component_id",
        "block_index",
        "left_block_id",
        "right_block_id",
        "left_section_name",
        "right_section_name",
        "left_row_count",
        "right_row_count",
        "row_count_match",
        "left_label_ratio",
        "right_label_ratio",
        "right_numeric_cell_count",
        "left_column_count",
        "right_column_count",
        "assembled_column_count",
        "column_order_status",
        "column_order_rule",
        "left_column_order",
        "right_column_order",
        "assembled_column_order",
        "constraint_status",
        "constraint_flags",
    ]
    row_sample_fields = [
        "v26_row_id",
        "v25_component_id",
        "block_index",
        "row_index",
        "row_label",
        "left_pdf_page",
        "right_pdf_page",
        "left_section_name",
        "right_section_name",
        "column_order_rule",
        "value_keys",
        "value_preview",
    ]
    candidate_attempt_fields = [
        "candidate_scope",
        "candidate_promotion_policy",
        "source_v25_edge_id",
        *attempt_fields,
    ]
    candidate_constraint_fields = [
        "candidate_scope",
        "source_v25_edge_id",
        *constraint_fields,
    ]
    candidate_row_sample_fields = [
        "candidate_scope",
        "source_v25_edge_id",
        *row_sample_fields,
    ]
    assembled_sample_rows = []
    for row in assembled_jsonl_rows:
        values = row.get("values", {})
        explicit_order = row.get("column_order", [])
        value_keys = list(explicit_order) if isinstance(explicit_order, list) else (list(values) if isinstance(values, dict) else [])
        assembled_sample_rows.append(
            {
                **{field: row.get(field, "") for field in row_sample_fields if field not in {"value_keys", "value_preview"}},
                "value_keys": "; ".join(value_keys[:18]),
                "value_preview": json.dumps({key: values[key] for key in value_keys[:10]}, ensure_ascii=False) if isinstance(values, dict) else "",
            }
        )
    candidate_assembled_sample_rows = []
    for row in candidate_assembled_jsonl_rows:
        values = row.get("values", {})
        explicit_order = row.get("column_order", [])
        value_keys = list(explicit_order) if isinstance(explicit_order, list) else (list(values) if isinstance(values, dict) else [])
        candidate_assembled_sample_rows.append(
            {
                **{field: row.get(field, "") for field in candidate_row_sample_fields if field not in {"value_keys", "value_preview"}},
                "value_keys": "; ".join(value_keys[:18]),
                "value_preview": json.dumps({key: values[key] for key in value_keys[:10]}, ensure_ascii=False) if isinstance(values, dict) else "",
            }
        )

    write_csv(OUT_DIR / "v26_row_assembly_attempts.csv", attempts, attempt_fields)
    write_csv(OUT_DIR / "v26_cross_page_constraints.csv", constraint_rows, constraint_fields)
    write_csv(OUT_DIR / "v26_row_assembly_review_queue.csv", review_rows, attempt_fields)
    write_csv(OUT_DIR / "v26_artifact_resolution_report.csv", artifact_resolution_rows, ARTIFACT_RESOLUTION_FIELDS)
    write_csv(OUT_DIR / "v26_candidate_row_assembly_attempts.csv", candidate_attempts, candidate_attempt_fields)
    write_csv(OUT_DIR / "v26_candidate_cross_page_constraints.csv", candidate_constraint_rows, candidate_constraint_fields)
    write_csv(OUT_DIR / "v26_candidate_artifact_resolution_report.csv", candidate_artifact_resolution_rows, ARTIFACT_RESOLUTION_FIELDS)
    complete_rows = [row for row in attempts if row.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}]
    partial_rows = [row for row in attempts if row.get("v26_status") == "partial_assembled_review"]
    candidate_test_pass_rows = [
        row
        for row in candidate_attempts
        if row.get("v26_status") in {"assembled_pass", "assembled_complete_value_review"}
    ]
    write_csv(OUT_DIR / "v26_structurally_complete_assemblies.csv", complete_rows, attempt_fields)
    write_csv(OUT_DIR / "v26_partial_assemblies.csv", partial_rows, attempt_fields)
    write_csv(OUT_DIR / "v26_candidate_test_pass_review_queue.csv", candidate_test_pass_rows, candidate_attempt_fields)
    write_csv(OUT_DIR / "v26_assembled_rows_sample.csv", assembled_sample_rows[:2000], row_sample_fields)
    write_csv(OUT_DIR / "v26_candidate_assembled_rows_sample.csv", candidate_assembled_sample_rows[:2000], candidate_row_sample_fields)
    with (OUT_DIR / "v26_assembled_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in assembled_jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "v26_candidate_assembled_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in candidate_assembled_jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    status_counts = Counter(str(row.get("v26_status")) for row in attempts)
    action_counts = Counter(str(row.get("recommended_action")) for row in attempts)
    resolution_counts = Counter(str(row.get("resolution_status")) for row in artifact_resolution_rows)
    resolution_method_counts = Counter(str(row.get("resolution_method")) for row in artifact_resolution_rows if row.get("resolution_method"))
    candidate_status_counts = Counter(str(row.get("v26_status")) for row in candidate_attempts)
    candidate_resolution_counts = Counter(str(row.get("resolution_status")) for row in candidate_artifact_resolution_rows)
    candidate_resolution_method_counts = Counter(str(row.get("resolution_method")) for row in candidate_artifact_resolution_rows if row.get("resolution_method"))
    summary_rows = [
        {"metric": "target_seed_pair_components", "value": len(target_components)},
        {"metric": "assembly_attempts", "value": len(attempts)},
        {"metric": "assembled_pass_components", "value": status_counts.get("assembled_pass", 0)},
        {"metric": "assembled_rows", "value": len(assembled_jsonl_rows)},
        {"metric": "review_or_blocked_components", "value": len(attempts) - status_counts.get("assembled_pass", 0)},
        {"metric": "candidate_row_label_dependency_edges", "value": len(candidate_edges)},
        {"metric": "candidate_assembly_attempts", "value": len(candidate_attempts)},
        {"metric": "candidate_test_pass_review_rows", "value": len(candidate_test_pass_rows)},
        {"metric": "candidate_assembled_rows", "value": len(candidate_assembled_jsonl_rows)},
    ]
    for key, value in sorted(status_counts.items()):
        summary_rows.append({"metric": f"status::{key}", "value": value})
    for key, value in sorted(action_counts.items()):
        summary_rows.append({"metric": f"recommended_action::{key}", "value": value})
    for key, value in sorted(resolution_counts.items()):
        summary_rows.append({"metric": f"artifact_resolution::{key}", "value": value})
    for key, value in sorted(resolution_method_counts.items()):
        summary_rows.append({"metric": f"artifact_resolution_method::{key}", "value": value})
    for key, value in sorted(candidate_status_counts.items()):
        summary_rows.append({"metric": f"candidate_status::{key}", "value": value})
    for key, value in sorted(candidate_resolution_counts.items()):
        summary_rows.append({"metric": f"candidate_artifact_resolution::{key}", "value": value})
    for key, value in sorted(candidate_resolution_method_counts.items()):
        summary_rows.append({"metric": f"candidate_artifact_resolution_method::{key}", "value": value})
    write_csv(OUT_DIR / "v26_summary.csv", summary_rows, ["metric", "value"])

    workbook_path = OUT_DIR / "v26_first_principles_row_assembly.xlsx"
    write_workbook(
        workbook_path,
        {
            "summary": (summary_rows, ["metric", "value"]),
            "assembly_attempts": (attempts, attempt_fields),
            "complete_structural": (complete_rows, attempt_fields),
            "partial_assemblies": (partial_rows, attempt_fields),
            "constraints": (constraint_rows, constraint_fields),
            "review_queue": (review_rows, attempt_fields),
            "artifact_resolution": (artifact_resolution_rows, ARTIFACT_RESOLUTION_FIELDS),
            "assembled_row_sample": (assembled_sample_rows[:2000], row_sample_fields),
            "candidate_attempts": (candidate_attempts, candidate_attempt_fields),
            "candidate_pass_review": (candidate_test_pass_rows, candidate_attempt_fields),
            "candidate_constraints": (candidate_constraint_rows, candidate_constraint_fields),
            "candidate_artifact_resolution": (candidate_artifact_resolution_rows, ARTIFACT_RESOLUTION_FIELDS),
            "candidate_row_sample": (candidate_assembled_sample_rows[:2000], candidate_row_sample_fields),
        },
    )

    bad_member = None
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"

    summary = {
        "v26_target_seed_pair_components": len(target_components),
        "v26_status_counts": dict(sorted(status_counts.items())),
        "v26_recommended_action_counts": dict(sorted(action_counts.items())),
        "v26_artifact_resolution_counts": dict(sorted(resolution_counts.items())),
        "v26_artifact_resolution_method_counts": dict(sorted(resolution_method_counts.items())),
        "v26_candidate_row_label_dependency_edges": len(candidate_edges),
        "v26_candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "v26_candidate_artifact_resolution_counts": dict(sorted(candidate_resolution_counts.items())),
        "v26_candidate_artifact_resolution_method_counts": dict(sorted(candidate_resolution_method_counts.items())),
        "v26_candidate_test_pass_review_rows": len(candidate_test_pass_rows),
        "v26_candidate_assembled_rows": len(candidate_assembled_jsonl_rows),
        "v26_assembled_rows": len(assembled_jsonl_rows),
        "v26_cross_page_constraint_rows": len(constraint_rows),
        "v26_workbook_zip_bad_member": bad_member,
    }
    (OUT_DIR / "v26_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
