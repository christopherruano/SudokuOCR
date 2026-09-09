#!/usr/bin/env python3
"""Refine logical sibling groupings from pairwise adjacency evidence.

v28 is intentionally conservative: most same-title consecutive logical groups
are review-only. This v29 pass looks inside those review-only runs and recovers
bounded sibling sequences where adjacent pages explicitly vary by religion/part
while sharing the same table frame. These are logical collection-only groups,
never physical merges.

No OCR/API work is performed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V27_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_manifest_v27"
V28_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_operational_manifest_v28"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_first_principles_sibling_refinement_v29"

MAX_REFINED_USE_NOW_PAGES = 12
MIN_STRONG_TITLE_SIM = 0.80


RELIGION_PATTERNS = [
    ("all_religions", r"\ball\s+religions?\b"),
    ("total_population", r"\btotal\s+population\b"),
    ("hindu", r"\bhindus?\b|\bbrahmanic\b"),
    ("musalman", r"\bmusalman(s)?\b|\bmuhamm?adan(s)?\b|\bmuslim(s)?\b"),
    ("christian", r"\bchristian(s)?\b"),
    ("animistic", r"\banimistic\b|\banimist(s)?\b"),
    ("jew", r"\bjew(s)?\b"),
    ("jain", r"\bjain(s)?\b"),
    ("sikh", r"\bsikh(s)?\b"),
    ("buddhist", r"\bbuddhist(s)?\b|\bbudhist\b"),
    ("parsi", r"\bparsi(s)?\b|\bzoroastrian(s)?\b"),
    ("aboriginal", r"\baboriginal\b"),
    ("unspecified", r"\bunspecified\b"),
    ("brahmo", r"\bbrahmo\b|\bbramho\b"),
    ("arya", r"\barya\b"),
]

STATUS_PATTERNS = [
    ("unmarried", r"\bunmarried\b|\bsingle\b"),
    ("married", r"\bmarried\b"),
    ("widowed", r"\bwidowed\b|\bwidowers?\b|\bwidows?\b"),
    ("divorced", r"\bdivorced\b|\bdivorved\b"),
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


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def float_or_none(value: object) -> float | None:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


def uniq(values: Iterable[object], limit: int = 30) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = compact(value)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return out[:limit] + [f"...(+{len(out) - limit})"]
    return out


def uniq_join(values: Iterable[object], limit: int = 30) -> str:
    return "; ".join(uniq(values, limit=limit))


def extract_terms(title: str, patterns: list[tuple[str, str]]) -> set[str]:
    text = clean_text(title)
    out: set[str] = set()
    for key, pattern in patterns:
        if re.search(pattern, text):
            out.add(key)
    return out


def extract_part_terms(title: str) -> set[str]:
    text = clean_text(title)
    out = {f"part_{m.group(1)}" for m in re.finditer(r"\bpart\s+([a-z])\b", text)}
    out |= {f"section_{m.group(1)}" for m in re.finditer(r"\b([a-z])\s+(hindu|musalman|christian|jew|animistic|parsi|sikh|jain|buddhist)\b", text)}
    return out


def strip_variant_terms(title: str) -> str:
    text = clean_text(title)
    text = re.sub(r"\b(contd|continued|continuation|concluded|concld|concl|cld)\b", " ", text)
    text = re.sub(r"\bpart\s+[a-z]\b", " ", text)
    for _, pattern in [*RELIGION_PATTERNS, *STATUS_PATTERNS]:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\b(section|part|population|persons|males|females)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_jaccard(a: str, b: str) -> float:
    aa = {t for t in strip_variant_terms(a).split() if len(t) > 1}
    bb = {t for t in strip_variant_terms(b).split() if len(t) > 1}
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def page_range_from_pages(pages: list[int]) -> str:
    return "; ".join(str(page) for page in pages)


def edge_variant_info(edge: dict[str, str]) -> dict[str, object]:
    prev_title = edge.get("prev_title", "")
    title = edge.get("title", "")
    prev_religions = extract_terms(prev_title, RELIGION_PATTERNS)
    religions = extract_terms(title, RELIGION_PATTERNS)
    prev_statuses = extract_terms(prev_title, STATUS_PATTERNS)
    statuses = extract_terms(title, STATUS_PATTERNS)
    prev_parts = extract_part_terms(prev_title)
    parts = extract_part_terms(title)
    title_sim = float_or_none(edge.get("title_similarity")) or 0.0
    base_sim = token_jaccard(prev_title, title)

    variant_category = "none"
    strong = False
    reason_parts: list[str] = []
    if prev_religions and religions and prev_religions != religions:
        variant_category = "religion_transition"
        strong = True
        reason_parts.append("explicit_religion_variant_transition")
    elif prev_religions and religions and prev_religions == religions:
        variant_category = "religion_continuation"
        strong = True
        reason_parts.append("explicit_religion_continuation")
    elif prev_parts and parts and prev_parts != parts:
        variant_category = "part_transition"
        strong = True
        reason_parts.append("explicit_part_variant_transition")
    elif prev_statuses and statuses and prev_statuses != statuses:
        variant_category = "status_transition"
        strong = True
        reason_parts.append("explicit_status_variant_transition")
    elif (prev_religions or religions or prev_parts or parts) and title_sim >= 0.90:
        variant_category = "one_sided_variant_review"
        reason_parts.append("one_sided_variant_marker_review")

    if title_sim < MIN_STRONG_TITLE_SIM or base_sim < 0.50:
        strong = False
        reason_parts.append(f"weak_base_or_title_similarity:title_{title_sim:.2f}:base_{base_sim:.2f}")

    return {
        "variant_category": variant_category,
        "strong_variant_edge": strong,
        "prev_variant_terms": "; ".join(sorted(prev_religions | prev_statuses | prev_parts)),
        "variant_terms": "; ".join(sorted(religions | statuses | parts)),
        "title_similarity_float": title_sim,
        "base_similarity": base_sim,
        "variant_reason": "; ".join(reason_parts),
    }


def build_refined_edges(edge_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for edge in edge_rows:
        if edge.get("v27_edge_class") != "logical_collection_edge_only":
            continue
        if edge.get("page_gap") != "1" or edge.get("missing_pages"):
            continue
        info = edge_variant_info(edge)
        if info["variant_category"] == "none":
            continue
        out.append(
            {
                "v29_edge_id": stable_id("v29edge", edge.get("v27_edge_id", "")),
                "v27_edge_id": edge.get("v27_edge_id", ""),
                "v25_edge_id": edge.get("v25_edge_id", ""),
                "v5_source_key": edge.get("v5_source_key", ""),
                "prev_pdf_page": edge.get("prev_pdf_page", ""),
                "pdf_page": edge.get("pdf_page", ""),
                "page_gap": edge.get("page_gap", ""),
                "variant_category": info["variant_category"],
                "strong_variant_edge": "yes" if info["strong_variant_edge"] else "no",
                "prev_variant_terms": info["prev_variant_terms"],
                "variant_terms": info["variant_terms"],
                "title_similarity": edge.get("title_similarity", ""),
                "base_similarity": f"{info['base_similarity']:.3f}",
                "variant_reason": info["variant_reason"],
                "prev_title": edge.get("prev_title", ""),
                "title": edge.get("title", ""),
                "prev_image_path": edge.get("prev_image_path", ""),
                "image_path": edge.get("image_path", ""),
            }
        )
    out.sort(key=lambda r: (str(r.get("v5_source_key")), int_or_none(r.get("prev_pdf_page")) or 10**9))
    return out


def build_refined_groups(edges: list[dict[str, object]]) -> list[dict[str, object]]:
    strong_edges = [e for e in edges if e.get("strong_variant_edge") == "yes"]
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for edge in strong_edges:
        by_source[str(edge.get("v5_source_key", ""))].append(edge)

    groups: list[dict[str, object]] = []
    for source, source_edges in by_source.items():
        source_edges.sort(key=lambda e: int_or_none(e.get("prev_pdf_page")) or 10**9)
        current: list[dict[str, object]] = []
        last_page: int | None = None
        for edge in source_edges:
            prev_page = int_or_none(edge.get("prev_pdf_page"))
            page = int_or_none(edge.get("pdf_page"))
            if prev_page is None or page is None:
                continue
            if not current or prev_page == last_page:
                current.append(edge)
            else:
                if current:
                    groups.append(make_group(source, current))
                current = [edge]
            last_page = page
        if current:
            groups.append(make_group(source, current))

    groups = [g for g in groups if int(g.get("n_pdf_pages") or 0) >= 2 and int(g.get("strong_transition_edge_count") or 0) >= 1]
    groups.sort(key=lambda r: (str(r.get("v29_tier")), str(r.get("v5_source_key")), int_or_none(str(r.get("page_ranges")).split(";", 1)[0]) or 10**9))
    return groups


def make_group(source: str, edges: list[dict[str, object]]) -> dict[str, object]:
    pages: list[int] = []
    for edge in edges:
        prev_page = int_or_none(edge.get("prev_pdf_page"))
        page = int_or_none(edge.get("pdf_page"))
        if prev_page is not None:
            pages.append(prev_page)
        if page is not None:
            pages.append(page)
    pages = sorted(set(pages))
    categories = [str(e.get("variant_category")) for e in edges]
    transition_count = sum(1 for c in categories if c.endswith("_transition"))
    n_pages = len(pages)
    if n_pages > MAX_REFINED_USE_NOW_PAGES:
        tier = "review_large_refined_sibling"
        action = "manual_review_large_refined_sibling_before_use"
    elif transition_count == 0:
        tier = "review_continuation_only_variant_sequence"
        action = "manual_review_before_logical_grouping"
    else:
        tier = "use_now_refined_logical_sibling"
        action = "use_as_logical_collection_only"
    return {
        "v29_group_id": stable_id("v29sibling", source, page_range_from_pages(pages), uniq_join(categories)),
        "v29_tier": tier,
        "assembly_policy": "logical_collection_only" if tier == "use_now_refined_logical_sibling" else "logical_collection_after_review",
        "recommended_action": action,
        "v5_source_key": source,
        "page_ranges": page_range_from_pages(pages),
        "n_pdf_pages": n_pages,
        "edge_count": len(edges),
        "strong_transition_edge_count": transition_count,
        "variant_categories": uniq_join(categories),
        "variant_sequence": uniq_join([e.get("prev_variant_terms", "") for e in edges] + [edges[-1].get("variant_terms", "")], limit=40),
        "title_sequence": uniq_join([e.get("prev_title", "") for e in edges] + [edges[-1].get("title", "")], limit=20),
        "source_edge_ids": uniq_join([e.get("v29_edge_id", "") for e in edges], limit=50),
        "physical_merge_allowed": "no",
        "logical_collection_only": "yes",
        "missing_pages": "",
        "review_flags": "refined_pairwise_variant_sibling; no_physical_merge",
        "reason": "adjacent pairwise logical edges with explicit religion/part/status variant evidence",
        "image_paths": uniq_join([e.get("prev_image_path", "") for e in edges] + [edges[-1].get("image_path", "")], limit=10),
    }


def parse_page_set(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for piece in str(value or "").split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if re.fullmatch(r"\d+", piece):
            out.append(int(piece))
        else:
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", piece)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                out.extend(range(min(a, b), max(a, b) + 1))
    return tuple(sorted(set(out)))


def annotate_new_vs_existing(groups: list[dict[str, object]], v28_use: list[dict[str, str]]) -> list[dict[str, object]]:
    existing = {
        (row.get("v5_source_key") or row.get("source_pdf") or "", parse_page_set(row.get("page_ranges", "")))
        for row in v28_use
    }
    out = []
    for group in groups:
        key = (str(group.get("v5_source_key", "")), parse_page_set(str(group.get("page_ranges", ""))))
        row = dict(group)
        row["already_in_v28_use_now"] = "yes" if key in existing else "no"
        if row["already_in_v28_use_now"] == "yes":
            row["v29_tier"] = "already_v28_use_now"
            row["recommended_action"] = "already_available_in_v28_use_now"
        out.append(row)
    return out


def build_v29_manifest(v28_manifest: list[dict[str, str]], groups: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [dict(row) for row in v28_manifest]
    existing_keys = {
        (row.get("v5_source_key") or row.get("source_pdf") or "", parse_page_set(row.get("page_ranges", "")), row.get("assembly_layer", ""))
        for row in v28_manifest
        if row.get("v28_tier") == "use_now"
    }
    for group in groups:
        if group.get("v29_tier") != "use_now_refined_logical_sibling":
            continue
        key = (str(group.get("v5_source_key", "")), parse_page_set(str(group.get("page_ranges", ""))), "logical_collection")
        if key in existing_keys:
            continue
        rows.append(
            {
                "v28_group_id": group.get("v29_group_id", ""),
                "v28_tier": "use_now",
                "v28_subgroup": "logical_refined_sibling_bounded",
                "assembly_layer": "logical_collection",
                "assembly_policy": "logical_collection_only",
                "reason": group.get("reason", ""),
                "v5_source_key": group.get("v5_source_key", ""),
                "source_pdf": "",
                "source_pdf_id": "",
                "page_ranges": group.get("page_ranges", ""),
                "n_physical_pages": group.get("n_pdf_pages", ""),
                "physical_merge_allowed": "no",
                "logical_collection_only": "yes",
                "missing_pages": "",
                "review_flags": group.get("review_flags", ""),
                "primary_title": group.get("title_sequence", ""),
                "source_evidence": "v29_pairwise_refined_sibling_edges",
            }
        )
    rows.sort(key=lambda r: (str(r.get("assembly_layer")), str(r.get("v28_tier")), str(r.get("v5_source_key") or r.get("source_pdf")), int_or_none(str(r.get("page_ranges")).split(";", 1)[0].split("-", 1)[0]) or 10**9))
    return rows


def validate(groups: list[dict[str, object]], manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    failures: list[tuple[str, list[str]]] = []
    bad_use_pages = [
        str(row.get("v29_group_id"))
        for row in groups
        if row.get("v29_tier") == "use_now_refined_logical_sibling"
        and ((int(row.get("n_pdf_pages") or 0) > MAX_REFINED_USE_NOW_PAGES) or row.get("missing_pages"))
    ]
    failures.append(("refined_use_now_groups_are_bounded_and_gapless", bad_use_pages))
    bad_physical = [
        str(row.get("v29_group_id"))
        for row in groups
        if row.get("physical_merge_allowed") != "no"
    ]
    failures.append(("refined_sibling_groups_never_allow_physical_merge", bad_physical))
    no_transition = [
        str(row.get("v29_group_id"))
        for row in groups
        if row.get("v29_tier") == "use_now_refined_logical_sibling"
        and int(row.get("strong_transition_edge_count") or 0) < 1
    ]
    failures.append(("refined_use_now_groups_have_explicit_variant_transition", no_transition))
    manifest_bad = [
        str(row.get("v28_group_id"))
        for row in manifest
        if row.get("v28_tier") == "use_now"
        and row.get("assembly_layer") == "logical_collection"
        and (row.get("physical_merge_allowed") != "no" or row.get("missing_pages"))
    ]
    failures.append(("manifest_use_now_logical_groups_are_collection_only_and_gapless", manifest_bad))
    return [
        {
            "check": name,
            "status": "pass" if not bad else "fail",
            "failure_count": len(bad),
            "details": json.dumps(bad[:50], sort_keys=True),
        }
        for name, bad in failures
    ]


def write_csvs_and_workbook(refined_edges, refined_groups, v29_manifest, audit) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_fields = [
        "v29_edge_id",
        "v27_edge_id",
        "v25_edge_id",
        "v5_source_key",
        "prev_pdf_page",
        "pdf_page",
        "page_gap",
        "variant_category",
        "strong_variant_edge",
        "prev_variant_terms",
        "variant_terms",
        "title_similarity",
        "base_similarity",
        "variant_reason",
        "prev_title",
        "title",
        "prev_image_path",
        "image_path",
    ]
    group_fields = [
        "v29_group_id",
        "v29_tier",
        "assembly_policy",
        "recommended_action",
        "already_in_v28_use_now",
        "v5_source_key",
        "page_ranges",
        "n_pdf_pages",
        "edge_count",
        "strong_transition_edge_count",
        "variant_categories",
        "variant_sequence",
        "title_sequence",
        "source_edge_ids",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "review_flags",
        "reason",
        "image_paths",
    ]
    manifest_fields = [
        "v28_group_id",
        "v28_tier",
        "v28_subgroup",
        "assembly_layer",
        "assembly_policy",
        "reason",
        "v5_source_key",
        "source_pdf",
        "source_pdf_id",
        "page_ranges",
        "n_physical_pages",
        "physical_merge_allowed",
        "logical_collection_only",
        "missing_pages",
        "review_flags",
        "primary_title",
        "source_evidence",
    ]
    audit_fields = ["check", "status", "failure_count", "details"]
    use_now_groups = [row for row in refined_groups if row.get("v29_tier") == "use_now_refined_logical_sibling"]
    new_use_now_groups = [row for row in use_now_groups if row.get("already_in_v28_use_now") == "no"]
    review_groups = [row for row in refined_groups if str(row.get("v29_tier", "")).startswith("review")]
    manifest_use_now = [row for row in v29_manifest if row.get("v28_tier") == "use_now"]

    write_csv(OUT_DIR / "v29_refined_sibling_edges.csv", refined_edges, edge_fields)
    write_csv(OUT_DIR / "v29_refined_sibling_groups.csv", refined_groups, group_fields)
    write_csv(OUT_DIR / "v29_new_use_now_refined_siblings.csv", new_use_now_groups, group_fields)
    write_csv(OUT_DIR / "v29_review_refined_siblings.csv", review_groups, group_fields)
    write_csv(OUT_DIR / "v29_operational_manifest.csv", v29_manifest, manifest_fields)
    write_csv(OUT_DIR / "v29_use_now_groups.csv", manifest_use_now, manifest_fields)
    write_csv(OUT_DIR / "v29_audit.csv", audit, audit_fields)

    summary_rows = [
        {"metric": "refined_variant_edges", "value": len(refined_edges)},
        {"metric": "strong_refined_variant_edges", "value": sum(1 for e in refined_edges if e.get("strong_variant_edge") == "yes")},
        {"metric": "refined_sibling_groups", "value": len(refined_groups)},
        {"metric": "use_now_refined_sibling_groups", "value": len(use_now_groups)},
        {"metric": "new_use_now_refined_sibling_groups", "value": len(new_use_now_groups)},
        {"metric": "review_refined_sibling_groups", "value": len(review_groups)},
        {"metric": "v29_operational_manifest_rows", "value": len(v29_manifest)},
        {"metric": "v29_use_now_manifest_rows", "value": len(manifest_use_now)},
        {"metric": "audit_failure_count", "value": sum(int(row["failure_count"]) for row in audit)},
    ]
    for key, value in sorted(Counter(row["variant_category"] for row in refined_edges).items()):
        summary_rows.append({"metric": f"edge_category::{key}", "value": value})
    for key, value in sorted(Counter(row["v29_tier"] for row in refined_groups).items()):
        summary_rows.append({"metric": f"group_tier::{key}", "value": value})
    write_csv(OUT_DIR / "v29_summary.csv", summary_rows, ["metric", "value"])

    workbook_path = OUT_DIR / "v29_sibling_refinement_manifest.xlsx"
    workbook = xlsxwriter.Workbook(str(workbook_path))
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, rows, fields in [
        ("summary", summary_rows, ["metric", "value"]),
        ("new_use_now_siblings", new_use_now_groups, group_fields),
        ("all_refined_groups", refined_groups, group_fields),
        ("refined_edges", refined_edges, edge_fields),
        ("v29_use_now", manifest_use_now, manifest_fields),
        ("audit", audit, audit_fields),
    ]:
        ws = workbook.add_worksheet(sheet_name[:31])
        for col, field in enumerate(fields):
            ws.write(0, col, field, header)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, field in enumerate(fields):
                fmt = wrap if any(token in field for token in ["title", "reason", "flags", "image", "sequence"]) else None
                ws.write(r_idx, c_idx, row.get(field, ""), fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(1, len(rows)), max(0, len(fields) - 1))
        for col, field in enumerate(fields):
            ws.set_column(col, col, 46 if any(token in field for token in ["title", "reason", "flags", "image", "sequence"]) else 18)
    workbook.close()

    bad_member = None
    try:
        with zipfile.ZipFile(workbook_path) as zf:
            bad_member = zf.testzip()
    except Exception as exc:
        bad_member = f"{type(exc).__name__}: {exc}"

    summary = {
        "v29_refined_variant_edges": len(refined_edges),
        "v29_strong_refined_variant_edges": sum(1 for e in refined_edges if e.get("strong_variant_edge") == "yes"),
        "v29_refined_sibling_groups": len(refined_groups),
        "v29_use_now_refined_sibling_groups": len(use_now_groups),
        "v29_new_use_now_refined_sibling_groups": len(new_use_now_groups),
        "v29_review_refined_sibling_groups": len(review_groups),
        "v29_operational_manifest_rows": len(v29_manifest),
        "v29_use_now_manifest_rows": len(manifest_use_now),
        "v29_edge_category_counts": dict(sorted(Counter(row["variant_category"] for row in refined_edges).items())),
        "v29_group_tier_counts": dict(sorted(Counter(row["v29_tier"] for row in refined_groups).items())),
        "v29_audit_failure_count": sum(int(row["failure_count"]) for row in audit),
        "v29_workbook_zip_bad_member": bad_member,
    }
    (OUT_DIR / "v29_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    edge_rows = read_csv(V27_DIR / "v27_pairwise_edge_manifest.csv")
    v28_use = read_csv(V28_DIR / "v28_use_now_groups.csv")
    v28_manifest = read_csv(V28_DIR / "v28_operational_manifest.csv")
    refined_edges = build_refined_edges(edge_rows)
    refined_groups = annotate_new_vs_existing(build_refined_groups(refined_edges), v28_use)
    v29_manifest = build_v29_manifest(v28_manifest, refined_groups)
    audit = validate(refined_groups, v29_manifest)
    write_csvs_and_workbook(refined_edges, refined_groups, v29_manifest, audit)


if __name__ == "__main__":
    main()
