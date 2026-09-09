#!/usr/bin/env python3
"""Build corpus-wide image/PDF-page provenance and missing-page workbooks.

The digitized table images are named with SHA1(pdf_path + ":" + pdf_page_number).
That lets us recover exact PDF page provenance for every rendered PNG as long as
we know the original source PDF path and the candidate table page window.

Outputs:
  page_provenance_dictionary.csv/json  one row per local image
  table_range_completeness.csv         one row per TOC/table range
  source_pdf_completeness.csv          one row per source PDF
  pdf_page_inventory.csv               one row per expected PDF page in ranges
  missing_pdf_pages.csv                one row per expected-but-undigitized page
  page_provenance_workbook.xlsx        human-browsable workbook
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import xlsxwriter
except ImportError:  # pragma: no cover
    xlsxwriter = None


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_ROOTS = [
    ROOT / "export/digitized_tables_referenced/digitized_tables",
    ROOT / "export/never_ocrd_images/digitized_tables",
    ROOT / "export/suspicious_review/digitized_tables",
    ROOT / "export/local_research_bundle/final_core_prep/review_assets/digitized_tables",
    ROOT / "export/ocr_results_referenced/digitized_tables",
]
DEFAULT_OUT_DIR = ROOT / "export_final/page_provenance"
DEFAULT_OCR_ROOT = ROOT / "export/ocr_results_referenced/digitized_tables"
DEFAULT_MANIFESTS = [
    ROOT / "export/cluster_sync/organized_manifest.jsonl",
    ROOT / "export/local_research_bundle/organized_manifest.jsonl",
]
DEFAULT_CATALOGS = [
    ROOT / "export/cluster_sync/export_inference_final_core_curated_v2/source_catalog.csv",
    ROOT / "export/cluster_sync/export_inference_tail_rescue_unreviewed_v2/source_catalog.csv",
    ROOT / "export/cluster_sync/export_assembled_final_core_plus_tail_strong_v2_facts/source_catalog.csv",
    ROOT / "export_final/export_assembled_all_read_strong_facts/source_catalog.csv",
    ROOT / "export_final/team_ready/catalogs/all_tables_full_metadata.csv",
]

HEX40 = re.compile(r"([0-9a-f]{40})", re.I)
TABLE_DIR = re.compile(r"([0-9a-f]{16})_t(\d{4})")
csv.field_size_limit(10**8)


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def clean_text(value: Any, limit: int = 32000) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text[:limit]


def basename_no_ext(path: str) -> str:
    base = os.path.basename(path or "")
    return os.path.splitext(base)[0]


def image_hash_from_path(path: str) -> str:
    m = HEX40.search(path or "")
    return m.group(1).lower() if m else ""


def table_key_from_path(path: str) -> tuple[str, str, str]:
    m = TABLE_DIR.search(path or "")
    if not m:
        return "", "", ""
    doc_hash, table_index = m.group(1).lower(), m.group(2)
    return f"{doc_hash}_t{table_index}", doc_hash, table_index


def split_paths(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    if " | " in value:
        return [p.strip() for p in value.split(" | ") if p.strip()]
    if ";" in value:
        return [p.strip() for p in value.split(";") if p.strip()]
    return [value]


def int_or_blank(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def compress_ints(values: list[int], max_parts: int = 40) -> str:
    if not values:
        return ""
    values = sorted(set(values))
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    if len(ranges) > max_parts:
        return ",".join(ranges[:max_parts]) + f",...(+{len(ranges) - max_parts} ranges)"
    return ",".join(ranges)


def missing_page_priority(title: str) -> str:
    text = (title or "").lower()
    low_terms = ["blank", "no table found", "introduction", "chapter", "diagram"]
    high_terms = ["table vii", "table-vii", "table vih", "age", "civil condition", "religion", "mortality", "birth", "death"]
    medium_terms = ["literacy", "education", "caste", "occupation", "subsidiary table", "imperial table"]
    if any(term in text for term in low_terms) and not any(term in text for term in high_terms):
        return "low_check_range"
    if any(term in text for term in high_terms):
        return "high_target"
    if any(term in text for term in medium_terms):
        return "medium_target"
    return "review"


def table_vii_signal_text(text: str) -> str:
    text_l = (text or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", text_l)
    compact = re.sub(r"[^a-z0-9]+", "", text_l)
    if re.search(r"\btable\s*(?:no\s*)?(?:vii|7)\b", normalized):
        return "explicit_table_vii"
    if any(token in compact for token in ["tablevii", "imperialtablevii", "finalcensustablevii"]):
        return "explicit_table_vii"
    # Common OCR confusions for VII in this corpus: VH, VIH, VIE, VJI.
    if any(token in compact for token in ["tablevh", "tablevih", "tablevie", "tablevji"]):
        return "probable_table_vii_ocr_variant"
    if ("age" in text_l and "sex" in text_l and ("civil" in text_l or "condition" in text_l)):
        return "age_sex_civil_condition_signal"
    return ""


def table_vii_signal(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k, "")) for k in [
        "title", "table_dir_name", "canonical_topic", "catalog_table_families", "catalog_table_ids", "titles", "table_keys"
    ])
    return table_vii_signal_text(text)


def local_ocr_json_path(table_dir_name: str, image_hash: str, ocr_root: Path) -> str:
    if not table_dir_name or not image_hash:
        return ""
    path = ocr_root / table_dir_name / f"{image_hash}_oneshot.json"
    return rel(path) if path.exists() else ""


def page_hash(pdf_path: str, page_number: int) -> str:
    return hashlib.sha1(f"{pdf_path}:{page_number}".encode("utf-8")).hexdigest()


def exact_pdf_page(image_hash: str, pdf_paths: list[str], start: int | None, end: int | None, max_bruteforce_pages: int) -> tuple[int | None, str]:
    if not image_hash or not pdf_paths:
        return None, "no_pdf_or_hash"
    ranges: list[tuple[int, int, str]] = []
    if start is not None and end is not None and start > 0 and end >= start:
        ranges.append((start, end, "hash_match_in_resolved_range"))
    ranges.append((1, max_bruteforce_pages, "hash_match_bruteforce"))
    seen_ranges = set()
    for lo, hi, method in ranges:
        key = (lo, hi)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        for pdf in pdf_paths:
            for page in range(lo, hi + 1):
                if page_hash(pdf, page) == image_hash:
                    return page, method
    return None, "hash_no_match"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def read_catalog_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["_catalog_path"] = rel(path)
                rows.append(row)
    return rows


def build_manifest_indexes(paths: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    range_by_key: dict[str, dict[str, Any]] = {}
    pdfs_by_doc: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        for row in load_jsonl(path):
            source_image = row.get("source_image") or ""
            image_hash = image_hash_from_path(source_image or row.get("result_json", ""))
            table_key, doc_hash, table_index = table_key_from_path(source_image or row.get("result_json", ""))
            queue = row.get("queue_row") or {}
            pdf = row.get("digitized_source_file") or queue.get("file") or ""
            start = int_or_blank(queue.get("resolved_start_page"))
            end = int_or_blank(queue.get("resolved_end_page"))
            item = {
                "source": rel(path),
                "image_hash": image_hash,
                "table_key": table_key,
                "doc_hash": doc_hash,
                "table_index": table_index,
                "source_pdf": pdf,
                "source_pdf_filename": os.path.basename(pdf),
                "source_image_original": source_image,
                "source_json_original": row.get("result_json", ""),
                "manifest_classification": row.get("classification", ""),
                "province": row.get("province", ""),
                "year": row.get("year", ""),
                "title": row.get("title", "") or queue.get("title", ""),
                "queue_title": queue.get("title", ""),
                "canonical_topic": queue.get("canonical_topic", ""),
                "relevance_tier": queue.get("relevance_tier", ""),
                "range_status": queue.get("range_status", ""),
                "resolved_start_page": start,
                "resolved_end_page": end,
            }
            if image_hash:
                by_hash[image_hash].append(item)
            if doc_hash and pdf:
                pdfs_by_doc[doc_hash].add(pdf)
            if table_key:
                existing = range_by_key.get(table_key, {})
                merged = {**existing, **{k: v for k, v in item.items() if v not in ("", None)}}
                range_by_key[table_key] = merged
    return by_hash, range_by_key, pdfs_by_doc


def add_catalog_reference(refs: dict[str, list[dict[str, Any]]], image_path: str, row: dict[str, str], role: str, member_idx: int | None = None) -> None:
    image_hash = image_hash_from_path(image_path)
    if not image_hash:
        return
    queue: dict[str, Any] = {}
    if row.get("queue_row_json"):
        try:
            queue = json.loads(row.get("queue_row_json") or "{}")
        except Exception:
            queue = {}
    refs[image_hash].append({
        "role": role,
        "member_index": "" if member_idx is None else member_idx,
        "table_id": row.get("table_id", ""),
        "layer": row.get("layer", ""),
        "dataset_dir": row.get("dataset_dir", ""),
        "catalog_path": row.get("_catalog_path", ""),
        "source_catalog_path": row.get("source_catalog_path", ""),
        "source_file": row.get("source_file", "") or queue.get("file", ""),
        "province": row.get("province", ""),
        "year": row.get("year", ""),
        "title": row.get("title", "") or row.get("queue_title", ""),
        "table_family": row.get("table_family", ""),
        "quality_tier": row.get("quality_tier", ""),
        "confidence_level": row.get("confidence_level", ""),
        "effective_quality_status": row.get("effective_quality_status", ""),
        "keep_for_inference": row.get("keep_for_inference", ""),
        "manual_final_decision": row.get("manual_final_decision", ""),
        "assembled_group_id": row.get("assembled_group_id", ""),
        "assembled_n_pages": row.get("assembled_n_pages", ""),
        "resolved_start_page": int_or_blank(queue.get("resolved_start_page")),
        "resolved_end_page": int_or_blank(queue.get("resolved_end_page")),
        "range_status": queue.get("range_status", ""),
    })


def build_catalog_refs(catalog_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog_rows:
        add_catalog_reference(refs, row.get("source_image", "") or row.get("source_image_original", ""), row, "primary")
        for idx, image_path in enumerate(split_paths(row.get("assembled_member_source_images", "")), 1):
            add_catalog_reference(refs, image_path, row, "assembled_member", idx)
    return refs


def local_image_rows(image_roots: list[Path]) -> tuple[list[dict[str, Any]], int]:
    by_hash: dict[str, dict[str, Any]] = {}
    path_count = 0
    for image_root in image_roots:
        if not image_root.exists():
            continue
        for path in sorted(image_root.glob("*/*.png")):
            path_count += 1
            image_hash = path.stem.lower()
            table_key, doc_hash, table_index = table_key_from_path(str(path))
            existing = by_hash.get(image_hash)
            local_path = rel(path)
            if existing:
                existing["all_source_image_local_paths"].append(local_path)
                if "digitized_tables_referenced" in local_path:
                    existing["source_image_local"] = local_path
                    existing["table_dir_name"] = path.parent.name
                    existing["table_key"] = table_key or existing.get("table_key", "")
                    existing["doc_hash"] = doc_hash or existing.get("doc_hash", "")
                    existing["table_index"] = table_index or existing.get("table_index", "")
                continue
            by_hash[image_hash] = {
                "image_hash": image_hash,
                "source_image_local": local_path,
                "all_source_image_local_paths": [local_path],
                "table_dir_name": path.parent.name,
                "table_key": table_key,
                "doc_hash": doc_hash,
                "table_index": table_index,
            }
    rows = []
    for row in by_hash.values():
        row["local_path_count"] = len(row["all_source_image_local_paths"])
        row["all_source_image_local_paths"] = " | ".join(row["all_source_image_local_paths"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["image_hash"]), path_count


def coalesce(values: list[Any]) -> str:
    out = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text not in out:
            out.append(text)
    return " | ".join(out)


def choose_range(local: dict[str, Any], manifests: list[dict[str, Any]], refs: list[dict[str, Any]], range_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(manifests)
    candidates.extend(refs)
    if local.get("table_key") and local["table_key"] in range_by_key:
        candidates.append(range_by_key[local["table_key"]])
    best: dict[str, Any] = {}
    for cand in candidates:
        if cand.get("source_pdf") or cand.get("source_file"):
            best.setdefault("source_pdf", cand.get("source_pdf") or cand.get("source_file"))
        for key in [
            "source_pdf_filename", "resolved_start_page", "resolved_end_page", "range_status",
            "province", "year", "title", "queue_title", "canonical_topic", "relevance_tier",
            "manifest_classification", "source_json_original", "source_image_original",
        ]:
            value = cand.get(key)
            if value not in ("", None) and key not in best:
                best[key] = value
    return best


def build_page_dictionary(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifests_by_hash, range_by_key, pdfs_by_doc = build_manifest_indexes(args.manifest_jsonl)
    catalog_rows = read_catalog_rows(args.source_catalog)
    catalog_refs = build_catalog_refs(catalog_rows)
    images, local_path_count = local_image_rows(args.image_roots)

    page_rows = []
    for local in images:
        image_hash = local["image_hash"]
        manifest_hits = manifests_by_hash.get(image_hash, [])
        catalog_hits = catalog_refs.get(image_hash, [])
        range_meta = choose_range(local, manifest_hits, catalog_hits, range_by_key)
        source_pdf_candidates = []
        if range_meta.get("source_pdf"):
            source_pdf_candidates.append(range_meta["source_pdf"])
        source_pdf_candidates.extend(sorted(pdfs_by_doc.get(local.get("doc_hash", ""), [])))
        for hit in manifest_hits:
            if hit.get("source_pdf"):
                source_pdf_candidates.append(hit["source_pdf"])
        for hit in catalog_hits:
            if hit.get("source_file"):
                source_pdf_candidates.append(hit["source_file"])
        source_pdf_candidates = list(dict.fromkeys([p for p in source_pdf_candidates if p]))
        start = int_or_blank(range_meta.get("resolved_start_page"))
        end = int_or_blank(range_meta.get("resolved_end_page"))
        page_number, method = exact_pdf_page(image_hash, source_pdf_candidates, start, end, args.max_bruteforce_pages)
        chosen_pdf = ""
        if page_number is not None:
            for pdf in source_pdf_candidates:
                if page_hash(pdf, page_number) == image_hash:
                    chosen_pdf = pdf
                    break
        if not chosen_pdf and source_pdf_candidates:
            chosen_pdf = source_pdf_candidates[0]
        if page_number is not None and method.startswith("hash_match"):
            next_action = "ready_exact_pdf_page_provenance"
        elif not chosen_pdf:
            next_action = "recover_source_pdf_for_doc_hash"
        else:
            next_action = "investigate_hash_mismatch_or_pdf_path"
        ocr_json = local_ocr_json_path(local.get("table_dir_name", ""), image_hash, args.ocr_root)
        ocr_status = "local_ocr_json_present" if ocr_json else "no_local_ocr_json"

        table_ids = [h.get("table_id", "") for h in catalog_hits if h.get("table_id")]
        row = {
            **local,
            "pdf_page_number": page_number or "",
            "provenance_method": method,
            "provenance_confidence": "exact" if page_number is not None and method.startswith("hash_match") else "unresolved",
            "next_action": next_action,
            "source_pdf_original": chosen_pdf,
            "source_pdf_filename": os.path.basename(chosen_pdf),
            "resolved_start_page": start or "",
            "resolved_end_page": end or "",
            "range_status": range_meta.get("range_status", ""),
            "position_in_resolved_range": (page_number - start + 1) if page_number is not None and start else "",
            "province": range_meta.get("province", ""),
            "year": range_meta.get("year", ""),
            "title": range_meta.get("title", "") or range_meta.get("queue_title", ""),
            "canonical_topic": range_meta.get("canonical_topic", ""),
            "relevance_tier": range_meta.get("relevance_tier", ""),
            "manifest_classification": range_meta.get("manifest_classification", ""),
            "source_image_original": range_meta.get("source_image_original", ""),
            "source_json_original": range_meta.get("source_json_original", ""),
            "local_ocr_json": ocr_json,
            "local_ocr_json_exists": "1" if ocr_json else "0",
            "ocr_status": ocr_status,
            "catalog_table_ids": coalesce(table_ids),
            "catalog_layers": coalesce([h.get("layer", "") for h in catalog_hits]),
            "catalog_table_families": coalesce([h.get("table_family", "") for h in catalog_hits]),
            "catalog_quality_tiers": coalesce([h.get("quality_tier", "") for h in catalog_hits]),
            "catalog_manual_decisions": coalesce([h.get("manual_final_decision", "") for h in catalog_hits]),
            "catalog_reference_count": len(catalog_hits),
            "manifest_reference_count": len(manifest_hits),
        }
        page_rows.append(row)

    meta = {
        "local_image_paths_indexed": local_path_count,
        "unique_local_image_hashes_indexed": len(images),
        "manifest_image_hashes": len(manifests_by_hash),
        "catalog_image_hashes": len(catalog_refs),
        "table_ranges_indexed": len(range_by_key),
    }
    return page_rows, list(range_by_key.values()), meta


def table_range_rows(page_rows: list[dict[str, Any]], range_rows_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_rows:
        if row.get("table_key"):
            by_key_pages[row["table_key"]].append(row)
    range_by_key: dict[str, dict[str, Any]] = {}
    for raw in range_rows_raw:
        key = raw.get("table_key")
        if key:
            range_by_key[key] = raw
    for key, pages in by_key_pages.items():
        range_by_key.setdefault(key, {
            "table_key": key,
            "doc_hash": pages[0].get("doc_hash", ""),
            "table_index": pages[0].get("table_index", ""),
            "source_pdf": pages[0].get("source_pdf_original", ""),
            "source_pdf_filename": pages[0].get("source_pdf_filename", ""),
            "province": pages[0].get("province", ""),
            "year": pages[0].get("year", ""),
            "title": pages[0].get("title", ""),
            "resolved_start_page": int_or_blank(pages[0].get("resolved_start_page")),
            "resolved_end_page": int_or_blank(pages[0].get("resolved_end_page")),
            "range_status": pages[0].get("range_status", ""),
        })

    rows = []
    for key, raw in sorted(range_by_key.items()):
        pages = by_key_pages.get(key, [])
        observed = sorted({int(r["pdf_page_number"]) for r in pages if str(r.get("pdf_page_number", "")).isdigit()})
        start = int_or_blank(raw.get("resolved_start_page"))
        end = int_or_blank(raw.get("resolved_end_page"))
        expected = list(range(start, end + 1)) if start and end and end >= start else []
        missing = sorted(set(expected) - set(observed)) if expected else []
        if not expected:
            completion_status = "no_resolved_range"
            recommended_next_step = "recover_or_validate_table_range"
        elif missing:
            completion_status = "missing_pages_in_resolved_range"
            recommended_next_step = "review_range_then_render_missing_pdf_pages"
        elif any(r.get("provenance_confidence") != "exact" for r in pages):
            completion_status = "local_pages_present_but_some_unplaced"
            recommended_next_step = "fix_source_pdf_mapping_or_hash_mismatch"
        else:
            completion_status = "range_covered_by_digitized_pages"
            recommended_next_step = "no_page_recovery_needed"
        rows.append({
            "table_key": key,
            "completion_status": completion_status,
            "recommended_next_step": recommended_next_step,
            "doc_hash": raw.get("doc_hash", ""),
            "table_index": raw.get("table_index", ""),
            "source_pdf_original": raw.get("source_pdf", ""),
            "source_pdf_filename": os.path.basename(raw.get("source_pdf", "")),
            "province": raw.get("province", ""),
            "year": raw.get("year", ""),
            "title": raw.get("title", "") or raw.get("queue_title", ""),
            "canonical_topic": raw.get("canonical_topic", ""),
            "relevance_tier": raw.get("relevance_tier", ""),
            "range_status": raw.get("range_status", ""),
            "resolved_start_page": start or "",
            "resolved_end_page": end or "",
            "expected_pages_in_range": len(expected),
            "digitized_pages_in_range": len(observed),
            "missing_pages_in_range": len(missing),
            "coverage_pct": round(100 * len(observed) / len(expected), 1) if expected else "",
            "observed_pdf_pages": compress_ints(observed),
            "missing_pdf_pages": compress_ints(missing),
            "first_observed_pdf_page": observed[0] if observed else "",
            "last_observed_pdf_page": observed[-1] if observed else "",
            "n_local_images": len(pages),
            "n_exact_page_images": len(observed),
            "provenance_unresolved_images": sum(1 for r in pages if not str(r.get("pdf_page_number", "")).isdigit()),
        })
    return rows


def source_pdf_rows(page_rows: list[dict[str, Any]], ranges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pages_by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ranges_by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_rows:
        if row.get("source_pdf_original"):
            pages_by_pdf[row["source_pdf_original"]].append(row)
    for row in ranges:
        if row.get("source_pdf_original"):
            ranges_by_pdf[row["source_pdf_original"]].append(row)

    summary = []
    inventory = []
    missing_rows = []
    for pdf in sorted(set(pages_by_pdf) | set(ranges_by_pdf)):
        pages = pages_by_pdf.get(pdf, [])
        rr = ranges_by_pdf.get(pdf, [])
        expected_pages: set[int] = set()
        page_ranges: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rr:
            start = int_or_blank(r.get("resolved_start_page"))
            end = int_or_blank(r.get("resolved_end_page"))
            if not start or not end or end < start:
                continue
            for p in range(start, end + 1):
                expected_pages.add(p)
                page_ranges[p].append(r)
        observed_pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pages:
            p = int_or_blank(row.get("pdf_page_number"))
            if p:
                observed_pages[p].append(row)
        missing = sorted(expected_pages - set(observed_pages))
        observed_expected = sorted(set(observed_pages) & expected_pages)
        observed_outside = sorted(set(observed_pages) - expected_pages)
        for p in sorted(expected_pages):
            obs = observed_pages.get(p, [])
            exp = page_ranges.get(p, [])
            inv = {
                "source_pdf_original": pdf,
                "source_pdf_filename": os.path.basename(pdf),
                "pdf_page_number": p,
                "digitized": "1" if obs else "0",
                "n_images": len(obs),
                "image_hashes": coalesce([o.get("image_hash", "") for o in obs]),
                "source_image_local": coalesce([o.get("source_image_local", "") for o in obs]),
                "expected_by_n_ranges": len(exp),
                "table_keys": coalesce([e.get("table_key", "") for e in exp]),
                "table_indexes": coalesce([e.get("table_index", "") for e in exp]),
                "titles": coalesce([e.get("title", "") for e in exp]),
                "province": exp[0].get("province", "") if exp else (obs[0].get("province", "") if obs else ""),
                "year": exp[0].get("year", "") if exp else (obs[0].get("year", "") if obs else ""),
            }
            inventory.append(inv)
            if not obs:
                priority = missing_page_priority(inv.get("titles", ""))
                missing_rows.append({
                    **inv,
                    "review_priority": priority,
                    "missing_reason": "expected_in_resolved_table_range_but_no_local_image",
                    "recommended_next_step": (
                        "render_and_ocr_if_target_table" if priority in {"high_target", "medium_target"}
                        else "check_range_before_rendering"
                    ),
                })
        if not expected_pages:
            pdf_status = "no_resolved_ranges"
            pdf_next_step = "recover_table_ranges_or_source_metadata"
        elif missing:
            pdf_status = "missing_expected_pages"
            pdf_next_step = "triage_missing pages by table_range_completeness and missing_pdf_pages"
        elif any(not str(p.get("pdf_page_number", "")).isdigit() for p in pages):
            pdf_status = "images_need_page_resolution"
            pdf_next_step = "recover_source_pdf_mapping_or_investigate_hash_mismatch"
        else:
            pdf_status = "resolved_ranges_covered"
            pdf_next_step = "no_page_recovery_needed"
        summary.append({
            "source_pdf_original": pdf,
            "pdf_completion_status": pdf_status,
            "recommended_next_step": pdf_next_step,
            "source_pdf_filename": os.path.basename(pdf),
            "province": coalesce([r.get("province", "") for r in rr] + [p.get("province", "") for p in pages]),
            "year": coalesce([r.get("year", "") for r in rr] + [p.get("year", "") for p in pages]),
            "n_table_ranges": len(rr),
            "min_resolved_page": min(expected_pages) if expected_pages else "",
            "max_resolved_page": max(expected_pages) if expected_pages else "",
            "expected_unique_pages_in_ranges": len(expected_pages),
            "digitized_unique_pages_in_ranges": len(observed_expected),
            "missing_unique_pages_in_ranges": len(missing),
            "coverage_pct": round(100 * len(observed_expected) / len(expected_pages), 1) if expected_pages else "",
            "observed_pages_outside_resolved_ranges": compress_ints(observed_outside),
            "missing_pdf_pages": compress_ints(missing),
            "n_local_images": len(pages),
            "n_exact_page_images": sum(1 for p in pages if str(p.get("pdf_page_number", "")).isdigit()),
            "n_unresolved_page_images": sum(1 for p in pages if not str(p.get("pdf_page_number", "")).isdigit()),
        })
    return summary, inventory, missing_rows


def unresolved_doc_hash_rows(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in page_rows:
        if not row.get("source_pdf_original"):
            grouped[row.get("doc_hash", "") or "(missing_doc_hash)"].append(row)
    rows = []
    for doc_hash, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        table_keys = sorted({r.get("table_key", "") for r in items if r.get("table_key")})
        local_roots = sorted({
            "/".join((r.get("source_image_local", "") or "").split("/")[:3])
            for r in items
            if r.get("source_image_local")
        })
        rows.append({
            "doc_hash": doc_hash,
            "n_images_without_source_pdf": len(items),
            "n_table_dirs": len(table_keys),
            "table_keys": coalesce(table_keys),
            "local_roots": coalesce(local_roots),
            "sample_image_paths": coalesce([r.get("source_image_local", "") for r in items[:8]]),
            "sample_titles": coalesce([r.get("title", "") for r in items[:8]]),
            "suggested_fix": "recover doc_hash -> original source PDF path, then rerun build_page_provenance_workbook.py",
        })
    return rows


def ordered_image_sequence_rows(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def source_group(row: dict[str, Any]) -> str:
        if row.get("source_pdf_filename"):
            return row["source_pdf_filename"]
        if row.get("doc_hash"):
            return f"unresolved_doc::{row['doc_hash']}"
        return "unresolved_doc::missing_doc_hash"

    def sort_key(row: dict[str, Any]) -> tuple[str, int, int, str, str]:
        page = int_or_blank(row.get("pdf_page_number"))
        table_index = int_or_blank(row.get("table_index")) or 9999
        return (
            source_group(row),
            0 if page is not None else 1,
            page if page is not None else 10**9,
            f"{table_index:04d}",
            row.get("image_hash", ""),
        )

    out = []
    last_group = None
    seq = 0
    for row in sorted(page_rows, key=sort_key):
        group = source_group(row)
        if group != last_group:
            seq = 0
            last_group = group
        seq += 1
        signal = table_vii_signal(row)
        out.append({
            "source_group": group,
            "sequence_index": seq,
            "pdf_page_number": row.get("pdf_page_number", ""),
            "source_pdf_filename": row.get("source_pdf_filename", ""),
            "source_pdf_original": row.get("source_pdf_original", ""),
            "doc_hash": row.get("doc_hash", ""),
            "table_key": row.get("table_key", ""),
            "table_index": row.get("table_index", ""),
            "image_hash": row.get("image_hash", ""),
            "source_image_local": row.get("source_image_local", ""),
            "all_source_image_local_paths": row.get("all_source_image_local_paths", ""),
            "local_ocr_json": row.get("local_ocr_json", ""),
            "local_ocr_json_exists": row.get("local_ocr_json_exists", ""),
            "ocr_status": row.get("ocr_status", ""),
            "title_or_directory": row.get("title", "") or row.get("table_dir_name", ""),
            "table_vii_signal": signal,
            "is_table_vii_candidate": "1" if signal else "0",
            "provenance_confidence": row.get("provenance_confidence", ""),
            "provenance_method": row.get("provenance_method", ""),
            "next_action": row.get("next_action", ""),
            "province": row.get("province", ""),
            "year": row.get("year", ""),
            "catalog_table_ids": row.get("catalog_table_ids", ""),
            "catalog_table_families": row.get("catalog_table_families", ""),
        })
    return out


def table_vii_accessibility_rows(sequence_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [r for r in sequence_rows if r.get("is_table_vii_candidate") == "1"]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[row["source_group"]].append(row)

    missing_by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing_rows:
        if table_vii_signal(row):
            missing_by_pdf[row.get("source_pdf_filename", "")].append(row)

    out = []
    for group, rows in sorted(by_group.items()):
        rows_sorted = sorted(rows, key=lambda r: (int_or_blank(r.get("pdf_page_number")) is None, int_or_blank(r.get("pdf_page_number")) or 10**9, r.get("image_hash", "")))
        n_without_pdf = sum(1 for r in rows_sorted if not r.get("source_pdf_original"))
        n_without_page = sum(1 for r in rows_sorted if not r.get("pdf_page_number"))
        pages = sorted({int(r["pdf_page_number"]) for r in rows_sorted if str(r.get("pdf_page_number", "")).isdigit()})
        missing = missing_by_pdf.get(rows_sorted[0].get("source_pdf_filename", ""), [])
        if n_without_pdf:
            status = "needs_source_pdf_mapping"
            next_step = "map doc_hash to original PDF, then rerun provenance workbook"
        elif n_without_page:
            status = "needs_page_hash_investigation"
            next_step = "check source PDF path/range or rerender source page"
        elif missing:
            status = "accessible_images_but_missing_expected_pages"
            next_step = "review table_vii_missing_pages and render true target pages"
        else:
            status = "accessible_and_ordered"
            next_step = "ready for Table VII review/assembly"
        out.append({
            "source_group": group,
            "table_vii_access_status": status,
            "recommended_next_step": next_step,
            "n_table_vii_candidate_images": len(rows_sorted),
            "n_exact_pdf_pages": sum(1 for r in rows_sorted if r.get("provenance_confidence") == "exact"),
            "n_without_source_pdf": n_without_pdf,
            "n_without_pdf_page": n_without_page,
            "local_pdf_page_span": f"{pages[0]}-{pages[-1]}" if pages else "",
            "local_pdf_pages": compress_ints(pages),
            "n_table_vii_missing_expected_pages": len(missing),
            "table_vii_missing_pdf_pages": compress_ints([int(r["pdf_page_number"]) for r in missing if str(r.get("pdf_page_number", "")).isdigit()]),
            "doc_hashes": coalesce([r.get("doc_hash", "") for r in rows_sorted]),
            "source_pdf_filename": rows_sorted[0].get("source_pdf_filename", ""),
            "source_pdf_original": rows_sorted[0].get("source_pdf_original", ""),
            "province": coalesce([r.get("province", "") for r in rows_sorted]),
            "year": coalesce([r.get("year", "") for r in rows_sorted]),
            "table_keys": coalesce([r.get("table_key", "") for r in rows_sorted]),
            "signals": coalesce([r.get("table_vii_signal", "") for r in rows_sorted]),
            "titles_or_dirs": coalesce([r.get("title_or_directory", "") for r in rows_sorted[:12]]),
            "first_image_path": rows_sorted[0].get("source_image_local", ""),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_text(row.get(k, "")) for k in columns})


def write_json_dictionary(path: Path, rows: list[dict[str, Any]]) -> None:
    data = {row["image_hash"]: row for row in rows}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def add_sheet(workbook: Any, name: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ws = workbook.add_worksheet(name[:31])
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#EDE1C9", "border": 1})
    ok_fmt = workbook.add_format({"bg_color": "#D9EAD3"})
    warn_fmt = workbook.add_format({"bg_color": "#FFF2CC"})
    bad_fmt = workbook.add_format({"bg_color": "#F4CCCC"})
    text_fmt = workbook.add_format({"text_wrap": False})
    for c, col in enumerate(columns):
        ws.write(0, c, col, header_fmt)
    for r, row in enumerate(rows, 1):
        for c, col in enumerate(columns):
            value = clean_text(row.get(col, ""))
            ws.write(r, c, value, text_fmt)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(len(rows), 1), max(len(columns) - 1, 0))
    for c, col in enumerate(columns):
        width = min(max(len(col) + 2, 12), 45)
        if col in {"title", "source_pdf_original", "source_image_local", "missing_pdf_pages", "observed_pdf_pages"}:
            width = 42
        ws.set_column(c, c, width)
    if "missing_pages_in_range" in columns:
        idx = columns.index("missing_pages_in_range")
        ws.conditional_format(1, idx, len(rows), idx, {"type": "cell", "criteria": ">", "value": 0, "format": warn_fmt})
    if "provenance_confidence" in columns:
        idx = columns.index("provenance_confidence")
        ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": "unresolved", "format": bad_fmt})
        ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": "exact", "format": ok_fmt})
    if "digitized" in columns:
        idx = columns.index("digitized")
        ws.conditional_format(1, idx, len(rows), idx, {"type": "cell", "criteria": "==", "value": 0, "format": bad_fmt})
    if "local_ocr_json_exists" in columns:
        idx = columns.index("local_ocr_json_exists")
        ws.conditional_format(1, idx, len(rows), idx, {"type": "cell", "criteria": "==", "value": 1, "format": ok_fmt})
        ws.conditional_format(1, idx, len(rows), idx, {"type": "cell", "criteria": "==", "value": 0, "format": bad_fmt})
    for status_col in ["completion_status", "pdf_completion_status", "table_vii_access_status", "next_action", "recommended_next_step"]:
        if status_col in columns:
            idx = columns.index(status_col)
            for token in ["ready", "covered", "accessible_and_ordered", "no_page_recovery_needed"]:
                ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": token, "format": ok_fmt})
            for token in ["missing", "recover", "mapping", "unresolved", "investigate", "needs"]:
                ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": token, "format": warn_fmt})
    if "review_priority" in columns:
        idx = columns.index("review_priority")
        ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": "high", "format": bad_fmt})
        ws.conditional_format(1, idx, len(rows), idx, {"type": "text", "criteria": "containing", "value": "medium", "format": warn_fmt})


def write_workbook(path: Path, sheets: list[tuple[str, list[dict[str, Any]], list[str]]], summary: dict[str, Any]) -> None:
    if xlsxwriter is None:
        return
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True})
    ws = workbook.add_worksheet("README")
    title_fmt = workbook.add_format({"bold": True, "font_size": 14})
    ws.write(0, 0, "Census page provenance workbook", title_fmt)
    lines = [
        "This workbook maps every local digitized table image to its source PDF and exact PDF page where recoverable.",
        "Exact PDF page recovery uses sha1(f'{source_pdf_original}:{pdf_page_number}') == image_hash.",
        "Missing pages are pages inside resolved TOC/table ranges that do not have a local rendered image.",
        "These missing pages are a worklist, not automatic proof every page contains relevant table data.",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    for i, line in enumerate(lines, 2):
        ws.write(i, 0, line)
    ws.set_column(0, 0, 120)
    for sheet_name, rows, columns in sheets:
        add_sheet(workbook, sheet_name, rows, columns)
    workbook.close()


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Page Provenance And Completeness",
        "",
        "This folder is the corpus-level dictionary for source-image provenance.",
        "",
        "Core rule: each rendered PNG hash is recoverable as `sha1(source_pdf_original + ':' + pdf_page_number)`.",
        "",
        "Files:",
        "",
        "- `page_provenance_dictionary.csv`: one row per unique local PNG hash, with all local paths preserved.",
        "- `ordered_image_sequence.csv`: every image ordered within its source PDF or unresolved doc hash.",
        "- `table_vii_image_sequence.csv`: likely Table VII / age-sex-civil-condition images in source order.",
        "- `table_vii_accessibility.csv`: source-level checklist for whether Table VII candidates are accessible and ordered.",
        "- `table_vii_missing_pages.csv`: missing expected pages for likely Table VII/age-sex-civil-condition ranges.",
        "- `table_range_completeness.csv`: one row per TOC/table range with observed vs missing PDF pages.",
        "- `source_pdf_completeness.csv`: one row per source PDF.",
        "- `pdf_page_inventory.csv`: one row per expected PDF page inside resolved table ranges.",
        "- `missing_pdf_pages.csv`: exact PDF pages that are expected by a resolved range but absent locally.",
        "- `page_provenance_dictionary.json`: image-hash keyed lookup dictionary.",
        "- `page_provenance_workbook.xlsx`: human-browsable version of the same outputs.",
        "",
        "Interpretation:",
        "",
        "- `provenance_confidence=exact` means the image hash matched a source PDF and PDF page.",
        "- Missing pages are a first-pass worklist from TOC/table ranges. Some may be non-table pages if a range is broad.",
        "- For final OCR targeting, prioritize rows in `missing_pdf_pages.csv` whose neighboring observed pages are relevant Table VII/age tables.",
        "",
        "Summary:",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact source PDF/page provenance for digitized census images.")
    parser.add_argument("--image-root", type=Path, action="append",
                        help="Digitized-table image root. May be repeated. Defaults to all known local image roots.")
    parser.add_argument("--manifest-jsonl", type=Path, action="append", default=[p for p in DEFAULT_MANIFESTS if p.exists()])
    parser.add_argument("--source-catalog", type=Path, action="append", default=[p for p in DEFAULT_CATALOGS if p.exists()])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--max-bruteforce-pages", type=int, default=2000)
    args = parser.parse_args()
    args.image_roots = [p.resolve() for p in (args.image_root or [p for p in DEFAULT_IMAGE_ROOTS if p.exists()])]
    args.ocr_root = args.ocr_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    page_rows, raw_ranges, meta = build_page_dictionary(args)
    ranges = table_range_rows(page_rows, raw_ranges)
    pdf_summary, inventory, missing = source_pdf_rows(page_rows, ranges)
    unresolved_docs = unresolved_doc_hash_rows(page_rows)
    ordered_sequence = ordered_image_sequence_rows(page_rows)
    table_vii_sequence = [row for row in ordered_sequence if row.get("is_table_vii_candidate") == "1"]
    table_vii_missing = [row for row in missing if table_vii_signal(row)]
    table_vii_accessibility = table_vii_accessibility_rows(ordered_sequence, missing)

    page_cols = [
        "image_hash", "source_image_local", "all_source_image_local_paths", "local_path_count",
        "source_image_original", "source_json_original",
        "source_pdf_original", "source_pdf_filename", "pdf_page_number", "provenance_confidence",
        "provenance_method", "next_action", "doc_hash", "table_index", "table_key", "table_dir_name",
        "local_ocr_json_exists", "ocr_status", "local_ocr_json",
        "resolved_start_page", "resolved_end_page", "position_in_resolved_range", "range_status",
        "province", "year", "title", "canonical_topic", "relevance_tier", "manifest_classification",
        "catalog_table_ids", "catalog_layers", "catalog_table_families", "catalog_quality_tiers",
        "catalog_manual_decisions", "catalog_reference_count", "manifest_reference_count",
    ]
    range_cols = [
        "table_key", "completion_status", "recommended_next_step", "doc_hash", "table_index", "source_pdf_original", "source_pdf_filename",
        "province", "year", "title", "canonical_topic", "relevance_tier", "range_status",
        "resolved_start_page", "resolved_end_page", "expected_pages_in_range",
        "digitized_pages_in_range", "missing_pages_in_range", "coverage_pct",
        "observed_pdf_pages", "missing_pdf_pages", "first_observed_pdf_page", "last_observed_pdf_page",
        "n_local_images", "n_exact_page_images", "provenance_unresolved_images",
    ]
    pdf_cols = [
        "source_pdf_original", "pdf_completion_status", "recommended_next_step", "source_pdf_filename", "province", "year", "n_table_ranges",
        "min_resolved_page", "max_resolved_page", "expected_unique_pages_in_ranges",
        "digitized_unique_pages_in_ranges", "missing_unique_pages_in_ranges", "coverage_pct",
        "missing_pdf_pages", "observed_pages_outside_resolved_ranges", "n_local_images",
        "n_exact_page_images", "n_unresolved_page_images",
    ]
    inventory_cols = [
        "source_pdf_original", "source_pdf_filename", "pdf_page_number", "digitized", "n_images",
        "image_hashes", "source_image_local", "expected_by_n_ranges", "table_keys", "table_indexes",
        "titles", "province", "year",
    ]
    missing_cols = inventory_cols + ["review_priority", "missing_reason", "recommended_next_step"]
    unresolved_doc_cols = [
        "doc_hash", "n_images_without_source_pdf", "n_table_dirs", "table_keys",
        "local_roots", "sample_image_paths", "sample_titles", "suggested_fix",
    ]
    sequence_cols = [
        "source_group", "sequence_index", "pdf_page_number", "source_pdf_filename", "source_pdf_original",
        "doc_hash", "table_key", "table_index", "image_hash", "source_image_local",
        "all_source_image_local_paths", "local_ocr_json_exists", "ocr_status", "local_ocr_json",
        "title_or_directory", "table_vii_signal", "is_table_vii_candidate",
        "provenance_confidence", "provenance_method", "next_action", "province", "year",
        "catalog_table_ids", "catalog_table_families",
    ]
    table_vii_access_cols = [
        "source_group", "table_vii_access_status", "recommended_next_step",
        "n_table_vii_candidate_images", "n_exact_pdf_pages", "n_without_source_pdf",
        "n_without_pdf_page", "local_pdf_page_span", "local_pdf_pages",
        "n_table_vii_missing_expected_pages", "table_vii_missing_pdf_pages",
        "doc_hashes", "source_pdf_filename", "source_pdf_original", "province", "year",
        "table_keys", "signals", "titles_or_dirs", "first_image_path",
    ]

    write_csv(args.output_dir / "page_provenance_dictionary.csv", page_rows, page_cols)
    write_csv(args.output_dir / "ordered_image_sequence.csv", ordered_sequence, sequence_cols)
    write_csv(args.output_dir / "table_vii_image_sequence.csv", table_vii_sequence, sequence_cols)
    write_csv(args.output_dir / "table_vii_accessibility.csv", table_vii_accessibility, table_vii_access_cols)
    write_csv(args.output_dir / "table_range_completeness.csv", ranges, range_cols)
    write_csv(args.output_dir / "source_pdf_completeness.csv", pdf_summary, pdf_cols)
    write_csv(args.output_dir / "pdf_page_inventory.csv", inventory, inventory_cols)
    write_csv(args.output_dir / "missing_pdf_pages.csv", missing, missing_cols)
    write_csv(args.output_dir / "table_vii_missing_pages.csv", table_vii_missing, missing_cols)
    write_csv(args.output_dir / "unresolved_source_pdf_doc_hashes.csv", unresolved_docs, unresolved_doc_cols)
    write_json_dictionary(args.output_dir / "page_provenance_dictionary.json", page_rows)

    summary = {
        **meta,
        "page_rows": len(page_rows),
        "exact_page_rows": sum(1 for r in page_rows if r.get("provenance_confidence") == "exact"),
        "unresolved_page_rows": sum(1 for r in page_rows if r.get("provenance_confidence") != "exact"),
        "local_ocr_json_present": sum(1 for r in page_rows if r.get("local_ocr_json_exists") == "1"),
        "no_local_ocr_json": sum(1 for r in page_rows if r.get("local_ocr_json_exists") != "1"),
        "table_ranges": len(ranges),
        "table_ranges_with_missing_pages": sum(1 for r in ranges if int_or_blank(r.get("missing_pages_in_range"))),
        "source_pdfs": len(pdf_summary),
        "missing_pdf_pages": len(missing),
        "unresolved_source_pdf_doc_hashes": len(unresolved_docs),
        "ordered_image_sequence_rows": len(ordered_sequence),
        "table_vii_candidate_images": len(table_vii_sequence),
        "table_vii_source_groups": len(table_vii_accessibility),
        "table_vii_missing_expected_pages": len(table_vii_missing),
        "provenance_methods": dict(Counter(r.get("provenance_method", "") for r in page_rows)),
    }
    write_readme(args.output_dir / "README.md", summary)
    write_workbook(
        args.output_dir / "page_provenance_workbook.xlsx",
        [
            ("table_vii_access", table_vii_accessibility, table_vii_access_cols),
            ("table_vii_images", table_vii_sequence, sequence_cols),
            ("table_vii_missing", table_vii_missing, missing_cols),
            ("ordered_images_all", ordered_sequence, sequence_cols),
            ("source_pdf_summary", pdf_summary, pdf_cols),
            ("table_ranges", ranges, range_cols),
            ("missing_pdf_pages", missing, missing_cols),
            ("unresolved_doc_hashes", unresolved_docs, unresolved_doc_cols),
            ("page_dictionary", page_rows, page_cols),
            ("pdf_page_inventory", inventory, inventory_cols),
        ],
        summary,
    )

    print("Page provenance build")
    print("---------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
