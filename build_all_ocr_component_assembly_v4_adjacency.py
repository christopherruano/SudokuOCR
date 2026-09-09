#!/usr/bin/env python3
"""Build adjacency-first OCR page components.

This is a safer successor to build_all_ocr_component_assembly_v2.py. It treats
the v3 component pages CSV as a flat page inventory, ignores broad heuristic
component assignments for grouping, and rebuilds components from pairwise
adjacent PDF-page edges.

Rules:
  * same source first;
  * real pdf_page adjacency before any title/family/table-dir evidence;
  * classifier/title/type compatibility before joining;
  * uncertain cases split and marked for review instead of over-merged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import xlsxwriter


ROOT = Path(__file__).resolve().parent
V3_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v3_refined"
LEDGER_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined"
CLASSIFIER_CSV = LEDGER_DIR / "gemini_ocr_worklists/gemini_classify_priority_all.csv"
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency"

V3_PAGES = V3_DIR / "all_ocr_component_pages.csv"
V3_SUMMARY = V3_DIR / "all_ocr_component_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stable_id(prefix: str, *parts: object) -> str:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def int_or_none(value: object) -> int | None:
    s = str(value or "").strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None


def uniq_join(values: Iterable[object], limit: int = 8) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > limit:
        return "; ".join(out[:limit]) + f"; ...(+{len(out) - limit})"
    return "; ".join(out)


def counter_json(values: Iterable[object]) -> str:
    return json.dumps(dict(Counter(str(v or "") for v in values)), sort_keys=True)


def page_gap_summary(rows: list[dict[str, object]]) -> tuple[str, int, str]:
    nums = sorted(n for n in {int_or_none(r.get("pdf_page")) for r in rows} if n is not None)
    if not nums:
        return "", 0, ""
    if len(nums) == 1:
        return str(nums[0]), 0, ""
    gaps: list[tuple[int, int]] = []
    for a, b in zip(nums, nums[1:]):
        if b > a + 1:
            gaps.append((a + 1, b - 1))
    gap_count = sum(b - a + 1 for a, b in gaps)
    labels = [str(a) if a == b else f"{a}-{b}" for a, b in gaps[:20]]
    return f"{nums[0]}-{nums[-1]}", gap_count, "; ".join(labels)


def source_id_from_pdf(source_pdf: str) -> str:
    s = str(source_pdf or "").strip()
    m = re.search(r"(\d{7,})", s)
    return m.group(1) if m else ""


def image_stem(path: str) -> str:
    return Path(str(path or "")).stem


def source_key(row: dict[str, object]) -> str:
    source_pdf = str(row.get("source_pdf") or "").strip()
    if source_pdf:
        return source_pdf
    heuristic = str(row.get("heuristic_source_key") or "").strip()
    if heuristic:
        return heuristic
    source_group = str(row.get("source_group_key") or "").strip()
    if source_group:
        return source_group.split("||", 1)[0]
    year = str(row.get("year") or "").strip()
    province = str(row.get("province") or "").strip()
    if year or province:
        return f"{year}::{province}"
    return "unknown_source"


def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_continuation_words(value: str) -> str:
    s = clean_text(value)
    s = re.sub(r"\b(contd|continued|concluded|concld|concl|part|continued)\b", " ", s)
    s = re.sub(r"\b[abcdeh]\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_blob(row: dict[str, object]) -> str:
    values = [
        row.get("classifier_printed_title"),
        row.get("core_ocr_title"),
        row.get("title"),
        row.get("original_overlap_titles"),
        row.get("family"),
        row.get("table_number"),
    ]
    return " ".join(str(v or "") for v in values if str(v or "").strip())


def has_continuation_signal(row: dict[str, object]) -> bool:
    blob = clean_text(title_blob(row))
    return bool(re.search(r"\b(contd|continued|concluded|concld|concl|continuation)\b", blob))


def is_context_or_blank(row: dict[str, object]) -> bool:
    cls = str(row.get("classifier_page_class") or "")
    if cls in {"tablevii_context_or_title", "blank_or_empty"}:
        return True
    blob = clean_text(title_blob(row))
    status = clean_text(row.get("ocr_status") or row.get("page_status"))
    return bool(
        status == "empty"
        or re.search(r"\b(blank|contents|index|preface|introduction|chapter|notes?|corrigenda|map showing|diagram showing|no table present)\b", blob)
    )


def is_derivative_selected(row: dict[str, object]) -> bool:
    cls = str(row.get("classifier_page_class") or "")
    blob = clean_text(title_blob(row) + " " + str(row.get("classifier_why") or ""))
    if cls == "derivative_civil_condition":
        return True
    # In these census volumes, "Table VIII. Civil Condition by Age" without
    # "by religion" is the selected caste/race/tribe derivative table. The main
    # 1891-style Table VIII pages spell out "Civil Condition by Religion and Age".
    if re.search(r"\b(table viii|imperial table viii)\b", blob) and re.search(r"\bcivil condition by age\b", blob) and "religion" not in blob:
        return True
    return bool(
        re.search(r"\b(selected castes?|selected races?|selected tribes?|castes races tribes|caste races tribe|races and tribes|caste and tribe|supplementary table e)\b", blob)
        or re.search(r"\b(table viii|imperial table viii)\b.*\b(selected|caste|race|tribe)\b", blob)
        or re.search(r"\b(caste|race|tribe)\b.*\b(civil condition by age|age sex and civil condition)\b", blob)
    )


def is_offtarget(row: dict[str, object]) -> bool:
    cls = str(row.get("classifier_page_class") or "")
    if cls == "offtarget_table":
        return True
    blob = clean_text(title_blob(row))
    return bool(re.search(r"\b(occupation|literacy|education|infirmities|birth place|language|industrial|agricultural|houses?|villages?|towns classified)\b", blob))


def is_main_age_civil(row: dict[str, object]) -> bool:
    cls = str(row.get("classifier_page_class") or "")
    if cls == "main_tablevii_numeric" and not is_derivative_selected(row):
        return True
    blob = clean_text(title_blob(row))
    if is_derivative_selected(row) or is_offtarget(row) or is_context_or_blank(row):
        return False
    tableish = bool(re.search(r"\b(table vii|imperial table vii|state table vii|c iii|c 3|form vi a|form via|civil condition by religion and age)\b", blob))
    subject = bool(re.search(r"\b(age sex.*civil condition|civil condition.*age|distribution.*civil condition.*age|age and civil condition)\b", blob))
    return tableish and subject


def page_kind(row: dict[str, object]) -> str:
    if is_context_or_blank(row):
        return "context_or_blank"
    if is_derivative_selected(row):
        return "derivative_selected_caste_race_tribe"
    if is_main_age_civil(row):
        return "main_age_civil_numeric"
    if is_offtarget(row):
        return "offtarget_table"
    blob = clean_text(title_blob(row))
    if "table" in blob or row.get("table_number"):
        return "other_table"
    if row.get("ocr_artifact") or int_or_none(row.get("rows_extracted")):
        return "structured_unknown"
    return "unknown"


def table_family_key(row: dict[str, object]) -> str:
    kind = str(row.get("page_kind") or page_kind(row))
    if kind == "main_age_civil_numeric":
        return "main_age_civil"
    if kind == "derivative_selected_caste_race_tribe":
        return "derivative_selected"
    if kind == "context_or_blank":
        return "context"
    if kind == "offtarget_table":
        return "offtarget"
    blob = clean_text(title_blob(row))
    m = re.search(r"\b(table|form|statement|c)\s*(viii|vii|vi+a?|iii|[0-9]+)\b", blob)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    fam = clean_text(row.get("family"))
    return fam or kind or "unknown"


def condition_focus(row: dict[str, object]) -> str:
    blob = clean_text(title_blob(row))
    hits = []
    for term in ["unmarried", "single", "married", "widowed", "population"]:
        if re.search(rf"\b{term}\b", blob):
            hits.append(term)
    if {"unmarried", "married", "widowed"} & set(hits) and len(set(hits) & {"unmarried", "married", "widowed"}) >= 2:
        return "multiple_civil_conditions"
    if hits:
        return "|".join(hits[:3])
    return ""


def title_similarity(a: dict[str, object], b: dict[str, object]) -> float:
    ta = strip_continuation_words(title_blob(a))
    tb = strip_continuation_words(title_blob(b))
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def row_quality_flags(row: dict[str, object]) -> list[str]:
    flags: list[str] = []
    checks = int_or_none(row.get("constraint_checks")) or 0
    failed = int_or_none(row.get("constraint_failed")) or 0
    rows = int_or_none(row.get("rows_extracted")) or int_or_none(row.get("core_rows")) or 0
    status = clean_text(row.get("ocr_status") or row.get("page_status") or row.get("core_ocr_status"))
    validation = clean_text(row.get("validation") or row.get("verdict") or row.get("core_ocr_decision"))
    if row.get("ocr_artifact") and checks == 0:
        flags.append("zero_constraint_checks")
    if failed:
        flags.append("constraint_failures")
    if rows == 0 and row.get("ocr_artifact"):
        flags.append("zero_rows_extracted")
    if re.search(r"\b(risk|rejected|failed|many off|few off|unchecked)\b", status + " " + validation):
        flags.append("low_quality_ocr")
    return flags


def compatible_adjacent(prev: dict[str, object], cur: dict[str, object]) -> tuple[bool, str, list[str]]:
    flags: list[str] = []
    prev_source = str(prev.get("adj_source_key") or source_key(prev))
    cur_source = str(cur.get("adj_source_key") or source_key(cur))
    if prev_source != cur_source:
        return False, "source_pdf_or_source_key_changed", ["source_changed"]

    prev_page = int_or_none(prev.get("pdf_page"))
    cur_page = int_or_none(cur.get("pdf_page"))
    if prev_page is None or cur_page is None:
        return False, "missing_real_pdf_page_for_pairwise_order", ["missing_pdf_page"]
    prev_kind = str(prev.get("page_kind") or page_kind(prev))
    cur_kind = str(cur.get("page_kind") or page_kind(cur))
    if cur_page == prev_page:
        if prev_kind == cur_kind:
            return True, "same_pdf_page_parallel_or_duplicate_record", ["duplicate_pdf_page_record"]
        return False, f"same_pdf_page_conflicting_page_kind_{prev_kind}_to_{cur_kind}", ["duplicate_pdf_page_record", "incompatible_page_class_transition"]
    if cur_page != prev_page + 1:
        missing = f"{prev_page + 1}" if cur_page == prev_page + 2 else f"{prev_page + 1}-{cur_page - 1}"
        return False, f"pdf_page_gap_before_current_missing_{missing}", ["missing_intermediate_pages"]

    if prev_kind != cur_kind:
        if "context_or_blank" in {prev_kind, cur_kind}:
            return False, f"context_or_blank_boundary_{prev_kind}_to_{cur_kind}", ["context_boundary"]
        return False, f"incompatible_page_kind_transition_{prev_kind}_to_{cur_kind}", ["incompatible_page_class_transition"]

    if cur_kind == "context_or_blank":
        return True, "adjacent_context_or_front_matter_pages_kept_separate_from_numeric_body", ["context_front_matter_run"]
    if cur_kind in {"offtarget_table", "unknown", "structured_unknown"}:
        fam_same = table_family_key(prev) == table_family_key(cur)
        sim = title_similarity(prev, cur)
        if fam_same and (has_continuation_signal(cur) or sim >= 0.82):
            return True, f"adjacent_non_target_pages_same_table_family_title_similarity_{sim:.2f}", ["non_target_run_review"]
        return False, f"non_target_or_unknown_adjacent_page_not_confident_similarity_{sim:.2f}", ["uncertain_non_target_break"]

    prev_family = table_family_key(prev)
    cur_family = table_family_key(cur)
    if prev_family != cur_family:
        return False, f"table_family_switch_{prev_family}_to_{cur_family}", ["table_number_or_subject_switch"]

    if cur_kind == "derivative_selected_caste_race_tribe":
        sim = title_similarity(prev, cur)
        if has_continuation_signal(cur) or sim >= 0.70:
            return True, f"adjacent_derivative_selected_table_continuation_similarity_{sim:.2f}", ["derivative_selected_table"]
        return False, f"selected_caste_race_tribe_derivative_boundary_similarity_{sim:.2f}", ["derivative_selected_table", "title_switch"]

    # Main age/civil numeric pages: require a real continuation signal, strong
    # title continuity, or absent/short continuation title. Do not let same
    # source/family alone merge the pair.
    sim = title_similarity(prev, cur)
    prev_focus = condition_focus(prev)
    cur_focus = condition_focus(cur)
    if prev_focus and cur_focus and prev_focus != cur_focus:
        if not has_continuation_signal(cur) and sim < 0.78:
            return False, f"civil_condition_or_column_focus_switch_{prev_focus}_to_{cur_focus}_similarity_{sim:.2f}", ["column_schema_or_subtable_switch"]

    if has_continuation_signal(cur):
        return True, f"adjacent_main_numeric_continued_heading_similarity_{sim:.2f}", []
    if sim >= 0.64:
        return True, f"adjacent_main_numeric_same_title_similarity_{sim:.2f}", []

    cur_title = strip_continuation_words(title_blob(cur))
    prev_title = strip_continuation_words(title_blob(prev))
    if len(cur_title) < 24 or len(prev_title) < 24:
        flags.append("uncertain_join_short_or_missing_title")
        return True, f"adjacent_main_numeric_short_title_no_switch_similarity_{sim:.2f}", flags

    return False, f"adjacent_but_no_structural_continuation_similarity_{sim:.2f}", ["uncertain_break_title_or_schema"]


def load_classifier() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    by_source_page: dict[tuple[str, str], dict[str, str]] = {}
    by_stem: dict[str, dict[str, str]] = {}
    for row in read_csv(CLASSIFIER_CSV):
        mapped = {
            "classifier_page_class": row.get("page_class", ""),
            "classifier_printed_title": row.get("printed_title", ""),
            "classifier_why": row.get("why", ""),
            "classifier_contains_numeric_grid": row.get("contains_numeric_grid", ""),
            "classifier_status": row.get("status", ""),
        }
        source_id = row.get("source_id", "")
        pdf_page = row.get("pdf_page", "")
        if source_id and pdf_page:
            by_source_page[(source_id, pdf_page)] = mapped
        if row.get("image_stem"):
            by_stem[row["image_stem"]] = mapped
    return by_source_page, by_stem


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_source_page, by_stem = load_classifier()
    enriched: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        new: dict[str, object] = dict(row)
        new["v3_component_id"] = row.get("component_id", "")
        new["v3_component_basis"] = row.get("component_basis", "")
        new["v3_component_review_flag"] = row.get("component_review_flag", "")
        sid = source_id_from_pdf(row.get("source_pdf", ""))
        page = row.get("pdf_page", "")
        classifier = by_stem.get(image_stem(row.get("image_path", ""))) or by_source_page.get((sid, page), {})
        new.update(classifier)
        new["adj_source_key"] = source_key(new)
        new["source_pdf_id"] = sid
        new["page_kind"] = page_kind(new)
        new["table_family_key"] = table_family_key(new)
        new["condition_focus"] = condition_focus(new)
        new["quality_flags"] = "; ".join(row_quality_flags(new))
        new["_input_order"] = idx
        enriched.append(new)
    return enriched


def sort_key(row: dict[str, object]) -> tuple[str, int, int, str, int]:
    page = int_or_none(row.get("pdf_page"))
    page_sort = page if page is not None else 10**9
    same_page_rank = 0 if row.get("source_record_type") == "master_tracking_page" else 1
    return (
        str(row.get("adj_source_key") or ""),
        page_sort,
        same_page_rank,
        str(row.get("image_path") or row.get("imghash") or ""),
        int(row.get("_input_order") or 0),
    )


def basis_for_run(rows: list[dict[str, object]]) -> str:
    kinds = Counter(str(r.get("page_kind") or "") for r in rows)
    primary = kinds.most_common(1)[0][0] if kinds else "unknown"
    if primary == "main_age_civil_numeric":
        return "adjacency_main_age_civil_numeric_run"
    if primary == "derivative_selected_caste_race_tribe":
        return "adjacency_derivative_selected_caste_race_tribe_run"
    if primary == "context_or_blank":
        return "adjacency_context_or_blank_run"
    if primary == "offtarget_table":
        return "adjacency_offtarget_table_run"
    if len(rows) == 1:
        return "adjacency_singleton_or_unsequenced"
    return "adjacency_other_or_unknown_run"


def component_review_flags(rows: list[dict[str, object]]) -> str:
    flags: list[str] = []
    for row in rows:
        for field in ["edge_review_flags", "quality_flags"]:
            for flag in str(row.get(field) or "").split(";"):
                flag = flag.strip()
                if flag:
                    flags.append(flag)
    kinds = {str(r.get("page_kind") or "") for r in rows if r.get("page_kind")}
    if len(kinds) > 1:
        flags.append("mixed_page_kind_in_component")
    families = {str(r.get("table_family_key") or "") for r in rows if r.get("table_family_key")}
    if len(families) > 1:
        flags.append("mixed_table_family_in_component")
    pages = [str(r.get("pdf_page") or "") for r in rows]
    duplicate_pages = [p for p, n in Counter(p for p in pages if p).items() if n > 1]
    if duplicate_pages:
        flags.append("duplicate_pdf_pages")
    _, gap_n, _ = page_gap_summary(rows)
    if gap_n:
        flags.append("internal_numeric_page_gaps")
    if not any(r.get("source_pdf") for r in rows):
        flags.append("missing_source_pdf")
    return uniq_join(flags, limit=30)


def assign_components(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    page_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("adj_source_key") or "")].append(row)

    def flush(run: list[dict[str, object]]) -> None:
        if not run:
            return
        source = str(run[0].get("adj_source_key") or "")
        pages = [int_or_none(r.get("pdf_page")) for r in run]
        first_page = min([p for p in pages if p is not None], default="")
        last_page = max([p for p in pages if p is not None], default="")
        basis = basis_for_run(run)
        cid = stable_id("adj", source, basis, first_page, last_page, len(run), run[0].get("v3_component_id", ""), run[0].get("_input_order", ""))
        for pos, row in enumerate(run, start=1):
            row["component_id"] = cid
            row["component_basis"] = basis
            row["component_scope"] = "adjacency_rebuilt"
            row["position_in_component"] = pos
            page_rows.append(row)

        prange, gap_n, gaps = page_gap_summary(run)
        titles = [title_blob(r) for r in run]
        source_pdfs = [str(r.get("source_pdf") or "") for r in run]
        component_flags = component_review_flags(run)
        review_flag = "review" if component_flags else "ok"
        if basis == "adjacency_main_age_civil_numeric_run" and not component_flags:
            review_flag = "ok_main_numeric"
        component_rows.append({
            "component_id": cid,
            "component_scope": "adjacency_rebuilt",
            "component_basis": basis,
            "review_flag": review_flag,
            "component_review_flags": component_flags,
            "source_group_key": source,
            "n_records": len(run),
            "n_unique_source_pages": len({(r.get("source_pdf", ""), r.get("pdf_page", "")) for r in run if r.get("pdf_page")}),
            "page_range": prange,
            "numeric_gap_count_within_component": gap_n,
            "numeric_gaps_within_component": gaps,
            "years": uniq_join([r.get("year", "") for r in run]),
            "provinces": uniq_join([r.get("province", "") for r in run]),
            "source_pdfs": uniq_join(source_pdfs),
            "source_pdf_ids": uniq_join([r.get("source_pdf_id", "") for r in run]),
            "page_kind_counts": counter_json([r.get("page_kind", "") for r in run]),
            "classifier_page_class_counts": counter_json([r.get("classifier_page_class", "") for r in run]),
            "table_family_key_counts": counter_json([r.get("table_family_key", "") for r in run]),
            "primary_title": Counter(titles).most_common(1)[0][0] if titles else "",
            "n_distinct_titles": len({t for t in titles if t}),
            "mixed_titles": len({t for t in titles if t}) > 1,
            "row_records_from_json_pages": sum(int_or_none(r.get("rows_extracted")) or int_or_none(r.get("core_rows")) or 0 for r in run),
            "constraint_checks_total": sum(int_or_none(r.get("constraint_checks")) or 0 for r in run),
            "constraint_failed_total": sum(int_or_none(r.get("constraint_failed")) or 0 for r in run),
            "quality_flag_counts": counter_json(";".join(str(r.get("quality_flags") or "") for r in run).split(";")),
            "v3_component_ids": uniq_join([r.get("v3_component_id", "") for r in run], limit=10),
            "v3_component_basis_counts": counter_json([r.get("v3_component_basis", "") for r in run]),
            "example_images": uniq_join([r.get("image_path", "") for r in run], limit=4),
        })

    for source, source_rows in sorted(grouped.items()):
        run: list[dict[str, object]] = []
        prev: dict[str, object] | None = None
        for row in sorted(source_rows, key=sort_key):
            if not run or prev is None:
                row["edge_from_prev_action"] = "start"
                row["edge_from_prev_reason"] = "first_page_in_source_or_unsequenced_group" if not prev else "new_component"
                row["edge_review_flags"] = row.get("quality_flags", "")
                row["prev_pdf_page"] = ""
                row["prev_page_kind"] = ""
                row["prev_title"] = ""
                run = [row]
                prev = row
                continue

            join, reason, flags = compatible_adjacent(prev, row)
            flags.extend(row_quality_flags(row))
            row["prev_pdf_page"] = prev.get("pdf_page", "")
            row["prev_page_kind"] = prev.get("page_kind", "")
            row["prev_title"] = title_blob(prev)
            row["edge_from_prev_reason"] = reason
            row["edge_review_flags"] = uniq_join(flags, limit=20)
            edge_rows.append({
                "source_group_key": source,
                "prev_pdf_page": prev.get("pdf_page", ""),
                "pdf_page": row.get("pdf_page", ""),
                "prev_page_kind": prev.get("page_kind", ""),
                "page_kind": row.get("page_kind", ""),
                "prev_classifier_page_class": prev.get("classifier_page_class", ""),
                "classifier_page_class": row.get("classifier_page_class", ""),
                "prev_table_family_key": prev.get("table_family_key", ""),
                "table_family_key": row.get("table_family_key", ""),
                "edge_action": "join" if join else "break",
                "edge_reason": reason,
                "edge_review_flags": row["edge_review_flags"],
                "prev_title": title_blob(prev),
                "title": title_blob(row),
                "prev_image_path": prev.get("image_path", ""),
                "image_path": row.get("image_path", ""),
                "prev_v3_component_id": prev.get("v3_component_id", ""),
                "v3_component_id": row.get("v3_component_id", ""),
            })
            if join:
                row["edge_from_prev_action"] = "join"
                run.append(row)
            else:
                row["edge_from_prev_action"] = "break_start"
                flush(run)
                run = [row]
            prev = row
        flush(run)

    component_rows.sort(key=lambda r: (
        str(r.get("review_flag", "")) == "ok_main_numeric",
        str(r.get("component_basis", "")),
        str(r.get("source_group_key", "")),
        str(r.get("page_range", "")),
    ))
    page_rows.sort(key=lambda r: (str(r.get("source_group_key", "")), str(r.get("component_id", "")), int(r.get("position_in_component") or 0)))
    return component_rows, page_rows, edge_rows


def current_overassociation_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    summaries = read_csv(V3_SUMMARY)
    pages = read_csv(V3_PAGES)
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pages:
        by_component[row.get("component_id", "")].append(row)

    examples: list[dict[str, object]] = []
    for row in summaries:
        gap_n = int_or_none(row.get("numeric_gap_count_within_component")) or 0
        cid = row.get("component_id", "")
        component_pages = by_component.get(cid, [])
        titles = {
            strip_continuation_words(r.get("title") or r.get("core_ocr_title") or "")
            for r in component_pages
            if (r.get("title") or r.get("core_ocr_title"))
        }
        table_numbers = {r.get("table_number", "") for r in component_pages if r.get("table_number")}
        kindish = {r.get("core_page_class", "") for r in component_pages if r.get("core_page_class")}
        mixed_signal = len(titles) > 3 or len(table_numbers) > 1 or len(kindish) > 1
        if gap_n or mixed_signal:
            examples.append({
                "component_id": cid,
                "component_basis": row.get("component_basis", ""),
                "n_records": row.get("n_records", ""),
                "page_range": row.get("page_range", ""),
                "numeric_gap_count_within_component": gap_n,
                "numeric_gaps_within_component": row.get("numeric_gaps_within_component", ""),
                "n_distinct_titles_in_pages": len(titles),
                "n_table_numbers": len(table_numbers),
                "n_core_page_classes": len(kindish),
                "primary_title": row.get("primary_title", ""),
                "source_pdfs": row.get("source_pdfs", ""),
                "v3_review_flag": row.get("review_flag", ""),
                "v3_range_review_flag": row.get("range_review_flag", ""),
                "example_images": row.get("example_images", ""),
            })
    examples.sort(key=lambda r: (
        int(r.get("numeric_gap_count_within_component") or 0),
        int(r.get("n_distinct_titles_in_pages") or 0),
        int(r.get("n_records") or 0),
    ), reverse=True)
    summary = {
        "v3_components_total": len(summaries),
        "v3_components_with_numeric_page_gaps": sum(1 for r in summaries if (int_or_none(r.get("numeric_gap_count_within_component")) or 0) > 0),
        "v3_components_with_gap_or_mixed_signal": len(examples),
        "v3_heuristic_or_core_components_with_numeric_page_gaps": sum(
            1
            for r in summaries
            if (int_or_none(r.get("numeric_gap_count_within_component")) or 0) > 0
            and (str(r.get("component_basis", "")).startswith("heuristic") or "core_money" in str(r.get("component_basis", "")))
        ),
    }
    return examples, summary


def autosize(ws, fields: list[str], rows: list[dict[str, object]], max_width: int = 70) -> None:
    widths = {field: min(max(len(field) + 2, 10), max_width) for field in fields}
    for row in rows[:2000]:
        for field in fields:
            widths[field] = min(max(widths[field], len(str(row.get(field, ""))) + 2), max_width)
    for i, field in enumerate(fields):
        ws.set_column(i, i, widths[field])


def add_sheet(wb: xlsxwriter.Workbook, name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    ws = wb.add_worksheet(name[:31])
    header_fmt = wb.add_format({"bold": True, "bg_color": "#EDEDED", "border": 1})
    text_fmt = wb.add_format({"text_wrap": False, "valign": "top"})
    for col, field in enumerate(fields):
        ws.write(0, col, field, header_fmt)
    for r, row in enumerate(rows, start=1):
        for c, field in enumerate(fields):
            value = row.get(field, "")
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, sort_keys=True)
            ws.write(r, c, value, text_fmt)
    ws.freeze_panes(1, 0)
    if rows:
        ws.autofilter(0, 0, len(rows), len(fields) - 1)
    autosize(ws, fields, rows)


def write_workbook(
    path: Path,
    components: list[dict[str, object]],
    pages: list[dict[str, object]],
    edges: list[dict[str, object]],
    overassoc: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    readme = wb.add_worksheet("README")
    lines = [
        ("purpose", "Adjacency-first component assembly. Same title/source/family never merges across non-adjacent or incompatible pages."),
        ("join rule", "A page joins the previous page only when source matches, pdf_page is adjacent or same physical page duplicate, and page kind/table structure is compatible."),
        ("break rule", "Breaks at page gaps, selected caste/race/tribe derivative boundaries, context/blank boundaries, table/subject switches, and uncertain schema/title transitions."),
        ("edge audit", "Pages sheet has edge_from_prev_action/reason/flags. Edges sheet has every pairwise join/break decision."),
        ("classifier", f"Gemini classifier signals loaded from {CLASSIFIER_CSV.relative_to(ROOT) if CLASSIFIER_CSV.exists() else CLASSIFIER_CSV}."),
    ]
    for i, (k, v) in enumerate(lines):
        readme.write(i, 0, k)
        readme.write(i, 1, v)

    component_fields = [
        "component_id", "component_scope", "component_basis", "review_flag", "component_review_flags",
        "source_group_key", "n_records", "n_unique_source_pages", "page_range",
        "numeric_gap_count_within_component", "numeric_gaps_within_component", "years", "provinces",
        "source_pdfs", "source_pdf_ids", "page_kind_counts", "classifier_page_class_counts",
        "table_family_key_counts", "primary_title", "n_distinct_titles", "mixed_titles",
        "row_records_from_json_pages", "constraint_checks_total", "constraint_failed_total",
        "quality_flag_counts", "v3_component_ids", "v3_component_basis_counts", "example_images",
    ]
    page_fields = [
        "component_id", "position_in_component", "component_scope", "component_basis",
        "edge_from_prev_action", "edge_from_prev_reason", "edge_review_flags",
        "prev_pdf_page", "prev_page_kind", "page_kind", "table_family_key", "condition_focus",
        "classifier_page_class", "classifier_contains_numeric_grid", "classifier_printed_title", "classifier_why",
        "quality_flags", "source_record_type", "year", "province", "source_pdf", "source_pdf_id", "pdf_page", "printed_page",
        "family", "title", "core_ocr_title", "table_number", "table_dir_key", "image_path", "ocr_artifact",
        "ocr_status", "page_status", "validation", "verdict", "rows_extracted", "sections_extracted",
        "constraint_checks", "constraint_failed", "v3_component_id", "v3_component_basis",
        "v3_component_review_flag", "range_review_flag", "source_group_key", "adj_source_key",
    ]
    edge_fields = [
        "source_group_key", "prev_pdf_page", "pdf_page", "edge_action", "edge_reason",
        "edge_review_flags", "prev_page_kind", "page_kind", "prev_classifier_page_class",
        "classifier_page_class", "prev_table_family_key", "table_family_key", "prev_title",
        "title", "prev_image_path", "image_path", "prev_v3_component_id", "v3_component_id",
    ]
    overassoc_fields = [
        "component_id", "component_basis", "n_records", "page_range", "numeric_gap_count_within_component",
        "numeric_gaps_within_component", "n_distinct_titles_in_pages", "n_table_numbers",
        "n_core_page_classes", "primary_title", "source_pdfs", "v3_review_flag",
        "v3_range_review_flag", "example_images",
    ]
    review_components = [r for r in components if r.get("review_flag") != "ok_main_numeric" and r.get("review_flag") != "ok"]
    main_components = [r for r in components if r.get("component_basis") == "adjacency_main_age_civil_numeric_run"]
    review_pages = [r for r in pages if str(r.get("edge_review_flags") or r.get("quality_flags") or "").strip()]

    add_sheet(wb, "Components", components, component_fields)
    add_sheet(wb, "MainAgeCivilRuns", main_components, component_fields)
    add_sheet(wb, "ReviewComponents", review_components, component_fields)
    add_sheet(wb, "Pages", pages, page_fields)
    add_sheet(wb, "ReviewPages", review_pages, page_fields)
    add_sheet(wb, "Edges", edges, edge_fields)
    add_sheet(wb, "V3OverAssocExamples", overassoc[:1000], overassoc_fields)
    wb.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_pages = read_csv(V3_PAGES)
    enriched = enrich_rows(raw_pages)
    components, pages, edges = assign_components(enriched)
    overassoc, overassoc_summary = current_overassociation_report()

    component_fields = [
        "component_id", "component_scope", "component_basis", "review_flag", "component_review_flags",
        "source_group_key", "n_records", "n_unique_source_pages", "page_range",
        "numeric_gap_count_within_component", "numeric_gaps_within_component", "years", "provinces",
        "source_pdfs", "source_pdf_ids", "page_kind_counts", "classifier_page_class_counts",
        "table_family_key_counts", "primary_title", "n_distinct_titles", "mixed_titles",
        "row_records_from_json_pages", "constraint_checks_total", "constraint_failed_total",
        "quality_flag_counts", "v3_component_ids", "v3_component_basis_counts", "example_images",
    ]
    page_fields = [
        "component_id", "position_in_component", "component_scope", "component_basis",
        "edge_from_prev_action", "edge_from_prev_reason", "edge_review_flags",
        "prev_pdf_page", "prev_page_kind", "page_kind", "table_family_key", "condition_focus",
        "classifier_page_class", "classifier_contains_numeric_grid", "classifier_printed_title", "classifier_why",
        "quality_flags", "source_record_type", "year", "province", "source_pdf", "source_pdf_id", "pdf_page", "printed_page",
        "family", "title", "core_ocr_title", "table_number", "table_dir_key", "image_path", "ocr_artifact",
        "ocr_status", "page_status", "validation", "verdict", "rows_extracted", "sections_extracted",
        "constraint_checks", "constraint_failed", "v3_component_id", "v3_component_basis",
        "v3_component_review_flag", "range_review_flag", "source_group_key", "adj_source_key",
    ]
    edge_fields = [
        "source_group_key", "prev_pdf_page", "pdf_page", "edge_action", "edge_reason",
        "edge_review_flags", "prev_page_kind", "page_kind", "prev_classifier_page_class",
        "classifier_page_class", "prev_table_family_key", "table_family_key", "prev_title",
        "title", "prev_image_path", "image_path", "prev_v3_component_id", "v3_component_id",
    ]
    overassoc_fields = [
        "component_id", "component_basis", "n_records", "page_range", "numeric_gap_count_within_component",
        "numeric_gaps_within_component", "n_distinct_titles_in_pages", "n_table_numbers",
        "n_core_page_classes", "primary_title", "source_pdfs", "v3_review_flag",
        "v3_range_review_flag", "example_images",
    ]

    write_csv(OUT_DIR / "all_ocr_component_summary.csv", components, component_fields)
    write_csv(OUT_DIR / "all_ocr_component_pages.csv", pages, page_fields)
    write_csv(OUT_DIR / "all_ocr_component_edges.csv", edges, edge_fields)
    write_csv(OUT_DIR / "v3_overassociation_examples.csv", overassoc, overassoc_fields)

    summary = {
        **overassoc_summary,
        "v4_components_total": len(components),
        "v4_page_records_total": len(pages),
        "v4_edges_total": len(edges),
        "v4_component_basis_counts": dict(Counter(str(r.get("component_basis", "")) for r in components)),
        "v4_review_flag_counts": dict(Counter(str(r.get("review_flag", "")) for r in components)),
        "v4_page_kind_counts": dict(Counter(str(r.get("page_kind", "")) for r in pages)),
        "v4_classifier_page_class_counts": dict(Counter(str(r.get("classifier_page_class", "")) for r in pages)),
        "v4_edge_action_counts": dict(Counter(str(r.get("edge_action", "")) for r in edges)),
        "v4_edge_review_flag_counts": dict(Counter(flag.strip() for r in edges for flag in str(r.get("edge_review_flags", "")).split(";") if flag.strip())),
        "outputs": {
            "components_csv": str(OUT_DIR / "all_ocr_component_summary.csv"),
            "pages_csv": str(OUT_DIR / "all_ocr_component_pages.csv"),
            "edges_csv": str(OUT_DIR / "all_ocr_component_edges.csv"),
            "workbook": str(OUT_DIR / "all_ocr_component_assembly_review.xlsx"),
        },
    }
    (OUT_DIR / "all_ocr_component_assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_workbook(OUT_DIR / "all_ocr_component_assembly_review.xlsx", components, pages, edges, overassoc)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
