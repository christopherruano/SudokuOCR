#!/usr/bin/env python3
"""Decompose bounded logical variant sets into ordered physical-run members.

No OCR/API calls are made. This pass is a supplemental normalization layer over
v17/v18. It does not change validated physical components. Its purpose is to
make old bounded variant sets explicit as logical collections with ordered
physical-run members, instead of leaving them as opaque inherited atoms.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import build_all_ocr_component_assembly_v5_structural as base
import build_all_ocr_component_assembly_v11_rule_audit as v11
import build_all_ocr_component_assembly_v13_scope_refined as v13


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v19_variant_decomposition"

V9_MEMBERS = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v9_split_title/v9_sequence_members.csv"
V9_EDGES = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v9_split_title/v9_sequence_edges.csv"
V10_VARIANTS = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v10_layered_manifest/v10_variant_sets.csv"
V17_LOGICAL_VARIANT_COLLECTIONS = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections/v17_logical_variant_collections.csv"
V17_LOGICAL_VARIANT_EDGES = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v17_collections/v17_logical_variant_collection_edges.csv"
V18_INVARIANTS = ROOT / "export_final/pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v18_quality_audit/v18_invariant_audit.csv"

COLLECTION_FIELDS = [
    "v19_variant_collection_id",
    "variant_source",
    "source_collection_id",
    "sequence_group_id",
    "v17_collection_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "all_pages_consecutive",
    "n_members",
    "n_physical_runs",
    "n_physical_pages",
    "religion_entities",
    "part_entities",
    "sex_entities",
    "table_number_keys",
    "primary_title",
    "sequence_join_types",
    "review_flags",
    "example_images",
]

MEMBER_FIELDS = [
    "v19_variant_collection_id",
    "position_in_variant_collection",
    "member_type",
    "physical_run_id",
    "v17_component_id",
    "sequence_group_id",
    "v5_source_key",
    "source_pdf",
    "source_pdf_id",
    "page_range",
    "n_physical_pages",
    "sequence_join_from_prev_type",
    "sequence_join_from_prev_reason",
    "sequence_join_from_prev_flags",
    "run_basis",
    "review_flag",
    "run_review_flags",
    "page_kind_counts",
    "table_family_counts",
    "primary_title",
    "example_images",
]

EDGE_FIELDS = [
    "v19_variant_collection_id",
    "variant_source",
    "edge_source",
    "sequence_edge_id",
    "collection_edge_id",
    "prev_member_id",
    "member_id",
    "v5_source_key",
    "prev_page_range",
    "page_range",
    "prev_last_pdf_page",
    "first_pdf_page",
    "page_gap",
    "missing_pages",
    "edge_action",
    "join_type",
    "edge_reason",
    "review_flags",
    "strict_title_similarity",
    "scope_similarity",
]

AUDIT_FIELDS = [
    "check",
    "status",
    "failure_count",
    "details",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pages(value: object) -> list[int]:
    return v11.parse_pages(value)


def all_consecutive(page_values: list[int]) -> bool:
    page_values = sorted(set(page_values))
    if not page_values:
        return False
    return all(b == a + 1 for a, b in zip(page_values, page_values[1:]))


def first_page(value: object) -> int:
    parsed = pages(value)
    return parsed[0] if parsed else 10**9


def last_page(value: object) -> int:
    parsed = pages(value)
    return parsed[-1] if parsed else -10**9


def read_v7_members_by_sequence(members: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member in members:
        out[str(member.get("sequence_group_id") or "")].append(member)
    for rows in out.values():
        rows.sort(key=lambda row: (v13.as_int(row.get("position_in_sequence")), first_page(row.get("page_range")), str(row.get("physical_run_id") or "")))
    return out


def edge_by_pair(edges: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(edge.get("prev_physical_run_id") or ""), str(edge.get("physical_run_id") or "")): edge
        for edge in edges
        if edge.get("prev_physical_run_id") and edge.get("physical_run_id")
    }


def inherited_collection_id(sequence_group_id: object) -> str:
    return base.stable_id("v19var", "inherited_v10", sequence_group_id)


def v17_collection_id(v17_id: object) -> str:
    return base.stable_id("v19var", "v17_logical", v17_id)


def build_inherited_variant_rows(
    variants: list[dict[str, str]],
    members_by_sequence: dict[str, list[dict[str, str]]],
    edges_by_pair: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    collections: list[dict[str, object]] = []
    members_out: list[dict[str, object]] = []
    edges_out: list[dict[str, object]] = []
    for variant in variants:
        sequence_group_id = str(variant.get("sequence_group_id") or "")
        rows = members_by_sequence.get(sequence_group_id, [])
        collection_id = inherited_collection_id(sequence_group_id)
        member_pages: list[int] = []
        for row in rows:
            member_pages.extend(pages(row.get("page_range")))
        collections.append(
            {
                "v19_variant_collection_id": collection_id,
                "variant_source": "inherited_v10_bounded_variant_set",
                "source_collection_id": variant.get("variant_set_id", ""),
                "sequence_group_id": sequence_group_id,
                "v17_collection_id": "",
                "v5_source_key": variant.get("v5_source_key", ""),
                "source_pdf": variant.get("source_pdf", ""),
                "source_pdf_id": "",
                "page_range": variant.get("page_range", ""),
                "all_pages_consecutive": str(all_consecutive(member_pages)),
                "n_members": len(rows),
                "n_physical_runs": variant.get("n_physical_runs", ""),
                "n_physical_pages": variant.get("n_physical_pages", ""),
                "religion_entities": variant.get("religion_entities", ""),
                "part_entities": variant.get("part_entities", ""),
                "sex_entities": variant.get("sex_entities", ""),
                "table_number_keys": variant.get("table_number_keys", ""),
                "primary_title": variant.get("primary_title", ""),
                "sequence_join_types": variant.get("sequence_join_types", ""),
                "review_flags": variant.get("sequence_review_flags", ""),
                "example_images": variant.get("example_images", ""),
            }
        )
        prev: dict[str, str] | None = None
        for idx, row in enumerate(rows, start=1):
            members_out.append(
                {
                    "v19_variant_collection_id": collection_id,
                    "position_in_variant_collection": idx,
                    "member_type": "physical_run",
                    "physical_run_id": row.get("physical_run_id", ""),
                    "v17_component_id": "",
                    "sequence_group_id": sequence_group_id,
                    "v5_source_key": row.get("v5_source_key", ""),
                    "source_pdf": row.get("source_pdf", ""),
                    "source_pdf_id": row.get("source_pdf_id", ""),
                    "page_range": row.get("page_range", ""),
                    "n_physical_pages": row.get("n_physical_pages", ""),
                    "sequence_join_from_prev_type": row.get("sequence_join_from_prev_type", ""),
                    "sequence_join_from_prev_reason": row.get("sequence_join_from_prev_reason", ""),
                    "sequence_join_from_prev_flags": row.get("sequence_join_from_prev_flags", ""),
                    "run_basis": row.get("run_basis", ""),
                    "review_flag": row.get("review_flag", ""),
                    "run_review_flags": row.get("run_review_flags", ""),
                    "page_kind_counts": row.get("page_kind_counts", ""),
                    "table_family_counts": row.get("table_family_counts", ""),
                    "primary_title": row.get("primary_title", ""),
                    "example_images": row.get("example_images", ""),
                }
            )
            if prev:
                edge = edges_by_pair.get((str(prev.get("physical_run_id") or ""), str(row.get("physical_run_id") or "")), {})
                edges_out.append(
                    {
                        "v19_variant_collection_id": collection_id,
                        "variant_source": "inherited_v10_bounded_variant_set",
                        "edge_source": "v7_sequence_edge",
                        "sequence_edge_id": edge.get("sequence_edge_id", ""),
                        "collection_edge_id": "",
                        "prev_member_id": prev.get("physical_run_id", ""),
                        "member_id": row.get("physical_run_id", ""),
                        "v5_source_key": row.get("v5_source_key", ""),
                        "prev_page_range": prev.get("page_range", ""),
                        "page_range": row.get("page_range", ""),
                        "prev_last_pdf_page": edge.get("prev_last_pdf_page", str(last_page(prev.get("page_range")) if last_page(prev.get("page_range")) > 0 else "")),
                        "first_pdf_page": edge.get("first_pdf_page", str(first_page(row.get("page_range")) if first_page(row.get("page_range")) < 10**9 else "")),
                        "page_gap": edge.get("page_gap", ""),
                        "missing_pages": edge.get("missing_pages", ""),
                        "edge_action": edge.get("sequence_edge_action", ""),
                        "join_type": edge.get("sequence_join_type", row.get("sequence_join_from_prev_type", "")),
                        "edge_reason": edge.get("sequence_edge_reason", row.get("sequence_join_from_prev_reason", "")),
                        "review_flags": edge.get("sequence_review_flags", row.get("sequence_join_from_prev_flags", "")),
                        "strict_title_similarity": edge.get("strict_title_similarity", ""),
                        "scope_similarity": edge.get("scope_similarity", ""),
                    }
                )
            prev = row
    return collections, members_out, edges_out


def build_v17_logical_rows(
    v17_collections: list[dict[str, str]],
    v17_edges: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    collections: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    edges_out: list[dict[str, object]] = []
    edges_by_collection: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in v17_edges:
        edges_by_collection[str(edge.get("prev_v17_component_id") or "")].append(edge)
    for collection in v17_collections:
        if str(collection.get("collection_kind") or "") != "bounded_logical_variant_set":
            continue
        collection_id = v17_collection_id(collection.get("v17_collection_id"))
        component_ids = [part.strip() for part in str(collection.get("v17_component_ids") or "").split(";") if part.strip()]
        collections.append(
            {
                "v19_variant_collection_id": collection_id,
                "variant_source": "v17_logical_variant_collection",
                "source_collection_id": "",
                "sequence_group_id": "",
                "v17_collection_id": collection.get("v17_collection_id", ""),
                "v5_source_key": collection.get("v5_source_key", ""),
                "source_pdf": collection.get("source_pdf", ""),
                "source_pdf_id": collection.get("source_pdf_id", ""),
                "page_range": collection.get("page_range", ""),
                "all_pages_consecutive": collection.get("all_pages_consecutive", ""),
                "n_members": len(component_ids),
                "n_physical_runs": "",
                "n_physical_pages": collection.get("n_physical_pages", ""),
                "religion_entities": "",
                "part_entities": "",
                "sex_entities": "",
                "table_number_keys": collection.get("table_number_keys", ""),
                "primary_title": collection.get("primary_title", ""),
                "sequence_join_types": collection.get("v17_collection_edge_actions", ""),
                "review_flags": collection.get("collection_review_flags", ""),
                "example_images": collection.get("example_images", ""),
            }
        )
        for idx, component_id in enumerate(component_ids, start=1):
            members.append(
                {
                    "v19_variant_collection_id": collection_id,
                    "position_in_variant_collection": idx,
                    "member_type": "v17_component",
                    "physical_run_id": "",
                    "v17_component_id": component_id,
                    "sequence_group_id": "",
                    "v5_source_key": collection.get("v5_source_key", ""),
                    "source_pdf": collection.get("source_pdf", ""),
                    "source_pdf_id": collection.get("source_pdf_id", ""),
                    "page_range": "",
                    "n_physical_pages": "",
                    "sequence_join_from_prev_type": "",
                    "sequence_join_from_prev_reason": "",
                    "sequence_join_from_prev_flags": "",
                    "run_basis": "",
                    "review_flag": "",
                    "run_review_flags": "",
                    "page_kind_counts": "",
                    "table_family_counts": "",
                    "primary_title": collection.get("primary_title", "") if idx == 1 else "",
                    "example_images": "",
                }
            )
        for edge in v17_edges:
            if str(edge.get("collection_edge_action") or "") != "group_logical_variant_sequence_high":
                continue
            edges_out.append(
                {
                    "v19_variant_collection_id": collection_id,
                    "variant_source": "v17_logical_variant_collection",
                    "edge_source": "v17_collection_edge",
                    "sequence_edge_id": edge.get("sequence_edge_id", ""),
                    "collection_edge_id": edge.get("collection_edge_id", ""),
                    "prev_member_id": edge.get("prev_v17_component_id", ""),
                    "member_id": edge.get("v17_component_id", ""),
                    "v5_source_key": edge.get("v5_source_key", ""),
                    "prev_page_range": edge.get("prev_component_page_range", ""),
                    "page_range": edge.get("component_page_range", ""),
                    "prev_last_pdf_page": edge.get("prev_last_pdf_page", ""),
                    "first_pdf_page": edge.get("first_pdf_page", ""),
                    "page_gap": edge.get("page_gap", ""),
                    "missing_pages": edge.get("missing_pages", ""),
                    "edge_action": edge.get("collection_edge_action", ""),
                    "join_type": edge.get("collection_join_type", ""),
                    "edge_reason": edge.get("collection_edge_reason", ""),
                    "review_flags": edge.get("collection_review_flags", ""),
                    "strict_title_similarity": edge.get("strict_title_similarity", ""),
                    "scope_similarity": edge.get("scope_similarity", ""),
                }
            )
    return collections, members, edges_out


def audit_rows(collections: list[dict[str, object]], edges: list[dict[str, object]]) -> list[dict[str, object]]:
    collection_gaps = [row for row in collections if str(row.get("all_pages_consecutive") or "") != "True"]
    empty_inherited_collections = [
        row for row in collections
        if str(row.get("variant_source") or "") == "inherited_v10_bounded_variant_set"
        and v13.as_int(row.get("n_members")) == 0
    ]
    edge_gaps = [
        row for row in edges
        if str(row.get("edge_action") or "") in {"join_sequence", "group_logical_variant_sequence_high"}
        and str(row.get("page_gap") or "") not in {"1", ""}
    ]
    missing_join_edges = [
        row for row in edges
        if str(row.get("variant_source") or "") == "inherited_v10_bounded_variant_set"
        and str(row.get("edge_action") or "") != "join_sequence"
    ]
    return [
        {
            "check": "inherited_variant_collections_have_members",
            "status": "pass" if not empty_inherited_collections else "fail",
            "failure_count": len(empty_inherited_collections),
            "details": json.dumps(empty_inherited_collections[:20], sort_keys=True),
        },
        {
            "check": "variant_collections_pages_consecutive",
            "status": "pass" if not collection_gaps else "fail",
            "failure_count": len(collection_gaps),
            "details": json.dumps(collection_gaps[:20], sort_keys=True),
        },
        {
            "check": "variant_join_edges_adjacent_or_component_boundary",
            "status": "pass" if not edge_gaps else "fail",
            "failure_count": len(edge_gaps),
            "details": json.dumps(edge_gaps[:20], sort_keys=True),
        },
        {
            "check": "inherited_variant_internal_edges_are_join_sequences",
            "status": "pass" if not missing_join_edges else "fail",
            "failure_count": len(missing_join_edges),
            "details": json.dumps(missing_join_edges[:20], sort_keys=True),
        },
    ]


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v19 Variant Decomposition",
        "",
        "No OCR/API calls are made. This supplemental pass decomposes older bounded variant sets into logical collections with ordered member rows.",
        "",
        "Use this layer when the downstream consumer needs the distinction between strict physical assembly and sibling table grouping.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    (OUT_DIR / "v19_rules_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv.field_size_limit(10**7)
    v9_members = read_csv(V9_MEMBERS)
    v9_edges = read_csv(V9_EDGES)
    variants = read_csv(V10_VARIANTS)
    v17_collections = read_csv(V17_LOGICAL_VARIANT_COLLECTIONS)
    v17_edges = read_csv(V17_LOGICAL_VARIANT_EDGES)
    v18_invariants = read_csv(V18_INVARIANTS)
    if not v9_members or not variants:
        raise SystemExit("Missing v9/v10 inputs; run earlier assembly passes first")

    members_by_sequence = read_v7_members_by_sequence(v9_members)
    inherited_collections, inherited_members, inherited_edges = build_inherited_variant_rows(
        variants,
        members_by_sequence,
        edge_by_pair(v9_edges),
    )
    v17_logical_collections, v17_logical_members, v17_logical_edges = build_v17_logical_rows(v17_collections, v17_edges)
    collections = [*inherited_collections, *v17_logical_collections]
    members = [*inherited_members, *v17_logical_members]
    edges = [*inherited_edges, *v17_logical_edges]
    audits = audit_rows(collections, edges)
    invariant_failures = sum(v13.as_int(row.get("failure_count")) for row in audits)
    source_counts = Counter(str(row.get("variant_source") or "") for row in collections)
    edge_source_counts = Counter(str(row.get("variant_source") or "") for row in edges)
    join_type_counts = Counter(str(row.get("join_type") or "") for row in edges)
    inherited_review_flags = Counter()
    for row in inherited_collections:
        for flag in str(row.get("review_flags") or "").split(";"):
            flag = flag.strip()
            if flag:
                inherited_review_flags[flag] += 1
    summary = {
        "v19_variant_collections": len(collections),
        "v19_variant_members": len(members),
        "v19_variant_edges": len(edges),
        "v19_collection_source_counts": dict(source_counts),
        "v19_edge_source_counts": dict(edge_source_counts),
        "v19_join_type_counts": dict(join_type_counts),
        "v19_inherited_review_flag_counts": dict(inherited_review_flags),
        "v19_invariant_failure_count": invariant_failures,
        "v19_invariant_status": "pass" if invariant_failures == 0 else "fail",
        "v19_v18_invariant_status_seen": base.uniq_join([row.get("status") for row in v18_invariants], limit=10),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_csv(OUT_DIR / "v19_variant_collections.csv", collections, COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v19_variant_members.csv", members, MEMBER_FIELDS)
    base.write_csv(OUT_DIR / "v19_variant_edges.csv", edges, EDGE_FIELDS)
    base.write_csv(OUT_DIR / "v19_variant_decomposition_audit.csv", audits, AUDIT_FIELDS)
    base.write_csv(OUT_DIR / "v19_inherited_v10_variant_collections.csv", inherited_collections, COLLECTION_FIELDS)
    base.write_csv(OUT_DIR / "v19_v17_logical_variant_collections.csv", v17_logical_collections, COLLECTION_FIELDS)
    with (OUT_DIR / "v19_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(summary)

    summary_rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    workbook = OUT_DIR / "v19_variant_decomposition.xlsx"
    base.write_workbook(
        workbook,
        {
            "summary": (summary_rows, ["metric", "value"]),
            "variant_collections": (collections, COLLECTION_FIELDS),
            "variant_members": (members, MEMBER_FIELDS),
            "variant_edges": (edges, EDGE_FIELDS),
            "audit": (audits, AUDIT_FIELDS),
            "inherited_v10_variants": (inherited_collections, COLLECTION_FIELDS),
            "v17_logical_variants": (v17_logical_collections, COLLECTION_FIELDS),
        },
    )
    with zipfile.ZipFile(workbook) as z:
        summary["v19_workbook_zip_bad_member"] = z.testzip()
    with (OUT_DIR / "v19_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
