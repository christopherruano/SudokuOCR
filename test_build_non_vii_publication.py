import csv

from build_non_vii_publication import (
    PROTECTED_RE,
    independent_constraints,
    publication_status,
    remediation,
    repair_determined_totals,
    semantic_category,
    payload_stats,
    write_tidy_csv,
)


def immigrant_payload():
    rows = []
    for label, total_m, total_f in (("A", 40, 60), ("B", 41, 59), ("C", 50, 50)):
        rows.append({
            "district": label,
            "Total": {"persons": 100, "males": total_m, "females": total_f},
            "Place 1": {"persons": 30, "males": 10, "females": 20},
            "Place 2": {"persons": 70, "males": 30, "females": 40},
        })
    return {
        "metadata": {
            "title": "Immigrants by age periods",
            "data_type": "absolute",
            "column_groups": ["Total", "Place 1", "Place 2"],
            "row_label_name": "district",
        },
        "data": [{"name": "Example", "rows": rows}],
    }


def test_determined_cross_group_repair_is_a_copy():
    source = immigrant_payload()
    repaired, repairs = repair_determined_totals(source)
    assert source["data"][0]["rows"][1]["Total"]["males"] == 41
    assert repaired["data"][0]["rows"][1]["Total"] == {"persons": 100, "males": 40, "females": 60}
    assert len(repairs) == 4
    validation = independent_constraints(repaired)
    assert validation["additive_failures"] == 0


def test_generic_rows_are_detected():
    payload = immigrant_payload()
    for index, row in enumerate(payload["data"][0]["rows"], 1):
        row["district"] = f"Row {index}"
    assert payload_stats(payload)["generic_label_ratio"] == 1.0


def test_nested_sex_axes_are_rejected_as_schema_collision():
    payload = {
        "metadata": {"title": "Statement", "column_groups": ["Both sexes", "Males", "Females"]},
        "data": [{"name": "Statement", "rows": [{
            "age": "0 year",
            "Both sexes": {"persons": 2497, "males": 2494, "females": 3},
            "Males": {"persons": 2508, "males": 2494, "females": 14},
            "Females": {"persons": 2485, "males": 2494, "females": -9},
        }]}],
    }
    assert payload_stats(payload)["sex_axis_collision"] is True


def test_repeated_serial_number_axis_is_rejected():
    payload = immigrant_payload()
    payload["metadata"]["row_label_name"] = "Number"
    payload["data"][0]["rows"][0]["district"] = "1"
    payload["data"][0]["rows"][1]["district"] = "1"
    assert payload_stats(payload)["repeated_serial_axis"] is True


def test_object_measure_columns_are_not_treated_as_sex_counts():
    payload = immigrant_payload()
    payload["metadata"]["title"] = "Ships and Tonnage"
    payload["metadata"]["column_groups"] = ["Foreign Ports"]
    for row in payload["data"][0]["rows"]:
        row["Foreign Ports"] = row.pop("Total")
        row.pop("Place 1")
        row.pop("Place 2")
    assert payload_stats(payload)["measure_axis_collision"] is True


def test_existing_sex_row_axis_rejects_nested_fixed_sex_fields():
    payload = immigrant_payload()
    payload["data"][0]["rows"][0]["district"] = "MALES."
    assert payload_stats(payload)["sex_axis_collision"] is True


def test_categorical_distribution_is_not_mislabeled_as_sex():
    payload = immigrant_payload()
    payload["metadata"]["data_type"] = "proportional"
    for row in payload["data"][0]["rows"]:
        for group in ("Total", "Place 1", "Place 2"):
            row[group] = {"persons": 600, "males": 300, "females": 100}
    assert payload_stats(payload)["categorical_triplet_axis"] is True


def test_enumeration_is_not_a_ratio_table():
    payload = immigrant_payload()
    payload["metadata"]["title"] = "Enumeration books"
    assert independent_constraints(payload)["table_kind"] == "count"


def test_agency_does_not_add_age_facet():
    payload = immigrant_payload()
    payload["metadata"]["title"] = "Enumeration books for the Central India Agency"
    _, _, facets = semantic_category(payload["metadata"]["title"], payload_stats(payload), {})
    assert "age" not in facets


def test_protected_table_number_variant_is_caught():
    assert PROTECTED_RE.search("TABLE No. VIL")
    assert PROTECTED_RE.search("hash_t0001_TABLE-VII")
    assert PROTECTED_RE.search("TABLE-VIIC")
    assert not PROTECTED_RE.search("TABLE-VIII")
    assert not PROTECTED_RE.search("TABLE-XVII")


def test_semantic_collision_gets_targeted_remediation():
    assert remediation("reread_required", ["measure_axis_schema_collision"]) == (
        2, "reextract_semantic_axes"
    )


def test_tidy_csv_preserves_total_row_as_age_group_header(tmp_path):
    payload = immigrant_payload()
    payload["data"][0]["rows"][0]["district"] = "Baroda State"
    payload["data"][0]["rows"][1]["district"] = "0-5"
    payload["data"][0]["rows"][2]["district"] = "5-15"
    target = tmp_path / "tidy.csv"
    write_tidy_csv(payload, target)
    with target.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["row_group"] == "Baroda State"


def test_category_uses_domain_before_table_number():
    payload = immigrant_payload()
    slug, _, facets = semantic_category(payload["metadata"]["title"], payload_stats(payload), {})
    assert slug == "migration_birthplace"
    assert "age" in facets


def test_verified_status_requires_clean_additive_checks():
    record = {
        "selected_json": "x.json",
        "source_image": "x.png",
        "frozen_status": "not_flagged",
        "remaining_frozen_columns": 0,
        "row_count": 3,
        "numeric_cells": 27,
        "generic_label_ratio": 0,
        "duplicate_labels": 0,
        "sex_axis_collision": 0,
        "repeated_serial_axis": 0,
        "measure_axis_collision": 0,
        "degenerate_sex_axis": 0,
        "categorical_triplet_axis": 0,
        "independent_additive_failures": 0,
        "unresolved_stored_failures": 0,
        "independent_additive_checks": 12,
        "table_kind": "count",
        "acceptance_decision": "accepted_high_confidence",
        "deterministic_repairs": 0,
        "category": "migration_birthplace",
        "generic_title": 0,
    }
    assert publication_status(record)[0] == "publishable_verified"
