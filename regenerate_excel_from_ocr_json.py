"""Regenerate Excel workbooks from OCR result JSON files.

This rebuilds publication-style `.xlsx` outputs from `_oneshot.json` payloads
using the same Excel exporter as the extraction pipeline.
"""

import argparse
import json
from pathlib import Path


TEXTUAL_ROW_FIELDS = {"age", "district", "location", "religion", "community", "row_label"}


def load_result_payload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def normalize_sections(data):
    if isinstance(data, list):
        return [section for section in data if isinstance(section, dict)]
    if isinstance(data, dict):
        if "rows" in data:
            return [{"name": data.get("name", "Section 1"), "rows": data.get("rows", [])}]
        return [{"name": "Section 1", "rows": [data]}]
    return []


def infer_row_label_name(sections):
    counts = {}
    for section in sections:
        for row in section.get("rows", []):
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, dict):
                    continue
                if key in ("district", "location", "religion", "community", "row_label", "age"):
                    return key
                counts[key] = counts.get(key, 0) + 1
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]
    return "age"


def infer_column_groups(sections, metadata):
    groups = metadata.get("column_groups") or []
    if groups:
        return [str(group) for group in groups]

    seen = []
    for section in sections:
        for row in section.get("rows", []):
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, dict) and key not in seen:
                    seen.append(key)
    return seen


def infer_has_persons_column(sections, groups):
    for section in sections:
        for row in section.get("rows", []):
            if not isinstance(row, dict):
                continue
            for group in groups:
                value = row.get(group)
                if isinstance(value, dict) and "persons" in value:
                    return True
    return False


def build_parsed_from_payload(payload):
    metadata = dict(payload.get("metadata") or {})
    sections = normalize_sections(payload.get("data"))
    row_label_name = metadata.get("row_label_name") or infer_row_label_name(sections)
    metadata.setdefault("row_label_name", row_label_name)

    if not metadata.get("row_axis_type"):
        if row_label_name == "age":
            metadata["row_axis_type"] = "age_rows"
        elif row_label_name == "district":
            metadata["row_axis_type"] = "district_rows"
        else:
            metadata["row_axis_type"] = "generic_rows"

    groups = infer_column_groups(sections, metadata)
    metadata["column_groups"] = groups
    metadata.setdefault("title", "")
    metadata.setdefault("region", "")
    metadata.setdefault("year", 0)
    metadata.setdefault("data_type", "absolute")
    metadata["has_persons_column"] = infer_has_persons_column(sections, groups)

    return {
        "metadata": metadata,
        "sections": sections,
    }


def build_export_inputs(parsed):
    from schema_discovery import TableSchema

    meta = parsed.get("metadata", {})
    sections = parsed.get("sections", [])
    groups = list(meta.get("column_groups") or [])
    row_label_name = meta.get("row_label_name", "age") or "age"

    schema = TableSchema(
        title=meta.get("title", ""),
        region=meta.get("region", ""),
        year=meta.get("year") or 0,
        data_type=meta.get("data_type", "absolute"),
        row_axis_type=meta.get("row_axis_type", "age_rows"),
        row_label_name=row_label_name,
        column_groups=[{"name": group} for group in groups],
        sections=[{"name": section.get("name", "")} for section in sections],
        has_persons_column=bool(meta.get("has_persons_column", True)),
    )

    all_results = {}
    for section_index, section in enumerate(sections, 1):
        sec_name = section.get("name") or f"Section {section_index}"
        all_results[sec_name] = {}
        rows = section.get("rows", [])
        for group in groups:
            out_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                gdata = row.get(group)
                if not isinstance(gdata, dict):
                    continue
                row_label = row.get(row_label_name)
                if row_label is None and row_label_name != "age":
                    row_label = row.get("age")
                record = {
                    row_label_name: row_label,
                    "age": row.get("age", row_label if row_label_name == "age" else ""),
                    "district": row.get("district"),
                    "persons": gdata.get("persons"),
                    "males": gdata.get("males"),
                    "females": gdata.get("females"),
                }
                if row_label_name not in ("age", "district"):
                    record[row_label_name] = row_label
                out_rows.append(record)
            all_results[sec_name][group] = {"rows": out_rows}

    return all_results, schema


def regenerate_excel_for_payload(payload, output_path):
    from pipeline import export_multigroup_to_excel

    parsed = build_parsed_from_payload(payload)
    all_results, schema = build_export_inputs(parsed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_multigroup_to_excel(all_results, schema, output_path)
    return output_path


def regenerate_excel_for_json(result_json_path, output_path=None):
    result_json_path = Path(result_json_path)
    if output_path is None:
        output_path = result_json_path.with_suffix(".xlsx")
    payload = load_result_payload(result_json_path)
    return regenerate_excel_for_payload(payload, Path(output_path))


def iter_result_jsons(root):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*_oneshot.json")):
        yield path


def main():
    parser = argparse.ArgumentParser(description="Regenerate Excel workbooks from OCR result JSON files")
    parser.add_argument("input", help="Result JSON path or root directory")
    parser.add_argument("--output", help="Output .xlsx path (single-file mode only)")
    parser.add_argument("--output-dir", help="Output directory root (directory mode)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_root = Path(args.output_dir).expanduser() if args.output_dir else None

    processed = 0
    skipped = 0
    for result_json in iter_result_jsons(input_path):
        if args.output and not input_path.is_file():
            raise SystemExit("--output can only be used with a single JSON file")
        if args.output:
            target = Path(args.output).expanduser()
        elif output_root:
            target = output_root / result_json.name.replace(".json", ".xlsx")
        else:
            target = result_json.with_suffix(".xlsx")

        if target.exists() and not args.overwrite:
            skipped += 1
            continue

        regenerate_excel_for_json(result_json, target)
        processed += 1
        print(f"REGENERATED {result_json} -> {target}")

    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")


if __name__ == "__main__":
    main()
