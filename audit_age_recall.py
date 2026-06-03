"""Audit recall of age-relevant tables across digitization, OCR, and inference.

This script is designed to answer three concrete questions:

1. Which age-relevant images in digitized_tables were never sent to / captured by OCR?
2. Which images were OCR'd but classified as non-age by the original organizer audit?
3. Which images were OCR'd but later dropped or kept only under a coarse family in the
   inference export?

The audit is page/image-level because OCR outputs are page-level. It distinguishes:

- classic_primary_candidate:
    Table VII style age-by-sex / civil-condition tables that the original pipeline
    was explicitly built for.
- age_structured_tail_candidate:
    Broader age-structured tables that can still matter for inference
    (age distribution, religion-by-age, literacy/education-by-age, etc.).

Outputs:
  - summary_overall.json
  - summary_by_province_year.csv
  - summary_by_topic.csv
  - candidate_inventory.csv
  - never_ocrd_candidates.csv
  - ocr_non_age_candidates.csv
  - dropped_by_inference_qc.csv
  - kept_tail_candidates.csv
  - never_ocrd_review_queue.txt
  - non_age_review_queue.txt
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from audit_ocr_results import classify_result
from build_digitized_ocr_manifest import (
    DEFAULT_STRICT_TOPICS,
    iter_selected_images,
    load_digitized_table,
    strict_age_dir_name_matches,
)


PRIMARY_TOPICS = {
    "age_sex_population",
    "age_sex_civil_condition",
    "age_sex_marital_status",
}

BROAD_AGE_TOPICS = PRIMARY_TOPICS | {
    "age_civil_condition",
    "age_by_religion",
    "age_distribution",
    "age_literacy_education",
}

TAIL_FACETS = {
    "religion",
    "civil_condition",
    "marital_status",
    "literacy",
    "education",
    "workers",
    "occupation",
    "language",
    "caste_tribe",
}

PRIMARY_TITLE_PATTERNS = [
    re.compile(r"\b(?:imperial\s+)?table\s+(?:vii|vil|vi1|v1i|7)\b", re.I),
    re.compile(r"\bage[-\s]*sex[-\s]*and[-\s]*civil[-\s]*condition\b", re.I),
    re.compile(r"\bpopulation\s+by\s+sex[, ]+\s*age\b", re.I),
    re.compile(r"\bage[-\s]*sex[-\s]*and[-\s]*marital", re.I),
]

PRIMARY_DIR_PATTERNS = [
    re.compile(r"\b(?:imperial[-_ .]*)?table[-_ .]*(?:vii|vii\.|vil|vi1|v1i|7)\b", re.I),
    re.compile(r"age[-_ .]*sex[-_ .]*and[-_ .]*civil[-_ .]*condition", re.I),
    re.compile(r"age[-_ .]*sex[-_ .]*and[-_ .]*marital", re.I),
]

BROAD_TITLE_PATTERNS = PRIMARY_TITLE_PATTERNS + [
    re.compile(r"\bage[-\s]*distribution\b", re.I),
    re.compile(r"\bage\b.*\breligion\b", re.I),
    re.compile(r"\bliteracy\b.*\bage\b", re.I),
    re.compile(r"\beducation\b.*\bage\b", re.I),
    re.compile(r"\boccupation\b.*\bage\b", re.I),
    re.compile(r"\blanguage\b.*\bage\b", re.I),
    re.compile(r"\bcaste\b.*\bage\b", re.I),
    re.compile(r"\btribe\b.*\bage\b", re.I),
]

RESULT_PRIORITY = {
    "age_table_validated": 5,
    "age_table_unvalidated": 4,
    "age_table_failed": 3,
    "non_age_table": 2,
    "corrupt": 1,
}

CORE_FAMILIES = {
    "age_sex_population",
    "age_civil_condition",
    "age_sex_marital_status",
    "age_civil_condition_religion",
}

TAIL_FAMILIES = {
    "age_distribution",
    "age_by_religion",
    "age_civil_condition_distribution",
    "proportional_age",
    "education_age",
    "education_religion",
    "occupation_age",
    "caste_class_age",
    "language_age",
    "infirmity_age",
    "single_year_age",
    "life_table",
    "birth_death",
}


def normalize_text(value):
    text = str(value or "").strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(value):
    text = normalize_text(value).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def path_keys(path_value):
    path_value = str(path_value or "").strip()
    if not path_value:
        return []
    path_obj = Path(path_value)
    keys = [path_value, path_obj.name]
    if len(path_obj.parts) >= 2:
        keys.append("/".join(path_obj.parts[-2:]))
    if len(path_obj.parts) >= 3:
        keys.append("/".join(path_obj.parts[-3:]))
    return keys


def coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            return loaded
        return [part.strip() for part in re.split(r"[|;,]", text) if part.strip()]
    return [str(value)]


def title_matches_any(text, patterns):
    text = str(text or "")
    return any(pattern.search(text) for pattern in patterns)


def rendered_image_rows(table_dir, digitized):
    if digitized:
        for rendered in digitized.get("rendered_pages") or []:
            image_path = rendered.get("image_path")
            if image_path:
                yield str(image_path)
        return
    for image_path in sorted(table_dir.glob("*.png")):
        yield str(image_path)


def assess_age_relevance(queue_row, table_dir_name):
    topic = queue_row.get("canonical_topic") or "unknown"
    relevance = queue_row.get("relevance_tier") or "unknown"
    query_tags = [normalize_token(tag) for tag in coerce_list(queue_row.get("query_tags"))]
    title = queue_row.get("display_identifier") or queue_row.get("title") or table_dir_name
    dir_name_match = strict_age_dir_name_matches(table_dir_name)
    primary_dir_match = title_matches_any(table_dir_name, PRIMARY_DIR_PATTERNS)

    evidence_flags = []
    if dir_name_match:
        evidence_flags.append("dir_name_match")
    if primary_dir_match:
        evidence_flags.append("primary_dir_signal")
    if topic in PRIMARY_TOPICS:
        evidence_flags.append("primary_topic")
    if topic in BROAD_AGE_TOPICS:
        evidence_flags.append("broad_topic")
    if relevance == "core":
        evidence_flags.append("core_relevance")
    if "age" in query_tags:
        evidence_flags.append("age_tag")
    if any(tag in TAIL_FACETS for tag in query_tags):
        evidence_flags.append("age_x_dimension")
    if title_matches_any(title, PRIMARY_TITLE_PATTERNS):
        evidence_flags.append("primary_title_signal")
    if title_matches_any(title, BROAD_TITLE_PATTERNS):
        evidence_flags.append("broad_title_signal")

    classic_primary = bool(
        topic in PRIMARY_TOPICS
        or "primary_title_signal" in evidence_flags
        or "primary_dir_signal" in evidence_flags
    )
    broad_age_relevant = bool(
        classic_primary
        or topic in BROAD_AGE_TOPICS
        or "age_tag" in evidence_flags
        or "broad_title_signal" in evidence_flags
        or dir_name_match
    )
    tail_candidate = broad_age_relevant and not classic_primary

    score = 0
    score += 5 if topic in PRIMARY_TOPICS else 0
    score += 3 if dir_name_match else 0
    score += 2 if "age_tag" in evidence_flags else 0
    score += 2 if "primary_title_signal" in evidence_flags else 0
    score += 1 if topic in BROAD_AGE_TOPICS else 0
    score += 1 if "age_x_dimension" in evidence_flags else 0

    if classic_primary:
        candidate_level = "classic_primary_candidate"
    elif tail_candidate:
        candidate_level = "age_structured_tail_candidate"
    else:
        candidate_level = "not_age_relevant"

    return {
        "title": title,
        "canonical_topic": topic,
        "relevance_tier": relevance,
        "query_tags": query_tags,
        "dir_name_match": int(dir_name_match),
        "classic_primary_candidate": int(classic_primary),
        "age_structured_tail_candidate": int(tail_candidate),
        "broad_age_relevant": int(broad_age_relevant),
        "candidate_level": candidate_level,
        "evidence_flags": evidence_flags,
        "evidence_score": score,
    }


def build_strict_selection_set(digitized_root):
    selected = set()
    rows = iter_selected_images(
        Path(digitized_root).expanduser(),
        include_topics=set(DEFAULT_STRICT_TOPICS),
        include_relevance={"core"},
        strict_age_only=True,
    )
    for row in rows:
        for key in path_keys(row.get("image_path")):
            selected.add(key)
    return selected


def choose_better_result(existing, candidate):
    if existing is None:
        return candidate
    if RESULT_PRIORITY[candidate["classification"]] > RESULT_PRIORITY[existing["classification"]]:
        return candidate
    if RESULT_PRIORITY[candidate["classification"]] < RESULT_PRIORITY[existing["classification"]]:
        return existing
    try:
        existing_checks = int(existing.get("total_checks") or 0)
    except Exception:
        existing_checks = 0
    try:
        candidate_checks = int(candidate.get("total_checks") or 0)
    except Exception:
        candidate_checks = 0
    if candidate_checks > existing_checks:
        return candidate
    return existing


def build_result_index(results_root, progress_every=500):
    results_root = Path(results_root).expanduser()
    files = sorted(results_root.rglob("*_oneshot.json"))
    index = {}
    unlinked = 0
    for idx, path in enumerate(files, 1):
        if progress_every and idx % progress_every == 0:
            print(f"[results {idx}/{len(files)}] indexed")
        payload = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            classification, info = classify_result(payload)
            source_image = payload.get("source_image") or ""
        except Exception:
            classification = "corrupt"
            info = {"reason": "unreadable_json"}
            source_image = ""
        if not source_image:
            unlinked += 1
            continue
        record = {
            "result_json": str(path),
            "source_image": source_image,
            "classification": classification,
            "info": info,
            "acceptance_decision": ((payload.get("acceptance") or {}).get("decision") if payload else "") or "",
            "total_checks": ((payload.get("constraints") or {}).get("total_checks") if payload else 0) or 0,
        }
        for key in path_keys(source_image):
            index[key] = choose_better_result(index.get(key), record)
    return index, {"result_jsons_scanned": len(files), "unlinked_results": unlinked}


def load_source_catalog_index(path):
    path_obj = Path(path).expanduser()
    if not path or not path_obj.exists():
        return {}, {"catalog_rows": 0}
    index = {}
    rows = 0
    with path_obj.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            for key in path_keys(row.get("source_image")):
                existing = index.get(key)
                if existing is None:
                    index[key] = row
                    continue
                existing_keep = int(existing.get("keep_for_inference") or 0)
                row_keep = int(row.get("keep_for_inference") or 0)
                if row_keep > existing_keep:
                    index[key] = row
    return index, {"catalog_rows": rows}


def scan_digitized_inventory(digitized_root, strict_selected_keys, result_index, source_catalog_index, progress_every=250):
    digitized_root = Path(digitized_root).expanduser()
    rows = []
    summary_by_py = defaultdict(Counter)
    summary_by_topic = defaultdict(Counter)

    table_dirs = sorted(path for path in digitized_root.iterdir() if path.is_dir())
    for idx, table_dir in enumerate(table_dirs, 1):
        if progress_every and idx % progress_every == 0:
            print(f"[digitized {idx}/{len(table_dirs)}] scanned")
        digitized = load_digitized_table(table_dir)
        queue_row = (digitized or {}).get("queue_row") or {}
        source_file = (digitized or {}).get("source_file") or ""
        relevance = assess_age_relevance(queue_row, table_dir.name)

        province = queue_row.get("province") or "unknown"
        year = queue_row.get("year") or "unknown"
        topic = relevance["canonical_topic"] or "unknown"

        for image_path in rendered_image_rows(table_dir, digitized):
            keys = path_keys(image_path)
            old_strict_selected = int(any(key in strict_selected_keys for key in keys))
            result_record = next((result_index[key] for key in keys if key in result_index), None)
            catalog_row = next((source_catalog_index[key] for key in keys if key in source_catalog_index), None)

            result_classification = result_record["classification"] if result_record else "no_result"
            result_info = result_record["info"] if result_record else {}
            inference_keep = int(catalog_row.get("keep_for_inference") or 0) if catalog_row else 0
            inference_family = (catalog_row.get("table_family") or "") if catalog_row else ""
            effective_quality_status = (catalog_row.get("effective_quality_status") or "") if catalog_row else ""
            qc_flags = (catalog_row.get("qc_flags") or "") if catalog_row else ""

            if not relevance["broad_age_relevant"]:
                audit_bucket = "not_age_relevant"
            elif not result_record:
                audit_bucket = "never_ocrd"
            elif result_classification == "non_age_table":
                audit_bucket = "ocr_non_age"
            elif result_classification == "corrupt":
                audit_bucket = "corrupt_result"
            elif inference_keep:
                if inference_family in CORE_FAMILIES:
                    audit_bucket = "kept_core"
                elif inference_family in TAIL_FAMILIES or relevance["age_structured_tail_candidate"]:
                    audit_bucket = "kept_tail"
                else:
                    audit_bucket = "kept_other_family"
            elif result_classification == "age_table_failed":
                audit_bucket = "ocr_age_failed"
            elif catalog_row is not None:
                audit_bucket = "dropped_by_inference_qc"
            else:
                audit_bucket = "ocr_age_untracked"

            row = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "table_dir": str(table_dir),
                "table_dir_name": table_dir.name,
                "source_file": source_file,
                "province": province,
                "year": year,
                "title": relevance["title"],
                "canonical_topic": topic,
                "relevance_tier": relevance["relevance_tier"],
                "query_tags": " | ".join(relevance["query_tags"]),
                "dir_name_match": relevance["dir_name_match"],
                "old_strict_selected": old_strict_selected,
                "classic_primary_candidate": relevance["classic_primary_candidate"],
                "age_structured_tail_candidate": relevance["age_structured_tail_candidate"],
                "broad_age_relevant": relevance["broad_age_relevant"],
                "candidate_level": relevance["candidate_level"],
                "evidence_score": relevance["evidence_score"],
                "evidence_flags": " | ".join(relevance["evidence_flags"]),
                "result_classification": result_classification,
                "result_reason": result_info.get("reason") or "",
                "result_age_rows": result_info.get("age_rows") or "",
                "result_total_rows": result_info.get("total_rows") or "",
                "result_checks": result_info.get("checks") or "",
                "result_failed_checks": result_info.get("failed") or "",
                "inference_keep": inference_keep,
                "inference_family": inference_family,
                "effective_quality_status": effective_quality_status,
                "inference_qc_flags": qc_flags,
                "audit_bucket": audit_bucket,
            }
            rows.append(row)

            py_counter = summary_by_py[(province, str(year))]
            py_counter["digitized_images"] += 1
            if relevance["broad_age_relevant"]:
                py_counter["broad_age_relevant"] += 1
            if relevance["classic_primary_candidate"]:
                py_counter["classic_primary_candidate"] += 1
            if relevance["age_structured_tail_candidate"]:
                py_counter["age_structured_tail_candidate"] += 1
            if old_strict_selected:
                py_counter["old_strict_selected"] += 1
            py_counter[audit_bucket] += 1

            topic_counter = summary_by_topic[topic]
            topic_counter["digitized_images"] += 1
            if relevance["broad_age_relevant"]:
                topic_counter["broad_age_relevant"] += 1
            if relevance["classic_primary_candidate"]:
                topic_counter["classic_primary_candidate"] += 1
            if relevance["age_structured_tail_candidate"]:
                topic_counter["age_structured_tail_candidate"] += 1
            if old_strict_selected:
                topic_counter["old_strict_selected"] += 1
            topic_counter[audit_bucket] += 1

    return rows, summary_by_py, summary_by_topic


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_queue(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row['image_path']}\n")


def main():
    parser = argparse.ArgumentParser(description="Audit recall of age-relevant tables across digitization, OCR, and inference")
    parser.add_argument("--digitized-root", default="~/census_toc/digitized_tables",
                        help="digitized_tables root")
    parser.add_argument("--results-root", default="~/census_toc/ocr_results",
                        help="raw OCR results root containing *_oneshot.json")
    parser.add_argument("--source-catalog", default="~/census_toc/export_inference_rich/source_catalog.csv",
                        help="Optional inference source catalog CSV from export_inference_dataset.py")
    parser.add_argument("--output-dir", default="~/census_toc/age_recall_audit",
                        help="Audit output directory")
    parser.add_argument("--progress-every", type=int, default=250,
                        help="Print progress every N table dirs/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_selected_keys = build_strict_selection_set(args.digitized_root)
    print(f"Old strict-pipeline selected image keys: {len(strict_selected_keys)}")

    result_index, result_meta = build_result_index(args.results_root, progress_every=args.progress_every)
    print(f"OCR-linked images indexed: {len(result_index)}")

    source_catalog_index, source_catalog_meta = load_source_catalog_index(args.source_catalog)
    print(f"Inference catalog rows indexed: {source_catalog_meta['catalog_rows']}")

    rows, summary_by_py, summary_by_topic = scan_digitized_inventory(
        args.digitized_root,
        strict_selected_keys,
        result_index,
        source_catalog_index,
        progress_every=args.progress_every,
    )

    candidate_rows = [row for row in rows if int(row["broad_age_relevant"])]
    never_ocrd = [row for row in candidate_rows if row["audit_bucket"] == "never_ocrd"]
    ocr_non_age = [row for row in candidate_rows if row["audit_bucket"] == "ocr_non_age"]
    dropped_qc = [row for row in candidate_rows if row["audit_bucket"] == "dropped_by_inference_qc"]
    kept_tail = [row for row in candidate_rows if row["audit_bucket"] == "kept_tail"]

    candidate_rows.sort(key=lambda row: (-int(row["evidence_score"]), row["province"], str(row["year"]), row["image_path"]))
    never_ocrd.sort(key=lambda row: (-int(row["evidence_score"]), row["province"], str(row["year"]), row["image_path"]))
    ocr_non_age.sort(key=lambda row: (-int(row["evidence_score"]), row["province"], str(row["year"]), row["image_path"]))
    dropped_qc.sort(key=lambda row: (-int(row["evidence_score"]), row["province"], str(row["year"]), row["image_path"]))
    kept_tail.sort(key=lambda row: (-int(row["evidence_score"]), row["province"], str(row["year"]), row["image_path"]))

    inventory_fields = [
        "image_path", "image_name", "table_dir", "table_dir_name", "source_file",
        "province", "year", "title", "canonical_topic", "relevance_tier", "query_tags",
        "dir_name_match", "old_strict_selected", "classic_primary_candidate",
        "age_structured_tail_candidate", "broad_age_relevant", "candidate_level",
        "evidence_score", "evidence_flags", "result_classification", "result_reason",
        "result_age_rows", "result_total_rows", "result_checks", "result_failed_checks",
        "inference_keep", "inference_family", "effective_quality_status",
        "inference_qc_flags", "audit_bucket",
    ]

    write_csv(output_dir / "candidate_inventory.csv", candidate_rows, inventory_fields)
    write_csv(output_dir / "never_ocrd_candidates.csv", never_ocrd, inventory_fields)
    write_csv(output_dir / "ocr_non_age_candidates.csv", ocr_non_age, inventory_fields)
    write_csv(output_dir / "dropped_by_inference_qc.csv", dropped_qc, inventory_fields)
    write_csv(output_dir / "kept_tail_candidates.csv", kept_tail, inventory_fields)

    py_rows = []
    for (province, year), counter in sorted(summary_by_py.items()):
        py_rows.append({
            "province": province,
            "year": year,
            **counter,
        })
    py_fields = [
        "province", "year",
        "digitized_images", "broad_age_relevant", "classic_primary_candidate",
        "age_structured_tail_candidate", "old_strict_selected",
        "never_ocrd", "ocr_non_age", "ocr_age_failed", "ocr_age_untracked",
        "dropped_by_inference_qc", "kept_core", "kept_tail", "kept_other_family",
        "corrupt_result", "not_age_relevant",
    ]
    write_csv(output_dir / "summary_by_province_year.csv", py_rows, py_fields)

    topic_rows = []
    for topic, counter in sorted(summary_by_topic.items()):
        topic_rows.append({
            "canonical_topic": topic,
            **counter,
        })
    topic_fields = [
        "canonical_topic",
        "digitized_images", "broad_age_relevant", "classic_primary_candidate",
        "age_structured_tail_candidate", "old_strict_selected",
        "never_ocrd", "ocr_non_age", "ocr_age_failed", "ocr_age_untracked",
        "dropped_by_inference_qc", "kept_core", "kept_tail", "kept_other_family",
        "corrupt_result", "not_age_relevant",
    ]
    write_csv(output_dir / "summary_by_topic.csv", topic_rows, topic_fields)

    strong_never_ocrd = [
        row for row in never_ocrd
        if int(row["evidence_score"]) >= 5
    ]
    strong_non_age = [
        row for row in ocr_non_age
        if int(row["evidence_score"]) >= 5
    ]
    write_review_queue(output_dir / "never_ocrd_review_queue.txt", strong_never_ocrd)
    write_review_queue(output_dir / "non_age_review_queue.txt", strong_non_age)

    overall = Counter(row["audit_bucket"] for row in candidate_rows)
    overall.update({
        "digitized_images_total": len(rows),
        "age_relevant_total": len(candidate_rows),
        "classic_primary_total": sum(int(row["classic_primary_candidate"]) for row in candidate_rows),
        "age_structured_tail_total": sum(int(row["age_structured_tail_candidate"]) for row in candidate_rows),
        "old_strict_selected_total": sum(int(row["old_strict_selected"]) for row in candidate_rows),
        "old_strict_missed_age_relevant": sum(1 for row in candidate_rows if not int(row["old_strict_selected"])),
        "review_queue_never_ocrd": len(strong_never_ocrd),
        "review_queue_non_age": len(strong_non_age),
        "results_unlinked": result_meta["unlinked_results"],
        "source_catalog_rows": source_catalog_meta["catalog_rows"],
    })
    with (output_dir / "summary_overall.json").open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2, sort_keys=True)

    print("\nRecall audit summary")
    print("--------------------")
    print(f"Digitized images scanned:            {len(rows)}")
    print(f"Broad age-relevant candidates:       {len(candidate_rows)}")
    print(f"Classic primary candidates:          {overall['classic_primary_total']}")
    print(f"Age-structured tail candidates:      {overall['age_structured_tail_total']}")
    print(f"Selected by old strict pipeline:     {overall['old_strict_selected_total']}")
    print(f"Missed by old strict pipeline:       {overall['old_strict_missed_age_relevant']}")
    print(f"Never OCR'd:                         {overall['never_ocrd']}")
    print(f"OCR'd but classified non-age:        {overall['ocr_non_age']}")
    print(f"OCR'd age but failed constraints:    {overall['ocr_age_failed']}")
    print(f"OCR'd age but not in inference:      {overall['dropped_by_inference_qc'] + overall['ocr_age_untracked']}")
    print(f"Kept core:                           {overall['kept_core']}")
    print(f"Kept tail:                           {overall['kept_tail']}")
    print(f"Strong never-OCR review queue:       {overall['review_queue_never_ocrd']}")
    print(f"Strong non-age review queue:         {overall['review_queue_non_age']}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
