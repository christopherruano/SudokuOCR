"""Classify and reorganize OCR outputs for the census table pipeline.

Goals:
  - preserve successful age-table outputs in an organized Province/Year layout
  - quarantine non-age results into a separate tree
  - keep failed/corrupt outputs available for reruns and audit
  - generate a filtered image list for future OCR runs by excluding images that
    already have successful age-table results

This script is copy-only by default and safe to rerun.
"""

import argparse
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from audit_ocr_results import classify_result
from build_digitized_ocr_manifest import iter_selected_images, load_digitized_table
from regenerate_excel_from_ocr_json import regenerate_excel_for_payload


SUCCESSFUL_AGE_CLASSES = {"age_table_validated", "age_table_unvalidated"}
ALL_AGE_CLASSES = SUCCESSFUL_AGE_CLASSES | {"age_table_failed"}
RESULT_EXTENSIONS = [".json", ".csv", ".xlsx"]
YEAR_RE = re.compile(r"\b(18[7-9]\d|19[0-6]\d)\b")


def safe_component(value, default="_unsorted") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w.\- ]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or default


def normalize_path_key(path):
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(Path(path).expanduser())


def iter_result_jsons(root):
    for path in sorted(root.rglob("*_oneshot.json")):
        if path.is_file():
            yield path


def related_artifacts(json_path):
    stem = json_path.stem
    yield json_path
    for suffix in RESULT_EXTENSIONS[1:]:
        sibling = json_path.with_suffix(suffix)
        if sibling.exists():
            yield sibling


def artifact_destination_path(dest_dir, result_path, artifact):
    return dest_dir / destination_filename(result_path, artifact)


def copy_if_needed(src, dst, overwrite=False, dry_run=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "skipped_existing"
    if dry_run:
        return "would_copy"
    shutil.copy2(src, dst)
    return "copied"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def source_table_dir_from_result(result_payload):
    src = result_payload.get("source_image")
    if not src:
        return None
    try:
        return Path(src).expanduser().resolve().parent
    except Exception:
        return Path(src).expanduser().parent


def load_digitized_context(result_payload):
    table_dir = source_table_dir_from_result(result_payload)
    if not table_dir:
        return None, {}
    digitized = load_digitized_table(table_dir)
    queue_row = (digitized or {}).get("queue_row") or {}
    return digitized, queue_row


def infer_year(result_payload, queue_row, digitized):
    for candidate in (
        queue_row.get("year"),
        result_payload.get("metadata", {}).get("year"),
    ):
        if candidate is None or candidate == "":
            continue
        match = YEAR_RE.search(str(candidate))
        if match:
            return match.group(1)

    for candidate in (
        (digitized or {}).get("source_file"),
        result_payload.get("source_image"),
        result_payload.get("metadata", {}).get("title"),
    ):
        if not candidate:
            continue
        match = YEAR_RE.search(str(candidate))
        if match:
            return match.group(1)
    return "_unsorted"


def infer_province(result_payload, queue_row, digitized):
    for candidate in (
        queue_row.get("province"),
        result_payload.get("metadata", {}).get("region"),
    ):
        if candidate:
            return str(candidate)

    for candidate in (
        result_payload.get("metadata", {}).get("title"),
        (digitized or {}).get("source_file"),
        result_payload.get("source_image"),
    ):
        if not candidate:
            continue
        text = str(candidate)
        if "district of" in text.lower():
            after = re.split(r"district of", text, flags=re.I, maxsplit=1)[-1]
            cleaned = re.split(r"[,:;()\-]", after, maxsplit=1)[0].strip()
            if cleaned:
                return cleaned
    return "_unsorted"


def target_root_for_class(classification, args):
    if classification in SUCCESSFUL_AGE_CLASSES:
        return Path(args.organized_root)
    if classification == "age_table_failed":
        return Path(args.failed_age_root)
    if classification == "non_age_table":
        return Path(args.non_age_root)
    return Path(args.corrupt_root)


def destination_dir_for_result(classification, province, year, result_path, args):
    root = target_root_for_class(classification, args)
    if classification == "non_age_table" and args.non_age_by_relative_path:
        return root / result_path.parent.name
    return root / safe_component(province) / safe_component(year)


def destination_filename(result_path, artifact):
    prefix = safe_component(result_path.parent.name, default="result")
    return f"{prefix}__{artifact.name}"


def manifest_row(result_path, classification, info, payload,
                 province, year, destinations, digitized,
                 queue_row, artifact_statuses, regenerated_excel=None):
    return {
        "result_json": str(result_path),
        "classification": classification,
        "classification_info": info,
        "province": province,
        "year": year,
        "source_image": (payload or {}).get("source_image"),
        "title": (payload or {}).get("metadata", {}).get("title"),
        "acceptance_decision": (payload or {}).get("acceptance", {}).get("decision"),
        "digitized_source_file": (digitized or {}).get("source_file"),
        "queue_row": queue_row,
        "destinations": destinations,
        "artifact_statuses": artifact_statuses,
        "regenerated_excel": regenerated_excel,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def build_future_image_rows(digitized_root, successful_images):
    if not digitized_root.exists():
        return
    seen = set()
    for row in iter_selected_images(digitized_root, include_topics=None, include_relevance={"core"}, strict_age_only=True):
        image_key = normalize_path_key(row.get("image_path"))
        if not image_key or image_key in successful_images or image_key in seen:
            continue
        seen.add(image_key)
        yield {
            "image_path": row["image_path"],
            "table_dir": row.get("table_dir"),
            "canonical_topic": row.get("canonical_topic"),
            "relevance_tier": row.get("relevance_tier"),
            "title": row.get("title"),
            "province": row.get("province"),
            "year": row.get("year"),
            "dir_name_match": row.get("dir_name_match"),
        }


def main():
    parser = argparse.ArgumentParser(description="Organize OCR outputs and prepare the next filtered OCR image list")
    parser.add_argument("results_root", nargs="?", default="~/census_toc/ocr_results",
                        help="OCR results root (default: ~/census_toc/ocr_results)")
    parser.add_argument("--digitized-root", default="~/census_toc/digitized_tables",
                        help="Digitized table root for metadata lookup and future image filtering")
    parser.add_argument("--organized-root", default="organized_results",
                        help="Destination root for successful age-table outputs")
    parser.add_argument("--non-age-root", default="non_age_results",
                        help="Destination root for non-age outputs")
    parser.add_argument("--failed-age-root", default="age_failed_results",
                        help="Destination root for age tables with failed checks")
    parser.add_argument("--corrupt-root", default="corrupt_results",
                        help="Destination root for corrupt/unreadable outputs")
    parser.add_argument("--future-image-list", default="pending_age_images.txt",
                        help="Output text file for remaining age-candidate images")
    parser.add_argument("--future-image-jsonl", default="pending_age_images.jsonl",
                        help="Output JSONL manifest for remaining age-candidate images")
    parser.add_argument("--manifest-jsonl", default="organized_results_manifest.jsonl",
                        help="Output JSONL manifest describing every classified OCR result")
    parser.add_argument("--progress-every", type=int, default=250,
                        help="Print progress every N results")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing copied artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan actions without copying files")
    parser.add_argument("--non-age-by-relative-path", action="store_true",
                        help="Store non-age outputs by original hash directory instead of Province/Year")
    parser.add_argument("--regenerate-excel", action="store_true",
                        help="Regenerate destination .xlsx files from JSON payloads instead of only copying existing workbooks")
    parser.add_argument("--regenerate-excel-scope", default="successful_age",
                        choices=["successful_age", "all_age", "all_non_corrupt"],
                        help="Which classified results should get regenerated Excel output when --regenerate-excel is set")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser()
    digitized_root = Path(args.digitized_root).expanduser()
    manifest_path = Path(args.manifest_jsonl).expanduser()
    future_image_list = Path(args.future_image_list).expanduser()
    future_image_jsonl = Path(args.future_image_jsonl).expanduser()

    result_files = list(iter_result_jsons(results_root))
    print(f"Results root: {results_root}")
    print(f"Digitized root: {digitized_root}")
    print(f"Result JSONs found: {len(result_files)}")

    stats = Counter()
    copy_stats = Counter()
    regen_stats = Counter()
    successful_images = set()
    manifest_rows = []

    for idx, result_path in enumerate(result_files, 1):
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[{idx}/{len(result_files)}] processed")

        artifact_statuses = {}
        destinations = []
        regenerated_excel = None
        province = "_unsorted"
        year = "_unsorted"
        digitized = None
        queue_row = {}

        try:
            payload = load_json(result_path)
            classification, info = classify_result(payload)
            digitized, queue_row = load_digitized_context(payload)
            province = infer_province(payload, queue_row, digitized)
            year = infer_year(payload, queue_row, digitized)
        except Exception as exc:
            payload = None
            classification = "corrupt"
            info = {"error": f"{type(exc).__name__}: {exc}"}

        stats[classification] += 1
        if classification in SUCCESSFUL_AGE_CLASSES:
            image_key = normalize_path_key((payload or {}).get("source_image"))
            if image_key:
                successful_images.add(image_key)

        should_regenerate_excel = False
        if args.regenerate_excel and payload is not None and classification != "corrupt":
            if args.regenerate_excel_scope == "successful_age":
                should_regenerate_excel = classification in SUCCESSFUL_AGE_CLASSES
            elif args.regenerate_excel_scope == "all_age":
                should_regenerate_excel = classification in ALL_AGE_CLASSES
            elif args.regenerate_excel_scope == "all_non_corrupt":
                should_regenerate_excel = True

        dest_dir = destination_dir_for_result(classification, province, year, result_path, args)
        for artifact in related_artifacts(result_path):
            if should_regenerate_excel and artifact.suffix.lower() == ".xlsx":
                continue
            dst = artifact_destination_path(dest_dir, result_path, artifact)
            status = copy_if_needed(artifact, dst, overwrite=args.overwrite, dry_run=args.dry_run)
            artifact_statuses[str(artifact)] = {"status": status, "destination": str(dst)}
            destinations.append(str(dst))
            copy_stats[status] += 1

        if should_regenerate_excel:
            target_xlsx = artifact_destination_path(dest_dir, result_path, result_path.with_suffix(".xlsx"))
            try:
                if args.dry_run:
                    regenerated_excel = {"status": "would_regenerate", "destination": str(target_xlsx)}
                else:
                    if target_xlsx.exists() and not args.overwrite:
                        regenerated_excel = {"status": "skipped_existing", "destination": str(target_xlsx)}
                    else:
                        regenerate_excel_for_payload(payload, target_xlsx)
                        regenerated_excel = {"status": "regenerated", "destination": str(target_xlsx)}
                regen_stats[regenerated_excel["status"]] += 1
            except Exception as exc:
                regenerated_excel = {
                    "status": "regeneration_failed",
                    "destination": str(target_xlsx),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                regen_stats["regeneration_failed"] += 1
            if str(target_xlsx) not in destinations:
                destinations.append(str(target_xlsx))

        manifest_rows.append(manifest_row(
            result_path=result_path,
            classification=classification,
            info=info,
            payload=payload,
            province=province,
            year=year,
            destinations=destinations,
            digitized=digitized,
            queue_row=queue_row,
            artifact_statuses=artifact_statuses,
            regenerated_excel=regenerated_excel,
        ))

    future_rows = list(build_future_image_rows(digitized_root, successful_images))
    future_image_list.parent.mkdir(parents=True, exist_ok=True)
    with open(future_image_list, "w", encoding="utf-8") as fh:
        for row in future_rows:
            fh.write(f"{row['image_path']}\n")
    write_jsonl(future_image_jsonl, future_rows)
    write_jsonl(manifest_path, manifest_rows)

    age_preserved = sum(stats[key] for key in SUCCESSFUL_AGE_CLASSES)
    print()
    print("=" * 70)
    print("OCR ORGANIZATION SUMMARY")
    print("=" * 70)
    print(f"Results scanned:           {len(result_files)}")
    print(f"Age preserved:             {age_preserved}")
    print(f"  age_table_validated:     {stats['age_table_validated']}")
    print(f"  age_table_unvalidated:   {stats['age_table_unvalidated']}")
    print(f"Age failed:                {stats['age_table_failed']}")
    print(f"Non-age quarantined:       {stats['non_age_table']}")
    print(f"Corrupt:                   {stats['corrupt']}")
    print(f"Artifacts copied:          {copy_stats['copied']}")
    print(f"Artifacts skipped:         {copy_stats['skipped_existing']}")
    if args.dry_run:
        print(f"Artifacts planned:         {copy_stats['would_copy']}")
    if args.regenerate_excel:
        print(f"Excel regenerated:         {regen_stats['regenerated']}")
        print(f"Excel regen skipped:       {regen_stats['skipped_existing']}")
        if args.dry_run:
            print(f"Excel regen planned:       {regen_stats['would_regenerate']}")
        print(f"Excel regen failed:        {regen_stats['regeneration_failed']}")
    print(f"Successful images known:   {len(successful_images)}")
    print(f"Future OCR images remain:  {len(future_rows)}")
    print()
    print(f"Manifest JSONL:            {manifest_path}")
    print(f"Future image list:         {future_image_list}")
    print(f"Future image JSONL:        {future_image_jsonl}")
    print(f"Organized age root:        {Path(args.organized_root).expanduser()}")
    print(f"Non-age root:              {Path(args.non_age_root).expanduser()}")
    print(f"Failed-age root:           {Path(args.failed_age_root).expanduser()}")
    print(f"Corrupt root:              {Path(args.corrupt_root).expanduser()}")


if __name__ == "__main__":
    main()
