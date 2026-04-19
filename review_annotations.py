"""Utilities for loading human review annotations."""

import json
from pathlib import Path


DEFAULT_ANNOTATIONS_ROOT = Path("review_annotations")


def annotation_path_for_image(image_path, annotations_root=DEFAULT_ANNOTATIONS_ROOT):
    image_path = Path(image_path)
    annotations_root = Path(annotations_root)
    cwd = Path.cwd()
    age_tables_root = cwd / "age_tables"
    try:
        rel = image_path.resolve().relative_to(age_tables_root.resolve())
    except Exception:
        rel = Path(image_path.name)
    return annotations_root / rel.with_suffix(".review.json")


def load_annotation_for_image(image_path, annotations_root=DEFAULT_ANNOTATIONS_ROOT):
    path = annotation_path_for_image(image_path, annotations_root)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def reviewed_data_for_result(result_json, annotations_root=DEFAULT_ANNOTATIONS_ROOT):
    image_path = result_json.get("source_image")
    if not image_path:
        return None
    annotation = load_annotation_for_image(image_path, annotations_root)
    if not annotation:
        return None
    if annotation.get("status") not in ("verified", "corrected"):
        return None
    return annotation.get("reviewed_data")


def apply_review_override(result_json, annotations_root=DEFAULT_ANNOTATIONS_ROOT):
    """Return result JSON with reviewed `data` substituted when available."""
    reviewed = reviewed_data_for_result(result_json, annotations_root)
    if reviewed is None:
        return result_json, None
    updated = dict(result_json)
    updated["data"] = reviewed
    updated["review_override"] = {
        "status": "applied",
        "annotations_root": str(annotations_root),
    }
    return updated, reviewed
