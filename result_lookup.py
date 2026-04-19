"""Utilities for locating existing extracted results for an image."""

import json
from pathlib import Path

from batch_production import compute_output_dir, compute_result_json_path


DEFAULT_RESULTS_ROOTS = ["results"]


def parse_results_roots(value):
    if not value:
        return [Path(p) for p in DEFAULT_RESULTS_ROOTS]
    if isinstance(value, (list, tuple)):
        return [Path(p) for p in value]
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def find_result_json_for_image(image_path, results_roots):
    """Return the first matching result JSON and loaded payload for an image."""
    image_path = Path(image_path)
    for root in parse_results_roots(results_roots):
        out_dir = compute_output_dir(image_path, root)
        candidate = compute_result_json_path(image_path, out_dir)
        if candidate.exists():
            with open(candidate) as f:
                return candidate, json.load(f)

        flat_candidate = compute_result_json_path(image_path, root)
        if flat_candidate != candidate and flat_candidate.exists():
            with open(flat_candidate) as f:
                return flat_candidate, json.load(f)

    return None, None


def find_all_results_for_image(image_path, results_roots):
    """Return the oneshot result for an image (if found)."""
    image_path = Path(image_path)
    found = {}
    for root in parse_results_roots(results_roots):
        out_dir = compute_output_dir(image_path, root)
        candidate = compute_result_json_path(image_path, out_dir)
        if candidate.exists():
            with open(candidate) as f:
                found["oneshot"] = {"path": str(candidate), "data": json.load(f)}
            return found

        flat_candidate = compute_result_json_path(image_path, root)
        if flat_candidate != candidate and flat_candidate.exists():
            with open(flat_candidate) as f:
                found["oneshot"] = {"path": str(flat_candidate), "data": json.load(f)}
            return found

    return found
