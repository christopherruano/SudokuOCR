#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-/n/home11/cruano/census_toc}"
LOCAL_ROOT="${3:-./export/digitized_tables_referenced}"

mkdir -p "$LOCAL_ROOT"
tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

python3 - "$tmp_manifest" <<'PY'
import csv
import sys
csv.field_size_limit(10_000_000)

manifest_path = sys.argv[1]
images = set()
catalogs = [
    "export/cluster_sync/export_inference_final_core_curated_v2/source_catalog.csv",
    "export/cluster_sync/export_inference_tail_rescue_unreviewed_v2/source_catalog.csv",
]
remote_root = "/n/home11/cruano/census_toc/"

for p in catalogs:
    try:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            img = (r.get("source_image") or "").strip()
            if img.startswith(remote_root):
                images.add(img[len(remote_root):])
    except FileNotFoundError:
        print(f"  skipping {p} (not found)")

with open(manifest_path, "w") as f:
    for p in sorted(images):
        f.write(p + "\n")

print(f"Manifest: {len(images)} referenced images")
PY

echo "Syncing referenced images from $REMOTE"
rsync -av --files-from="$tmp_manifest" "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"

echo
echo "Synced into: $LOCAL_ROOT"
