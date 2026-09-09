#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-/n/home11/cruano/census_toc}"
INPUT_CSV="${3:-export_final/browse/final_submission_review/review_decisions.csv}"
LOCAL_ASSET_ROOT="${4:-export_final/browse/final_submission_review/review_assets}"

mkdir -p "$LOCAL_ASSET_ROOT"
tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

python3 - "$INPUT_CSV" "$REMOTE_ROOT" "$tmp_manifest" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1]).expanduser()
remote_root = Path(sys.argv[2]).expanduser()
manifest_path = Path(sys.argv[3])

rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
paths = set()

for row in rows:
    for key in ("source_image", "source_json"):
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        try:
            rel = path.relative_to(remote_root)
        except Exception:
            continue
        paths.add(rel.as_posix())

with manifest_path.open("w", encoding="utf-8") as fh:
    for rel in sorted(paths):
        fh.write(f"{rel}\n")

print(f"Manifest entries: {len(paths)}")
PY

echo "Syncing release review assets from $REMOTE"
rsync -av --files-from="$tmp_manifest" "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ASSET_ROOT/"

echo
echo "Synced assets into: $LOCAL_ASSET_ROOT"
