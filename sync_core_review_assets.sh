#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-/n/home11/cruano/census_toc}"
DECISIONS_CSV="${3:-export/local_research_bundle/final_core_prep/core_review_decisions.csv}"
LOCAL_ASSET_ROOT="${4:-export/local_research_bundle/final_core_prep/review_assets}"

mkdir -p "$LOCAL_ASSET_ROOT"
tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

python3 - "$DECISIONS_CSV" "$REMOTE_ROOT" "$tmp_manifest" <<'PY'
import csv
import sys
from pathlib import Path

decisions_csv = Path(sys.argv[1]).expanduser()
remote_root = Path(sys.argv[2]).expanduser()
manifest_path = Path(sys.argv[3])

rows = list(csv.DictReader(decisions_csv.open(encoding="utf-8", newline="")))
paths = set()

for row in rows:
    for key in ("image_path", "result_json"):
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

echo "Syncing review assets from $REMOTE"
rsync -av --files-from="$tmp_manifest" "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ASSET_ROOT/"

echo
echo "Synced assets into: $LOCAL_ASSET_ROOT"
