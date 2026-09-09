#!/usr/bin/env bash
set -uo pipefail
# Sync ALL never-OCR'd page PNGs from FASRC to local, using the image_path
# column of never_ocrd_queues/all.csv as the file list.
#
# Usage:
#   ./sync_never_ocrd_images.sh [REMOTE] [REMOTE_ROOT] [LOCAL_ROOT] [QUEUE_CSV]
# Defaults:
#   REMOTE      = cruano@login.rc.fas.harvard.edu
#   REMOTE_ROOT = /n/home11/cruano/census_toc
#   LOCAL_ROOT  = ./export/never_ocrd_images
#   QUEUE_CSV   = ./export/cluster_sync/never_ocrd_queues/all.csv
#
# NOTE: this does NOT use `set -e`, so a transient rsync error won't silently
# abort with a partial copy looking "done" — instead we re-check at the end and
# report exactly how many files are still missing. Re-run until missing == 0
# (rsync resumes; already-present files are skipped).

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-/n/home11/cruano/census_toc}"
LOCAL_ROOT="${3:-./export/never_ocrd_images}"
QUEUE_CSV="${4:-./export/cluster_sync/never_ocrd_queues/all.csv}"

if [ ! -f "$QUEUE_CSV" ]; then
  echo "ERROR: queue CSV not found: $QUEUE_CSV" >&2
  exit 1
fi

mkdir -p "$LOCAL_ROOT"
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

# Build the relative-path manifest from the CSV's image_path column.
python3 - "$QUEUE_CSV" "$REMOTE_ROOT" "$manifest" <<'PY'
import csv, sys
csv.field_size_limit(10_000_000)
queue_csv, remote_root, manifest_path = sys.argv[1], sys.argv[2], sys.argv[3]
if not remote_root.endswith("/"):
    remote_root += "/"
rels = set()
with open(queue_csv, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        img = (r.get("image_path") or "").strip()
        if not img:
            continue
        if img.startswith(remote_root):
            rels.add(img[len(remote_root):])
        else:
            print(f"  WARN: image_path not under remote root: {img}", file=sys.stderr)
with open(manifest_path, "w") as f:
    for p in sorted(rels):
        f.write(p + "\n")
print(f"Manifest: {len(rels)} unique never-OCR'd images")
PY

n_manifest=$(wc -l < "$manifest" | tr -d ' ')
echo "Syncing $n_manifest images from $REMOTE:$REMOTE_ROOT -> $LOCAL_ROOT"
echo

rsync -av --partial --progress \
  --files-from="$manifest" \
  "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"
rc=$?
echo
echo "rsync exit code: $rc  (0 = clean; 23/24 = some files missing remotely/vanished)"

# ---- completeness verification (the part that answers 'did it skip anything?') ----
echo "Verifying local copy against manifest ..."
python3 - "$manifest" "$LOCAL_ROOT" <<'PY'
import os, sys
manifest_path, local_root = sys.argv[1], sys.argv[2]
missing = []
total = 0
with open(manifest_path) as fh:
    for line in fh:
        rel = line.strip()
        if not rel:
            continue
        total += 1
        if not os.path.exists(os.path.join(local_root, rel)):
            missing.append(rel)
print(f"  manifest: {total}  present locally: {total - len(missing)}  MISSING: {len(missing)}")
if missing:
    out = os.path.join(local_root, "_still_missing.txt")
    with open(out, "w") as f:
        f.write("\n".join(missing) + "\n")
    print(f"  -> wrote {len(missing)} still-missing paths to {out} (re-run this script to fetch them)")
else:
    print("  ✅ COMPLETE: every manifest image is present locally.")
PY
