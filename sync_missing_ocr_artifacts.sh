#!/usr/bin/env bash
set -u

# Pull expected OCR artifacts for local images that currently have no matching
# local *_oneshot.json. This is intentionally targeted: it does not sync the
# whole OCR tree.
#
# Usage:
#   ./sync_missing_ocr_artifacts.sh [REMOTE] [REMOTE_ROOT] [LOCAL_OCR_ROOT] [FILES_FROM]
#
# Defaults:
#   REMOTE         = cruano@login.rc.fas.harvard.edu
#   REMOTE_ROOT    = /n/home11/cruano/census_toc
#   LOCAL_OCR_ROOT = ./export/ocr_results_referenced
#   FILES_FROM     = ./export_final/page_provenance/no_local_ocr_json_cluster_files.txt
#
# For the narrower current Table VII candidate subset:
#   ./sync_missing_ocr_artifacts.sh cruano@login.rc.fas.harvard.edu /n/home11/cruano/census_toc \
#     ./export/ocr_results_referenced \
#     ./export_final/page_provenance/no_local_ocr_json_table_vii_cluster_files.txt

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-/n/home11/cruano/census_toc}"
LOCAL_OCR_ROOT="${3:-./export/ocr_results_referenced}"
FILES_FROM="${4:-./export_final/page_provenance/no_local_ocr_json_cluster_files.txt}"

if [[ ! -f "$FILES_FROM" ]]; then
  echo "Missing files-from list: $FILES_FROM" >&2
  exit 2
fi

mkdir -p "$LOCAL_OCR_ROOT"

echo "Remote:       $REMOTE"
echo "Remote OCR:   ${REMOTE_ROOT}/ocr_results/"
echo "Local OCR:    $LOCAL_OCR_ROOT"
echo "Files-from:   $FILES_FROM"
echo "File entries: $(wc -l < "$FILES_FROM" | tr -d ' ')"
echo

rsync -av --files-from="$FILES_FROM" \
  "${REMOTE}:${REMOTE_ROOT}/ocr_results/" \
  "${LOCAL_OCR_ROOT}/"

status=$?
if [[ "$status" -eq 23 ]]; then
  echo
  echo "rsync exited with 23: some listed artifacts were not present on the cluster."
  echo "That is expected if those pages truly have not been OCRed yet."
  exit 0
fi
exit "$status"
