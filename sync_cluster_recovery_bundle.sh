#!/usr/bin/env bash
set -euo pipefail

# Sync the current best recovery/export artifacts from FASRC to the local machine.
#
# Usage:
#   ./sync_cluster_recovery_bundle.sh [REMOTE] [REMOTE_ROOT] [LOCAL_ROOT]
#
# Defaults:
#   REMOTE      = cruano@login.rc.fas.harvard.edu
#   REMOTE_ROOT = ~/census_toc
#   LOCAL_ROOT  = ./export/cluster_sync

REMOTE="${1:-cruano@login.rc.fas.harvard.edu}"
REMOTE_ROOT="${2:-~/census_toc}"
LOCAL_ROOT="${3:-./export/cluster_sync}"

mkdir -p "$LOCAL_ROOT"

sync_dir() {
  local name="$1"
  echo "Syncing directory: $name"
  rsync -av "${REMOTE}:${REMOTE_ROOT}/${name}/" "${LOCAL_ROOT}/${name}/"
}

sync_dir_optional() {
  local name="$1"
  echo "Syncing directory (optional): $name"
  if rsync -av "${REMOTE}:${REMOTE_ROOT}/${name}/" "${LOCAL_ROOT}/${name}/" 2>/dev/null; then
    :
  fi
}

sync_file() {
  local name="$1"
  echo "Syncing file: $name"
  rsync -av "${REMOTE}:${REMOTE_ROOT}/${name}" "${LOCAL_ROOT}/"
}

sync_dir "export_inference_rich"
sync_dir "export_inference_plus_core_rescue_v5"
sync_dir_optional "export_inference_final_core_curated_v1"
sync_dir_optional "export_inference_tail_rescue_unreviewed_v1"
sync_dir_optional "export_assembled_final_core_plus_tail_strong_v1_facts"
sync_dir "non_age_rescue_v5"
sync_dir "never_ocrd_queues"
sync_dir "recovery_status"
sync_dir "age_recall_audit"
sync_dir_optional "assembled_final_core_curated_v1"
sync_dir_optional "assembled_final_core_curated_v1_improved"
sync_dir_optional "assembled_final_core_plus_tail_v1"
sync_dir_optional "assembled_final_core_plus_tail_v1_improved"

sync_file "organized_manifest.jsonl"
if rsync -av "${REMOTE}:${REMOTE_ROOT}/data_guide.xlsx" "${LOCAL_ROOT}/" 2>/dev/null; then
  :
fi
if rsync -av "${REMOTE}:${REMOTE_ROOT}/gap_fillability_guide.xlsx" "${LOCAL_ROOT}/" 2>/dev/null; then
  :
fi

echo
echo "Sync complete."
echo "Local root: ${LOCAL_ROOT}"
