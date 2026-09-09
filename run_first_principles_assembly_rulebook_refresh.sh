#!/usr/bin/env bash
set -euo pipefail

# Rebuild the first-principles assembly frontier/control/rulebook layers from a
# supplied assembly output. This performs no OCR/API calls.
#
# Environment overrides:
#   ASSEMBLY_DIR      Existing first-principles assembly output to consume.
#   CLASSIFIER_CSV    Optional page-classifier CSV to join as break/review signal.
#   EXTRA_RENDER_ROOT Local render/artifact root for intake matching, defaults to ~/Downloads.
#
# Usage:
#   ./run_first_principles_assembly_rulebook_refresh.sh
#   ./run_first_principles_assembly_rulebook_refresh.sh 20260617_after_new_ocr

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
ASSEMBLY_DIR="${ASSEMBLY_DIR:-export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_v29_logical_review_context}"
CLASSIFIER_CSV="${CLASSIFIER_CSV:-export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined/gemini_ocr_worklists/gemini_classify_priority_all.csv}"
EXTRA_RENDER_ROOT="${EXTRA_RENDER_ROOT:-$HOME/Downloads}"

INTAKE_DIR="export_final/pdf_recovery_incoming_2026-06-16_assembly_ocr_intake_${TAG}"
FRONTIER_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_${TAG}"
COMPONENT_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_components_${TAG}"
LOGICAL_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_family_frontier_${TAG}"
CONTROL_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_control_${TAG}"
RULEBOOK_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_rulebook_${TAG}"
SEQUENCE_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_source_sequence_map_${TAG}"
QUALITY_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_quality_audit_${TAG}"
RECOMMENDATIONS_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_recommendations_${TAG}"
CONTRACT_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_contract_${TAG}"
EVIDENCE_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_evidence_${TAG}"
LOGICAL_EVIDENCE_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_collection_evidence_${TAG}"
GROUPING_MANIFEST_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_static_grouping_manifest_${TAG}"
CANDIDATE_PROMOTION_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_candidate_promotion_${TAG}"
ASSEMBLY_FRONTIER_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_frontier_${TAG}"
MANUAL_ADJUDICATION_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_manual_adjudication_${TAG}"
LOGICAL_SIBLING_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_logical_sibling_sets_${TAG}"
MATERIALIZED_PHYSICAL_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_materialized_physical_assembly_${TAG}"
RELEASE_SNAPSHOT_DIR="export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_${TAG}"

echo "assembly_dir=$ASSEMBLY_DIR"
echo "tag=$TAG"

python3 build_assembly_ocr_intake_report.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --extra-root "$EXTRA_RENDER_ROOT" \
  --out-dir "$INTAKE_DIR"

python3 build_first_principles_assembly_frontier_v30.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --intake-dir "$INTAKE_DIR" \
  --out-dir "$FRONTIER_DIR"

python3 build_first_principles_assembly_components_v31.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --frontier-dir "$FRONTIER_DIR" \
  --out-dir "$COMPONENT_DIR"

python3 build_first_principles_logical_family_frontier_v32.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --out-dir "$LOGICAL_DIR"

python3 build_first_principles_assembly_control_v33.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --frontier-dir "$FRONTIER_DIR" \
  --component-dir "$COMPONENT_DIR" \
  --logical-dir "$LOGICAL_DIR" \
  --intake-dir "$INTAKE_DIR" \
  --out-dir "$CONTROL_DIR"

python3 build_first_principles_assembly_rulebook_v34.py \
  --assembly-dir "$ASSEMBLY_DIR" \
  --frontier-dir "$FRONTIER_DIR" \
  --component-dir "$COMPONENT_DIR" \
  --logical-dir "$LOGICAL_DIR" \
  --control-dir "$CONTROL_DIR" \
  --classifier-csv "$CLASSIFIER_CSV" \
  --out-dir "$RULEBOOK_DIR"

python3 build_first_principles_source_sequence_map_v35.py \
  --rulebook-dir "$RULEBOOK_DIR" \
  --out-dir "$SEQUENCE_DIR"

python3 build_first_principles_assembly_quality_audit_v36.py \
  --rulebook-dir "$RULEBOOK_DIR" \
  --sequence-dir "$SEQUENCE_DIR" \
  --logical-dir "$LOGICAL_DIR" \
  --out-dir "$QUALITY_DIR"

python3 build_first_principles_assembly_recommendations_v37.py \
  --rulebook-dir "$RULEBOOK_DIR" \
  --sequence-dir "$SEQUENCE_DIR" \
  --quality-dir "$QUALITY_DIR" \
  --out-dir "$RECOMMENDATIONS_DIR"

python3 build_first_principles_assembly_contract_v38.py \
  --rulebook-dir "$RULEBOOK_DIR" \
  --sequence-dir "$SEQUENCE_DIR" \
  --quality-dir "$QUALITY_DIR" \
  --recommendations-dir "$RECOMMENDATIONS_DIR" \
  --out-dir "$CONTRACT_DIR"

python3 build_first_principles_assembly_evidence_v39.py \
  --rulebook-dir "$RULEBOOK_DIR" \
  --contract-dir "$CONTRACT_DIR" \
  --out-dir "$EVIDENCE_DIR" \
  --extra-root "$EXTRA_RENDER_ROOT"

python3 build_first_principles_logical_collection_evidence_v40.py \
  --logical-dir "$LOGICAL_DIR" \
  --quality-dir "$QUALITY_DIR" \
  --rulebook-dir "$RULEBOOK_DIR" \
  --contract-dir "$CONTRACT_DIR" \
  --out-dir "$LOGICAL_EVIDENCE_DIR"

python3 build_first_principles_static_grouping_manifest_v41.py \
  --contract-dir "$CONTRACT_DIR" \
  --physical-evidence-dir "$EVIDENCE_DIR" \
  --logical-evidence-dir "$LOGICAL_EVIDENCE_DIR" \
  --out-dir "$GROUPING_MANIFEST_DIR"

python3 build_first_principles_candidate_promotion_v42.py \
  --grouping-dir "$GROUPING_MANIFEST_DIR" \
  --physical-evidence-dir "$EVIDENCE_DIR" \
  --out-dir "$CANDIDATE_PROMOTION_DIR"

python3 build_first_principles_assembly_frontier_v43.py \
  --physical-evidence-dir "$EVIDENCE_DIR" \
  --candidate-promotion-dir "$CANDIDATE_PROMOTION_DIR" \
  --grouping-dir "$GROUPING_MANIFEST_DIR" \
  --out-dir "$ASSEMBLY_FRONTIER_DIR"

python3 build_first_principles_manual_adjudication_v44.py \
  --frontier-dir "$ASSEMBLY_FRONTIER_DIR" \
  --out-dir "$MANUAL_ADJUDICATION_DIR"

python3 build_first_principles_logical_sibling_sets_v45.py \
  --logical-evidence-dir "$LOGICAL_EVIDENCE_DIR" \
  --out-dir "$LOGICAL_SIBLING_DIR"

python3 build_first_principles_materialized_physical_assembly_v46.py \
  --frontier-dir "$ASSEMBLY_FRONTIER_DIR" \
  --candidate-promotion-dir "$CANDIDATE_PROMOTION_DIR" \
  --control-dir "$CONTROL_DIR" \
  --out-dir "$MATERIALIZED_PHYSICAL_DIR"

python3 build_first_principles_assembly_release_snapshot_v47.py \
  --contract-dir "$CONTRACT_DIR" \
  --intake-dir "$INTAKE_DIR" \
  --rulebook-dir "$RULEBOOK_DIR" \
  --frontier-dir "$ASSEMBLY_FRONTIER_DIR" \
  --quality-dir "$QUALITY_DIR" \
  --manual-adjudication-dir "$MANUAL_ADJUDICATION_DIR" \
  --logical-sibling-dir "$LOGICAL_SIBLING_DIR" \
  --materialized-physical-dir "$MATERIALIZED_PHYSICAL_DIR" \
  --out-dir "$RELEASE_SNAPSHOT_DIR"

python3 build_assembly_grouping_record.py \
  --release-dir "$RELEASE_SNAPSHOT_DIR"

python3 build_assembly_manual_decision_overlay.py \
  --release-dir "$RELEASE_SNAPSHOT_DIR"

python3 build_assembly_review_page.py \
  --release-dir "$RELEASE_SNAPSHOT_DIR"

python3 build_ai_assembly_grouping_ledger.py \
  --release-dir "$RELEASE_SNAPSHOT_DIR"

echo "rulebook_dir=$RULEBOOK_DIR"
echo "sequence_dir=$SEQUENCE_DIR"
echo "quality_dir=$QUALITY_DIR"
echo "recommendations_dir=$RECOMMENDATIONS_DIR"
echo "contract_dir=$CONTRACT_DIR"
echo "evidence_dir=$EVIDENCE_DIR"
echo "logical_evidence_dir=$LOGICAL_EVIDENCE_DIR"
echo "grouping_manifest_dir=$GROUPING_MANIFEST_DIR"
echo "candidate_promotion_dir=$CANDIDATE_PROMOTION_DIR"
echo "assembly_frontier_dir=$ASSEMBLY_FRONTIER_DIR"
echo "manual_adjudication_dir=$MANUAL_ADJUDICATION_DIR"
echo "logical_sibling_dir=$LOGICAL_SIBLING_DIR"
echo "materialized_physical_dir=$MATERIALIZED_PHYSICAL_DIR"
echo "release_snapshot_dir=$RELEASE_SNAPSHOT_DIR"
echo "ai_assembly_grouping_ledger_dir=$RELEASE_SNAPSHOT_DIR/ai_assembly_grouping_ledger"
