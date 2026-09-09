# First-Principles Assembly Pipeline

This directory is the stable assembly output. The versioned v25-v29 scripts are
implementation stages; downstream work should read these stable files first.

## Process

1. Sort pages within each source by real `pdf_page`.
2. Decide pairwise adjacency edges only between consecutive pages.
3. Allow physical assembly only for strict pairwise, consecutive pages that need
   row-label/half-spread completion.
4. Verify row-label inheritance and cross-page value constraints from OCR
   artifacts where available.
5. Build logical collection groups separately from physical merges. These can
   group bounded sibling tables such as Muslim/Christian/Jew subtables, but they
   never imply physical row assembly.
6. Block anything crossing missing pages or missing OCR artifacts; review
   title-only/same-table candidates rather than automatically merging them.

## Main Files

- `assembly_manifest.csv`: every final group with stable columns.
- `use_now_groups.csv`: groups safe enough for downstream assembly/static-site use.
- `review_groups.csv`: plausible groups requiring human or stronger automated review.
- `blocked_groups.csv`: groups blocked by missing pages or missing OCR artifacts.
- `assembly_review_queue.csv`: prioritized residual work with blockers/actions.
- `evidence_gap_summary.csv`: counts by blocker and queue bucket.
- `logical_child_segments.csv`: small actionable children split from large logical parents.
- `pairwise_edges.csv`: explicit edge-level reasons for join/break decisions.
- `row_assembly_attempts.csv`: row-label inheritance and value-check evidence.
- `physical_grouping_decisions.csv`: grouping-first decision layer for every seed physical pair; grouping safety is separated from numeric value cleanliness.
- `confirmed_physical_groupings.csv`: consecutive two-page physical groupings with structural row-label inheritance proof. Some still require value review before numeric publication.
- `partial_physical_groupings.csv`: bounded partial page pairs where at least one block aligns, but the full pair must not be merged without review.
- `column_order_contract.csv`: per physical assembly block column-order checks.
- `assembled_row_previews.csv`: flattened left-page then right-page row preview.
- `layout_suspect_physical_join_candidates.csv` / `potential_physical_join_candidates.csv`: non-promotional adjacent row-label-dependency candidates whose stored layout flags resemble a physical split; these are layout-suspect review leads, not confirmed physical assemblies.
- `row_label_dependency_watchlist.csv`: broader adjacent row-label-deficiency watchlist for classifier/profile misses; these are not physical candidates unless promoted by stronger evidence.
- `refined_sibling_groups.csv`: bounded logical sibling collections recovered after v28.
- `assembly_workbook.xlsx`: spreadsheet version of the same evidence.

## Current Counts

```json
{
  "assembled_row_preview_rows": 513,
  "assembly_manifest_rows": 846,
  "blocked_groups": 261,
  "column_order_contract_by_status": {
    "pass": 27
  },
  "column_order_contract_rows": 27,
  "confirmed_physical_grouping_rows": 12,
  "confirmed_physical_groupings_safe_to_publish_values": 1,
  "confirmed_physical_groupings_value_review_required": 11,
  "logical_child_segments": 29,
  "logical_collection_groups": 564,
  "manifest_by_decision": {
    "blocked": 261,
    "review": 504,
    "use_now": 81
  },
  "manifest_by_tier": {
    "blocked_missing_ocr_artifact": 59,
    "blocked_missing_pages": 202,
    "manual_review": 6,
    "needs_value_review": 11,
    "partial_review": 3,
    "review_large_collection": 17,
    "review_large_logical_parent": 11,
    "review_only": 456,
    "use_now": 81
  },
  "partial_physical_grouping_rows": 3,
  "physical_grouping_decision_rows": 80,
  "physical_grouping_decisions_by_confidence": {
    "blocked": 59,
    "high_structural_pass_value_pass": 1,
    "high_structural_pass_value_review": 11,
    "medium_partial_structural_pass": 3,
    "review_required": 6
  },
  "physical_grouping_decisions_by_status": {
    "blocked_missing_ocr_artifact": 59,
    "confirmed_pairwise_physical_grouping": 12,
    "partial_pairwise_physical_grouping_review": 3,
    "review_failed_constraints": 5,
    "review_no_safe_block_alignment": 1
  },
  "potential_physical_candidate_rows": 7,
  "potential_physical_candidates_by_blocker": {
    "missing_ocr_artifact_for_row_assembly": 6,
    "missing_schema_profile": 1
  },
  "potential_physical_candidates_by_evidence_status": {
    "stored_layout_signal_only_not_confirmed": 7
  },
  "potential_physical_candidates_by_tier": {
    "high_core_table_vii_row_label_pair": 1,
    "low_structured_row_label_pair": 2,
    "medium_age_civil_row_label_pair": 4
  },
  "potential_physical_candidates_by_type": {
    "layout_suspect_physical_pair_review_only": 7
  },
  "potential_physical_core_table_vii_candidates": 1,
  "required_physical_groups": 80,
  "review_groups": 504,
  "review_queue_by_bucket": {
    "large_logical_collection_review": 17,
    "large_logical_parent_context": 11,
    "missing_pdf_pages": 202,
    "physical_failed_constraints": 6,
    "physical_missing_ocr_artifact": 59,
    "physical_partial_review": 3,
    "physical_value_review": 11,
    "same_table_logical_review": 39,
    "small_same_table_logical_review": 417
  },
  "review_queue_by_priority": {
    "1": 11,
    "2": 3,
    "3": 6,
    "4": 59,
    "5": 202,
    "6": 11,
    "7": 17,
    "8": 417,
    "9": 39
  },
  "review_queue_rows": 765,
  "row_label_dependency_watchlist_by_evidence_status": {
    "not_a_physical_layout_candidate": 109,
    "stored_layout_signal_only_not_confirmed": 7
  },
  "row_label_dependency_watchlist_by_tier": {
    "high_core_table_vii_row_label_pair": 58,
    "low_structured_row_label_pair": 6,
    "medium_age_civil_row_label_pair": 45,
    "medium_core_table_vii_needs_title_review": 7
  },
  "row_label_dependency_watchlist_by_type": {
    "classifier_repair_row_label_pair_review": 10,
    "layout_suspect_physical_pair_review_only": 7,
    "strict_same_kind_family_watchlist": 99
  },
  "row_label_dependency_watchlist_core_table_vii": 65,
  "row_label_dependency_watchlist_rows": 116,
  "use_now_by_subgroup": {
    "logical_child_segment": 29,
    "logical_refined_sibling_bounded": 38,
    "logical_self_contained_sequence": 6,
    "logical_sibling_bounded": 7,
    "physical_required_row_assembled": 1
  },
  "use_now_groups": 81
}
```
