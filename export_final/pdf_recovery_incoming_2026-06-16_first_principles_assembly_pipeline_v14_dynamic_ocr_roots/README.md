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
- `adjacency_decision_index.csv`: one normalized decision row for every consecutive source edge plus missing-page bridge candidates; this is the compact no-missed-groupings coverage table.
- `adjacency_decision_summary.csv`: counts for the adjacency decision index.
- `pairwise_edges.csv`: explicit edge-level reasons for join/break decisions.
- `row_assembly_attempts.csv`: row-label inheritance and value-check evidence.
- `artifact_resolution_report.csv`: exact-match resolver report showing whether missing row-assembly OCR artifacts were found locally by source/page or image hash.
- `missing_artifact_targets.csv`: unresolved OCR artifact targets by component/page/image, with expected source-page and image-hash filename patterns for incoming OCR drops.
- `candidate_row_assembly_attempts.csv`: conservative row-label-dependency candidate tests outside the seed physical set; passing rows still require manual review before promotion.
- `candidate_test_pass_review_queue.csv`: candidate tests that structurally passed and should be reviewed, not auto-merged.
- `candidate_artifact_resolution_report.csv`: resolver report for candidate-test OCR artifacts.
- `physical_grouping_decisions.csv`: grouping-first decision layer for every seed physical pair; grouping safety is separated from numeric value cleanliness.
- `confirmed_physical_groupings.csv`: consecutive two-page physical groupings with structural row-label inheritance proof. Some still require value review before numeric publication.
- `partial_physical_groupings.csv`: bounded partial page pairs where at least one block aligns, but the full pair must not be merged without review.
- `logical_grouping_decisions.csv`: grouping-first decision layer for logical collections; confirmed rows are bounded, consecutive, collection-only, and never physical merges.
- `confirmed_logical_collections.csv`: use-now logical sibling/child/self-contained collections suitable for grouping on the site without physical row merging.
- `logical_collection_members.csv`: page-level ordered membership for every confirmed logical collection, including PDF page, title/image evidence, and explicit no-physical-merge flags.
- `column_order_contract.csv`: per physical assembly block column-order checks.
- `assembled_row_previews.csv`: flattened left-page then right-page row preview.
- `layout_suspect_physical_join_candidates.csv` / `potential_physical_join_candidates.csv`: non-promotional adjacent row-label-dependency candidates whose stored layout flags resemble a physical split; these are layout-suspect review leads, not confirmed physical assemblies.
- `row_label_dependency_watchlist.csv`: broader adjacent row-label-deficiency watchlist for classifier/profile misses; these are not physical candidates unless promoted by stronger evidence.
- `refined_sibling_groups.csv`: bounded logical sibling collections recovered after v28.
- `assembly_workbook.xlsx`: spreadsheet version of the same evidence.

## Current Counts

```json
{
  "adjacency_decision_index_rows": 7275,
  "adjacency_decisions_by_layer": {
    "blocked_physical_missing_ocr_artifact": 59,
    "confirmed_physical_pair": 12,
    "failed_physical_constraints_review": 9,
    "hard_break_or_not_a_grouping_candidate": 5398,
    "logical_collection_only_confirmed": 138,
    "logical_collection_only_review": 1430,
    "missing_intermediate_pages_candidate": 110,
    "partial_physical_pair_review": 3,
    "row_label_dependency_review_candidate": 116
  },
  "adjacency_decisions_by_scope": {
    "consecutive_pdf_pages": 7165,
    "missing_page_bridge": 110
  },
  "adjacency_decisions_safe_to_logical_group": 150,
  "adjacency_decisions_safe_to_physical_merge": 12,
  "artifact_resolution_by_method": {
    "existing": 44
  },
  "artifact_resolution_by_status": {
    "existing_artifact": 44,
    "unresolved_no_match": 116
  },
  "artifact_resolution_report_rows": 160,
  "assembled_row_preview_rows": 513,
  "assembly_manifest_rows": 846,
  "blocked_groups": 261,
  "candidate_artifact_resolution_by_method": {
    "existing": 134
  },
  "candidate_artifact_resolution_by_status": {
    "existing_artifact": 134,
    "unresolved_no_match": 98
  },
  "candidate_artifact_resolution_report_rows": 232,
  "candidate_row_assembly_attempt_rows": 116,
  "candidate_row_assembly_attempts_by_status": {
    "blocked_missing_ocr_artifact": 98,
    "review_failed_constraints": 15,
    "review_no_safe_block_alignment": 3
  },
  "candidate_row_assembly_test_pass_review_rows": 0,
  "column_order_contract_by_status": {
    "pass": 27
  },
  "column_order_contract_rows": 27,
  "confirmed_logical_collection_rows": 80,
  "confirmed_physical_grouping_rows": 12,
  "confirmed_physical_groupings_safe_to_publish_values": 1,
  "confirmed_physical_groupings_value_review_required": 11,
  "logical_child_segments": 29,
  "logical_collection_groups": 564,
  "logical_collection_member_rows": 219,
  "logical_collection_members_by_confidence": {
    "high_bounded_child_variant_collection": 69,
    "high_bounded_sibling_collection": 138,
    "high_self_contained_sequence_collection": 12
  },
  "logical_collection_members_by_image_path_exists": {
    "": 3,
    "yes": 216
  },
  "logical_collection_members_by_source": {
    "logical_child_segment_image_list": 69,
    "page_only_no_image_evidence": 3,
    "refined_sibling_edge_page_node": 113,
    "source_stage_logical_image_list": 34
  },
  "logical_collection_members_with_images": 216,
  "logical_grouping_decision_rows": 564,
  "logical_grouping_decisions_by_confidence": {
    "high_bounded_child_variant_collection": 29,
    "high_bounded_sibling_collection": 45,
    "high_self_contained_sequence_collection": 6,
    "review_large_or_parent_context": 28,
    "review_required": 456
  },
  "logical_grouping_decisions_by_status": {
    "confirmed_bounded_logical_collection": 80,
    "review_large_logical_collection": 28,
    "review_same_table_logical_candidate": 456
  },
  "logical_grouping_review_rows": 484,
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
  "missing_artifact_target_rows": 116,
  "missing_artifact_targets_by_priority": {
    "blocks_pairwise_physical_grouping": 116
  },
  "missing_artifact_targets_by_source_bucket": {
    "final_tableVII_render": 96,
    "legacy_never_ocrd_image": 12,
    "legacy_referenced_digitized_table": 8
  },
  "missing_artifact_targets_with_existing_images": 116,
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

## When More OCR Artifacts Arrive

The resolver looks for existing OCR artifacts in these roots:

- all local `export_final/gemini_no_gap_ocr_2026-06-16*` directories, including rerun batches from the OCR window
- `export/money_gapfill_ocr`
- `results`
- `export_final/sendout_final/sendout_v1/mistral_md`

Expected filenames are listed per page in `missing_artifact_targets.csv` as
`expected_source_page_globs` and `expected_image_hash_globs`. After new OCR
artifacts land, rerun:

```bash
python3 build_first_principles_assembly_pipeline.py --out-dir export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_pipeline_next
```
