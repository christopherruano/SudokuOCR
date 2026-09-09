# v22 Candidate Assembly Audit

No OCR/API calls are made. This layer audits remaining non-joined boundaries after v21.

## Rules

- Existing v17/v21 joins and v21 boundary decisions are treated as already handled.
- Non-adjacent rows are never joins. They can only become `possible_missing_bridge_extra_review`.
- Derivative selected-caste/race/tribe tables, explicit table switches, and obvious scope/status switches are rejected.
- Adjacent core-table rows with left-stub, continuation-title, or high-similarity evidence become review candidates, not automatic joins.
- `can_auto_join_now` is deliberately `no` for every v22 candidate.

## Summary

- `v22_audit_failure_count`: 0
- `v22_audit_status`: pass
- `v22_auto_join_candidates`: 0
- `v22_candidate_class_counts`: {'review_possible_left_stub_continuation': 13, 'possible_missing_bridge_extra_review': 121, 'review_continued_same_table': 57, 'review_high_similarity_same_core_table': 2, 'low_confidence_same_core_table': 59}
- `v22_candidate_edges`: 252
- `v22_input_edges`: 8856
- `v22_no_row_label_current_candidates`: 0
- `v22_rejected_or_handled_edges`: 8604
- `v22_rejection_class_counts_top20`: {'reject_low_signal': 3801, 'already_joined': 1451, 'not_adjacent_no_group': 1391, 'reject_derivative_selected': 541, 'reject_explicit_switch': 338, 'offtarget_or_other_same_table_high_similarity': 332, 'missing_page_number_or_nonadjacent': 172, 'already_boundary_decided': 85, 'reject_obvious_switch_status_scope_switch_none_to_married': 69, 'reject_obvious_switch_status_scope_switch_married_to_none': 53, 'reject_obvious_switch_status_scope_switch_widowed_to_none': 33, 'reject_obvious_switch_status_scope_switch_none_to_unmarried': 32, 'reject_obvious_switch_status_scope_switch_none_to_widowed': 31, 'reject_obvious_switch_status_scope_switch_unmarried_to_none': 29, 'reject_obvious_switch_selected_caste_race_tribe_derivative_blocks_assembly': 20, 'reject_obvious_switch_status_scope_switch_sample_to_none': 18, 'reject_obvious_switch_status_scope_switch_none_to_sample': 16, 'reject_obvious_switch_geo_scope_switch_table vii age sex and civil condition part ': 13, 'reject_obvious_switch_geo_scope_switch_sagaing district_to_lower chindwin district': 8, 'reject_obvious_switch_geo_scope_switch_table viii education b district_to_educatio': 8}
- `v22_sources_with_candidates`: 62
- `v22_total_edges_classified`: 8856
- `v22_v21_boundary_decisions_carried`: 85
