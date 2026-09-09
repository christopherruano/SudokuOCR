# v13 Scope-Refined Assembly

No OCR/API calls are made. This pass refines v12 using scope modifiers and classifier-confirmed split/core numeric fragments.

## Added Joins

- Classifier-confirmed split-title Table VII fragments.
- Classifier-confirmed Form VIA/core civil-condition fragments.
- Plain same-table high-similarity main Table VII boundaries.
- Same-scope short-title boundaries without geo/detail/special-status scope changes.

## Added Hard Breaks

- Remaining related-not-grouped table-family switches.
- Same-table boundaries where only one side has a scope modifier such as fly-sheet, summary, geo detail, special civil-status, or caste/race marker.

## Summary

- `input_v10_groups`: 7789
- `input_v12_edges`: 8856
- `input_v12_members`: 9109
- `v13_component_layer_counts`: {'single_physical_run': 7157, 'bounded_body_sequence': 428, 'same_table_refined_body_sequence': 4, 'bounded_variant_set': 11, 'continuation_refined_body_sequence': 44, 'required_physical_only': 16, 'same_scope_short_title_body_sequence': 4, 'classifier_core_numeric_form_sequence': 1, 'classifier_assisted_body_sequence': 1, 'classifier_split_title_body_sequence': 2}
- `v13_component_members`: 7789
- `v13_components`: 7668
- `v13_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'hard_break_demoted_related_context_noise': 653, 'hard_break_family_switch': 101, 'join_plain_same_table_high': 4, 'keep_missing_bridge': 80, 'hard_break_table_number_switch': 120, 'hard_break_scope_modifier_switch': 26, 'hard_break_scope_entity_switch': 80, 'join_continuation_same_table_high': 104, 'review_same_table_boundary_not_joined': 26, 'join_same_scope_short_title_high': 7, 'join_classifier_core_numeric_form_high': 2, 'hard_break_demoted_row_label_context_noise': 91, 'review_required_row_label_join_tablelike': 3, 'join_classifier_main_tablevii_high': 2, 'review_classifier_main_tablevii_low_similarity': 1, 'join_classifier_split_title_tablevii_high': 2}
- `v13_edge_rows`: 8856
- `v13_hard_break_family_switch_edges`: 101
- `v13_hard_break_scope_modifier_switch_edges`: 26
- `v13_invariant_failure_count`: 0
- `v13_invariant_failures`: []
- `v13_join_classifier_core_numeric_form_edges`: 2
- `v13_join_classifier_main_edges`: 2
- `v13_join_classifier_split_title_edges`: 2
- `v13_join_continuation_same_table_edges`: 104
- `v13_join_plain_same_table_edges`: 4
- `v13_join_same_scope_short_title_edges`: 7
- `v13_kept_missing_bridges`: 80
- `v13_remaining_related_review_edges`: 0
- `v13_remaining_required_row_label_review_edges`: 3
- `v13_remaining_same_table_review_edges`: 26
- `v13_review_queue_edges`: 110
