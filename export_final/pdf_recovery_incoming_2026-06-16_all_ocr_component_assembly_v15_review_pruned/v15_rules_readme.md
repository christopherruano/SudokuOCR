# v15 Review-Pruned Assembly

No OCR/API calls are made. This pass refines v14 while preserving the adjacent-real-PDF-page invariant.

## Added Hard Breaks

- False row-label candidates where the current run already starts with a left-stub row-label page.
- False row-label candidates from narrative/context pages without right-half layout evidence and without a matching table title.

## Summary

- `input_v10_groups`: 7789
- `input_v14_edges`: 8856
- `input_v14_members`: 9109
- `v15_component_layer_counts`: {'single_physical_run': 7147, 'bounded_body_sequence': 426, 'same_table_refined_body_sequence': 4, 'bounded_variant_set': 11, 'continuation_refined_body_sequence': 44, 'required_physical_only': 16, 'same_scope_short_title_body_sequence': 4, 'classifier_core_numeric_form_sequence': 1, 'explicit_same_scope_body_sequence': 5, 'classifier_assisted_body_sequence': 1, 'classifier_same_part_tablevii_sequence': 1, 'classifier_split_title_body_sequence': 2}
- `v15_component_members`: 7789
- `v15_components`: 7662
- `v15_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'hard_break_demoted_related_context_noise': 653, 'hard_break_family_switch': 101, 'join_plain_same_table_high': 4, 'keep_missing_bridge': 80, 'hard_break_table_number_switch': 120, 'hard_break_scope_modifier_switch': 26, 'hard_break_scope_entity_switch': 80, 'join_continuation_same_table_high': 104, 'hard_break_explicit_scope_switch': 11, 'review_same_table_boundary_not_joined': 10, 'join_same_scope_short_title_high': 7, 'join_classifier_core_numeric_form_high': 2, 'hard_break_demoted_row_label_context_noise': 91, 'hard_break_pruned_false_row_label_candidate': 3, 'join_explicit_same_scope_same_table_high': 5, 'join_classifier_main_tablevii_high': 2, 'join_classifier_same_part_tablevii_continuation_high': 1, 'join_classifier_split_title_tablevii_high': 2}
- `v15_edge_rows`: 8856
- `v15_hard_break_pruned_false_row_label_edges`: 3
- `v15_invariant_failure_count`: 0
- `v15_invariant_failures`: []
- `v15_kept_missing_bridges`: 80
- `v15_remaining_required_row_label_review_edges`: 0
- `v15_remaining_same_table_review_edges`: 10
- `v15_review_queue_edges`: 90
