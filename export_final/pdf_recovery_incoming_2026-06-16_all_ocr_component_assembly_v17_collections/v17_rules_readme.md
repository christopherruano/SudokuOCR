# v17 Physical Components and Logical Collections

No OCR/API calls are made. This pass refines v16 and adds a separate collection layer.

## Physical Layer

- Physical joins remain strict: adjacent real PDF pages only, no internal component gaps.
- Added `join_current_right_half_same_scope_required_high` for right-half pages completing an immediately previous single-page same-scope table page.

## Collection Layer

- Logical collections can group consecutive sibling components, such as adjacent religion/subtable variants.
- Logical collections do not erase physical component boundaries.
- Mixed-page and missing-page boundaries are recorded but not grouped.

## Summary

- `input_v10_groups`: 7789
- `input_v16_edges`: 8856
- `input_v16_members`: 9109
- `v17_collection_action_counts`: {'hard_break_or_unrelated_boundary': 7319, 'review_missing_bridge_not_grouped': 80, 'review_mixed_page_boundary_not_grouped': 5, 'group_logical_variant_sequence_high': 1}
- `v17_collection_edges`: 7405
- `v17_collection_kind_counts`: {'physical_component': 7645, 'existing_bounded_variant_set': 11, 'bounded_logical_variant_set': 1}
- `v17_collection_members`: 7658
- `v17_collections`: 7657
- `v17_component_layer_counts`: {'single_physical_run': 7140, 'bounded_body_sequence': 426, 'same_table_refined_body_sequence': 4, 'bounded_variant_set': 11, 'continuation_refined_body_sequence': 43, 'required_physical_only': 16, 'same_scope_short_title_body_sequence': 4, 'classifier_core_numeric_form_sequence': 1, 'explicit_same_scope_body_sequence': 5, 'required_left_stub_right_half_sequence': 3, 'classifier_assisted_body_sequence': 1, 'classifier_same_part_tablevii_sequence': 1, 'classifier_split_title_body_sequence': 2, 'required_right_half_completion_sequence': 1}
- `v17_component_members`: 7789
- `v17_components`: 7658
- `v17_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'hard_break_demoted_related_context_noise': 653, 'hard_break_family_switch': 101, 'join_plain_same_table_high': 4, 'keep_missing_bridge': 80, 'hard_break_table_number_switch': 120, 'hard_break_scope_modifier_switch': 26, 'hard_break_scope_entity_switch': 80, 'join_continuation_same_table_high': 104, 'hard_break_explicit_scope_switch': 11, 'review_same_table_boundary_not_joined': 5, 'join_same_scope_short_title_high': 7, 'join_classifier_core_numeric_form_high': 2, 'hard_break_demoted_row_label_context_noise': 91, 'hard_break_pruned_false_row_label_candidate': 3, 'join_explicit_same_scope_same_table_high': 5, 'join_prev_left_stub_same_table_required_high': 3, 'keep_logical_variant_collection_boundary': 1, 'join_classifier_main_tablevii_high': 2, 'join_classifier_same_part_tablevii_continuation_high': 1, 'join_classifier_split_title_tablevii_high': 2, 'join_current_right_half_same_scope_required_high': 1}
- `v17_edge_rows`: 8856
- `v17_invariant_failure_count`: 0
- `v17_invariant_failures`: []
- `v17_kept_missing_bridges`: 80
- `v17_logical_variant_boundaries_kept_nonphysical`: 1
- `v17_logical_variant_collection_edges`: 1
- `v17_promoted_left_stub_joins`: 3
- `v17_promoted_right_half_joins`: 1
- `v17_remaining_same_table_review_edges`: 5
- `v17_review_queue_edges`: 85
