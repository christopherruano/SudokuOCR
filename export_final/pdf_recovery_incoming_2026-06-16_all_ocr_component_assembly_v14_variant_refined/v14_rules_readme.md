# v14 Variant-Refined Assembly

No OCR/API calls are made. This pass refines v13 while preserving the adjacent-real-PDF-page invariant.

## Added Joins

- Classifier-confirmed adjacent Table VII same-part numeric continuations.
- Adjacent same-table edges with an explicit matching scope marker such as State Total, Towns, Details by Districts, or the same civil-status/religion/part marker.

## Added Hard Breaks

- Special-status switches, including OCR misspellings such as DIVORVED.
- Different explicit province/district/state/division/taluk labels when both sides expose one.
- Statement-number and subsidiary-subtable marker switches.

## Summary

- `input_v10_groups`: 7789
- `input_v13_edges`: 8856
- `input_v13_members`: 9109
- `v14_component_layer_counts`: {'single_physical_run': 7147, 'bounded_body_sequence': 426, 'same_table_refined_body_sequence': 4, 'bounded_variant_set': 11, 'continuation_refined_body_sequence': 44, 'required_physical_only': 16, 'same_scope_short_title_body_sequence': 4, 'classifier_core_numeric_form_sequence': 1, 'explicit_same_scope_body_sequence': 5, 'classifier_assisted_body_sequence': 1, 'classifier_same_part_tablevii_sequence': 1, 'classifier_split_title_body_sequence': 2}
- `v14_component_members`: 7789
- `v14_components`: 7662
- `v14_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'hard_break_demoted_related_context_noise': 653, 'hard_break_family_switch': 101, 'join_plain_same_table_high': 4, 'keep_missing_bridge': 80, 'hard_break_table_number_switch': 120, 'hard_break_scope_modifier_switch': 26, 'hard_break_scope_entity_switch': 80, 'join_continuation_same_table_high': 104, 'hard_break_explicit_scope_switch': 11, 'review_same_table_boundary_not_joined': 10, 'join_same_scope_short_title_high': 7, 'join_classifier_core_numeric_form_high': 2, 'hard_break_demoted_row_label_context_noise': 91, 'review_required_row_label_join_tablelike': 3, 'join_explicit_same_scope_same_table_high': 5, 'join_classifier_main_tablevii_high': 2, 'join_classifier_same_part_tablevii_continuation_high': 1, 'join_classifier_split_title_tablevii_high': 2}
- `v14_edge_rows`: 8856
- `v14_hard_break_explicit_scope_switch_edges`: 11
- `v14_invariant_failure_count`: 0
- `v14_invariant_failures`: []
- `v14_join_classifier_same_part_tablevii_edges`: 1
- `v14_join_explicit_same_scope_edges`: 5
- `v14_kept_missing_bridges`: 80
- `v14_remaining_classifier_low_similarity_edges`: 0
- `v14_remaining_required_row_label_review_edges`: 3
- `v14_remaining_same_table_review_edges`: 10
- `v14_review_queue_edges`: 93
