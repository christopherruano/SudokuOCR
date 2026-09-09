# v12 Continuation-Refined Assembly

No OCR/API calls are made. This pass refines v11 edge recommendations and rebuilds components.

## Added Rule

- Join adjacent pages when they are same table family, same normalized table key, table-like, no scope/entity switch, no derivative/context/offtarget signal, title similarity >= 0.74, and one side has a continuation/concluded token.

## Demotions

- Row-label and related-edge review candidates involving context/blank/offtarget/derivative pages are hard breaks.
- Part/sex/parenthetical entity switches and explicit table-number switches are hard breaks, not manual review, unless a future rule explicitly handles them.

## Summary

- `input_v10_groups`: 7789
- `input_v11_edges`: 8856
- `input_v11_members`: 9109
- `v12_component_layer_counts`: {'single_physical_run': 7175, 'bounded_body_sequence': 431, 'bounded_variant_set': 11, 'continuation_refined_body_sequence': 47, 'required_physical_only': 18, 'classifier_assisted_body_sequence': 1}
- `v12_component_members`: 7789
- `v12_components`: 7683
- `v12_demoted_related_noise_edges`: 853
- `v12_demoted_row_label_noise_edges`: 91
- `v12_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'hard_break_demoted_related_context_noise': 653, 'review_related_not_grouped': 101, 'review_same_table_boundary_not_joined': 63, 'keep_missing_bridge': 80, 'hard_break_table_number_switch': 120, 'hard_break_scope_entity_switch': 80, 'join_continuation_same_table_high': 104, 'review_classifier_main_tablevii_boundary': 2, 'hard_break_demoted_row_label_context_noise': 91, 'review_required_row_label_join_tablelike': 3, 'join_classifier_main_tablevii_high': 2, 'review_classifier_main_tablevii_low_similarity': 3}
- `v12_edge_rows`: 8856
- `v12_invariant_failure_count`: 0
- `v12_invariant_failures`: []
- `v12_join_classifier_main_edges`: 2
- `v12_join_continuation_same_table_edges`: 104
- `v12_kept_existing_joins`: 1320
- `v12_kept_missing_bridges`: 80
- `v12_remaining_related_review_edges`: 101
- `v12_remaining_required_row_label_review_edges`: 3
- `v12_remaining_same_table_review_edges`: 63
- `v12_review_queue_edges`: 252
