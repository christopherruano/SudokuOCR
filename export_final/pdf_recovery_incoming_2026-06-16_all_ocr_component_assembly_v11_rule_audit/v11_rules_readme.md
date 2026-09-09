# v11 Rule Audit

This pass makes no OCR/API calls. It overlays Gemini page classes from the existing classifier CSV onto v10 components.

## Join Policy

- Existing v10 joins are preserved.
- Missing PDF-page gaps remain bridge candidates and are not joined.
- Classifier-assisted joins require adjacent PDF pages, both pages classified `main_tablevii_numeric`, matching normalized table key, no scope/entity conflict, and strong title/scope similarity.
- Row-label, split-title, low-similarity main Table VII, and related-but-not-grouped edges are exported for review rather than merged.

## Summary

- `classifier_member_matches`: 267
- `classifier_rows`: 267
- `input_v10_edges`: 8856
- `input_v10_groups`: 7789
- `input_v10_members`: 9109
- `v11_classifier_high_join_edges`: 2
- `v11_classifier_review_join_edges`: 5
- `v11_component_layer_counts`: {'single_physical_run': 7284, 'bounded_body_sequence': 465, 'bounded_variant_set': 12, 'required_physical_only': 25, 'classifier_assisted_body_sequence': 1}
- `v11_component_members`: 7789
- `v11_components`: 7787
- `v11_edge_action_counts`: {'hard_break_or_missing_page': 1684, 'hard_break': 4550, 'keep_existing_join': 1320, 'review_related_not_grouped': 1121, 'keep_missing_bridge': 80, 'review_classifier_main_tablevii_boundary': 2, 'review_required_row_label_join': 94, 'join_classifier_main_tablevii_high': 2, 'review_classifier_main_tablevii_low_similarity': 3}
- `v11_edge_rows`: 8856
- `v11_hard_breaks`: 6234
- `v11_invariant_failure_count`: 0
- `v11_invariant_failures`: []
- `v11_kept_existing_joins`: 1320
- `v11_kept_missing_bridges`: 80
- `v11_required_row_label_review_edges`: 94
- `v11_review_queue_edges`: 1300
