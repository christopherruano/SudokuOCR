# v23 Candidate Collections

No OCR/API calls are made. v23 groups the v22 immediate review edges into bounded consecutive candidate collections.

## Semantics

- `physical_dependency_review_collection`: possible required page assembly, usually left-stub/right-half evidence.
- `logical_sibling_subtable_review_collection`: likely sibling subtables such as adjacent religion/part variants.
- `physical_or_logical_continuation_review_collection`: continuation evidence exists, but review is required to decide whether this is physical assembly or logical grouping.
- `same_core_table_review_collection`: same core table evidence exists but no safe rule supports automatic grouping.

Every collection has `can_auto_apply=no`. This layer is a review/scalability package, not a merge pass.

## Summary

- `v23_audit_failure_count`: 0
- `v23_audit_status`: pass
- `v23_auto_apply_collections`: 0
- `v23_candidate_collection_edges`: 72
- `v23_candidate_collection_members`: 112
- `v23_candidate_collections`: 40
- `v23_collection_type_counts`: {'physical_dependency_review_collection': 7, 'logical_sibling_subtable_review_collection': 25, 'physical_or_logical_continuation_review_collection': 6, 'logical_continuation_review_collection': 2}
- `v23_edge_candidate_class_counts`: {'review_possible_left_stub_continuation': 13, 'review_continued_same_table': 57, 'review_high_similarity_same_core_table': 2}
- `v23_input_immediate_edges`: 72
