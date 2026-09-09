# v9 Split-Title Assembly Rules

This output is assembly logic only. It performs no OCR/API calls.

## Input Layer

- Reads v6 strict physical runs through v7/v8 helpers.
- Physical runs remain the only required row-label/half-spread assembly layer.
- Logical sequence groups can only join adjacent physical runs in the same source.

## Added v9 Rule

- Adds `split_title_core_table_contiguous_sequence_v9` for adjacent Table VII title fragments.
- Typical pattern: `TABLE VII - AGE, SEX` followed by `AND CIVIL CONDITION`.
- The rule excludes context/blank, off-target, unknown, and derivative selected-caste/race/tribe pages.
- Explicit non-VII table numbers, part/sex/parenthetical conflicts, religion conflicts, and missing pages prevent a direct join.
- Every v9 join carries `v9_split_title_classifier_switch_review`.

## Review Outputs

- `v9_sequence_edges_detailed.csv`: every boundary with join/break/bridge reason and row-label/layout context.
- `v9_candidate_related_edges.csv`: adjacent related-looking edges deliberately not grouped.
- `v9_large_group_audit.csv`: large consecutive groups to inspect for over-broad logical grouping.

## Summary

- `break_sequence_edges`: 7456
- `bridge_candidate_edges`: 80
- `candidate_related_not_grouped_edges`: 1161
- `core_table_vii_bridge_candidates`: 25
- `core_table_vii_multi_run_sequence_groups`: 238
- `core_table_vii_sequence_edges`: 808
- `core_table_vii_sequence_groups`: 1025
- `core_table_vii_sequence_members`: 1810
- `input_v6_physical_runs`: 9109
- `input_v6_required_physical_edges`: 82
- `input_v6_run_members`: 9191
- `joined_sequence_edges`: 1320
- `large_core_table_vii_group_audit_rows`: 13
- `large_group_audit_rows`: 16
- `large_sequence_review_groups`: 16
- `max_sequence_physical_pages`: 37
- `max_sequence_physical_runs`: 37
- `multi_run_sequence_groups`: 479
- `old_v6_gap_crossing_audit_rows`: 92
- `old_v6_max_collection_runs`: 39
- `old_v6_multi_run_collection_groups`: 582
- `old_v6_overassociation_audit_rows`: 294
- `old_v6_split_by_v9_audit_rows`: 287
- `religion_variant_sequence_groups`: 12
- `sequence_groups`: 7789
- `sequence_invariant_failure_count`: 0
- `sequence_invariant_failures`: []
- `sequence_review_queue_rows`: 7991
- `v8_relaxed_join_sequence_edges`: 74
- `v9_split_title_bridge_candidate_edges`: 5
- `v9_split_title_join_sequence_edges`: 80
