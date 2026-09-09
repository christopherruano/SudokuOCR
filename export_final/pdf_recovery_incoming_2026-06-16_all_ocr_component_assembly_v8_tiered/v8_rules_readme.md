# v8 Tiered Assembly Rules

This output is assembly logic only. It performs no OCR/API calls.

## Layers

- `required_physical`: inherited from v6 physical runs. These are adjacent pages that must be assembled to interpret the table, usually because a right-side page lacks row labels or image layout shows a half-spread.
- `bounded_same_table_sequence`: adjacent physical runs that appear to be the same published table or continuation. These are logical groupings, not proof that every page physically depends on the prior page.
- `bounded_variant_set`: adjacent same-scope subtables, especially religion variants, where the title scope is compatible and no part/sex/parenthetical-number conflict is detected.
- `missing_bridge_candidate`: related pages separated by missing PDF pages. These are never joined; the missing page range is recorded.
- `candidate_related_not_grouped`: adjacent pages with evidence of relation that remains too risky to group automatically, usually because of classifier/page-kind/family switches or weaker similarity.
- `break`: no safe assembly relation found.

## Hard Invariants

- Sequence joins require the same source key and adjacent real PDF pages (`page_gap == 1`).
- Missing intermediate PDF pages are bridge candidates, not joins.
- Non-increasing page order is always a break.
- Explicit table-number switches are always breaks.
- Part, sex, and parenthetical-number conflicts are breaks unless the entity is missing on one side and flagged for review.

## Relaxed v8 Joins

- v8 adds reviewed logical joins below the v7 title threshold only when the table number/source/subject evidence is stable.
- These joins carry `v8_relaxed_logical_join_review` and should be treated as reviewable logical sequence edges.
- The relaxed path still cannot cross a missing page; it becomes a bridge candidate instead.

## Summary

- `break_sequence_edges`: 7541
- `bridge_candidate_edges`: 75
- `candidate_related_not_grouped_edges`: 1217
- `core_table_vii_bridge_candidates`: 22
- `core_table_vii_multi_run_sequence_groups`: 220
- `core_table_vii_sequence_edges`: 728
- `core_table_vii_sequence_groups`: 1105
- `core_table_vii_sequence_members`: 1811
- `input_v6_physical_runs`: 9109
- `input_v6_required_physical_edges`: 82
- `input_v6_run_members`: 9191
- `joined_sequence_edges`: 1240
- `large_sequence_review_groups`: 14
- `max_sequence_physical_pages`: 37
- `max_sequence_physical_runs`: 37
- `multi_run_sequence_groups`: 460
- `old_v6_gap_crossing_audit_rows`: 92
- `old_v6_max_collection_runs`: 39
- `old_v6_multi_run_collection_groups`: 582
- `old_v6_overassociation_audit_rows`: 294
- `old_v6_split_by_v8_audit_rows`: 287
- `religion_variant_sequence_groups`: 12
- `sequence_groups`: 7869
- `sequence_invariant_failure_count`: 0
- `sequence_invariant_failures`: []
- `sequence_review_queue_rows`: 7981
- `v8_relaxed_bridge_candidate_edges`: 3
- `v8_relaxed_join_sequence_edges`: 74
