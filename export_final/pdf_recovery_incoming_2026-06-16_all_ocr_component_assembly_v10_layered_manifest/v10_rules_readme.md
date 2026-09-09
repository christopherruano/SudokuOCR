# v10 Layered Assembly Manifest

No OCR/API calls are made. This pass reads v6 physical assembly outputs and v9 logical sequence outputs.

## Layers

- `required_physical_row_label_assembly`: adjacent pages that must be combined to read a table, usually a right half-page lacking row labels.
- `bounded_body_sequence`: adjacent physical runs that form the same published table body.
- `bounded_variant_set`: adjacent same-scope subtables, e.g. Muslim/Christian/Jew pages for the same table scope.
- `missing_bridge_candidate`: related pages separated by missing PDF pages; never joined automatically.
- `possible_required_row_label_assembly`: review-only candidates where a break may still need row labels from the previous page.

## Static Site Guidance

- Use `v10_static_site_action` on groups and `v10_static_site_edge_action` on edges.
- `review_before_static_merge` means the group is consecutive but too uncertain or broad to merge silently.
- Missing bridge candidates should be shown as gaps, not assembled components.

## Summary

- `bounded_body_sequence_groups`: 467
- `bounded_variant_set_groups`: 12
- `bridge_edges_not_joined`: 80
- `candidate_related_not_grouped_edges`: 1161
- `groups_with_front_matter_or_context`: 3172
- `groups_with_required_physical_runs`: 54
- `input_v9_detailed_edges`: 8856
- `input_v9_groups`: 7789
- `input_v9_members`: 9109
- `logical_join_edges`: 1320
- `missing_bridge_candidates`: 80
- `old_overassociation_audit_rows`: 294
- `possible_required_edge_review_candidates`: 150
- `required_physical_edges`: 82
- `required_physical_only_groups`: 25
- `review_before_static_merge_groups`: 143
- `static_group_candidates`: 336
- `v10_edges`: 8856
- `v10_groups`: 7789
- `v10_invariant_failure_count`: 0
- `v10_invariant_failures`: []
- `v10_members`: 9109
- `variant_sets`: 12
