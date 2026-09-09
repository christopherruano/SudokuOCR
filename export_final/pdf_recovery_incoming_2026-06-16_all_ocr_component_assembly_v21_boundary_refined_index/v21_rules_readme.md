# v21 Boundary-Refined Master Assembly Index

No OCR/API calls are made. This package is the canonical index over current assembly artifacts plus explicit boundary decisions.

## Layers

- `physical_component`: strict v17 physical assembly groups.
- `v17_*`: collection/display groups over physical components.
- `v19_logical_variant_collection`: decomposed sibling religion/subtable collections.
- `missing_bridge_candidate`: likely assemblable after missing pages are available.
- `section_split_candidate`: adjacent pages where only part of a page should logically group with its neighbor; do not whole-page merge.
- `boundary_hard_break_confirmed`: visually confirmed table-scope switch; do not merge.
- `unresolved_same_table_boundary`: adjacent mixed/page-slicing boundaries that are not safe whole-page joins and still need classification.

## Boundary Rule

A boundary may be a logical continuation without being a safe whole-page join. v21 keeps those as `section_split_candidate` records until a future page-splitting layer can assign page regions to separate logical groups.

## Summary

- `v21_audit_failure_count`: 0
- `v21_audit_status`: pass
- `v21_boundary_decision_counts`: {'missing_pages_required_before_join': 80, 'requires_section_split': 4, 'confirmed_hard_break': 1}
- `v21_boundary_decisions`: 85
- `v21_confirmed_hard_break_boundary_groups`: 1
- `v21_edge_type_counts`: {'hard_break': 7319, 'physical_join': 1451, 'missing_bridge_candidate': 80, 'section_split_candidate': 4, 'logical_variant_boundary_nonphysical': 1, 'hard_break_confirmed_by_visual_boundary_review': 1, 'collection_boundary': 7405, 'variant_member_boundary': 85}
- `v21_edges`: 16346
- `v21_group_type_counts`: {'physical_component': 7658, 'v17_physical_component': 7645, 'v17_existing_bounded_variant_set': 11, 'v17_bounded_logical_variant_set': 1, 'v19_logical_variant_collection': 13, 'missing_bridge_candidate': 80, 'section_split_candidate': 4, 'boundary_hard_break_confirmed': 1}
- `v21_groups`: 15413
- `v21_members`: 15715
- `v21_missing_bridge_groups`: 80
- `v21_page_index_rows`: 8981
- `v21_physical_components_indexed`: 7658
- `v21_required_join_rows_carried`: 82
- `v21_section_split_candidate_groups`: 4
- `v21_unresolved_boundary_groups`: 0
- `v21_v17_collections_indexed`: 7657
- `v21_v19_variant_collections_indexed`: 13
