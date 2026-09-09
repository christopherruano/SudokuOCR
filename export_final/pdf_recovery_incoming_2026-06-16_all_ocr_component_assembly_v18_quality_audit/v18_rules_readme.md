# v18 Assembly Quality Audit

No OCR/API calls are made. This pass verifies v17 assembly and records the evidence needed before using it downstream.

## Gate Checks

- Physical joins must be adjacent real PDF pages.
- Multi-run physical components must not have internal page gaps.
- Logical collections must not have internal page gaps.
- Logical collections must not cross mixed-page boundaries.
- Known required physical joins from the row-label/right-half passthrough must be covered in v17 components.

## Summary

- `v18_invariant_failure_count`: 0
- `v18_invariant_status`: pass
- `v18_logical_variant_collection_edges`: 1
- `v18_missing_bridge_review_edges`: 80
- `v18_overassociation_audit_rows`: 294
- `v18_required_candidate_rows`: 150
- `v18_required_candidate_status_counts`: {'rejected_by_v17_hard_break': 80, 'covered_same_v17_component': 70}
- `v18_required_join_coverage_counts`: {'covered_same_v17_component': 82}
- `v18_required_join_edges`: 82
- `v18_unresolved_same_table_boundaries`: 5
