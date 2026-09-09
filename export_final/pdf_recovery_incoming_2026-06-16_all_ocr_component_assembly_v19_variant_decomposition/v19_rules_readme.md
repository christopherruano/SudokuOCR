# v19 Variant Decomposition

No OCR/API calls are made. This supplemental pass decomposes older bounded variant sets into logical collections with ordered member rows.

Use this layer when the downstream consumer needs the distinction between strict physical assembly and sibling table grouping.

## Summary

- `v19_collection_source_counts`: {'inherited_v10_bounded_variant_set': 12, 'v17_logical_variant_collection': 1}
- `v19_edge_source_counts`: {'inherited_v10_bounded_variant_set': 84, 'v17_logical_variant_collection': 1}
- `v19_inherited_review_flag_counts`: {'no_ocr_structural_profile_for_pair': 12, 'uncertain_without_ocr_profile': 12, 'low_quality_ocr': 7, 'no_ocr_structural_profile_for_sequence_edge': 12, 'collection_no_ocr_structural_profile': 12, 'religion_variant_same_table_scope': 12, 'large_bounded_sequence_review': 3, 'zero_constraint_checks': 2, 'zero_rows_extracted': 2, 'ocr_profile_read_error': 2, 'layout_geometry_rejected_subtable_switch': 1, 'possible_unrelated_subtable': 2, 'layout_geometry_rejected_non_core_topic': 3, 'incompatible_page_class_or_table_subject': 5, 'weak_strict_title_same_scope_review': 2, 'weak_title_sequence_review': 2, 'missing_intermediate_pages': 2, 'part_entity_incomplete': 2, 'duplicate_records_for_same_physical_page': 2, 'same_pdf_page_multiple_titles': 2, 'same_pdf_page_conflicting_page_kind': 2, 'same_pdf_page_conflicting_table_family': 2, 'image_layout_required_join': 2, 'layout_join_review': 2, 'v8_relaxed_logical_join_review': 2, 'same_table_number_relaxed_title_threshold': 2}
- `v19_invariant_failure_count`: 0
- `v19_invariant_status`: pass
- `v19_join_type_counts`: {'religion_variant_subtable_sequence': 20, 'same_title_contiguous_sequence': 54, 'same_scope_contiguous_sequence': 2, 'same_published_table_contiguous_sequence_v8': 8, 'bounded_logical_variant_set': 1}
- `v19_v18_invariant_status_seen`: pass
- `v19_variant_collections`: 13
- `v19_variant_edges`: 85
- `v19_variant_members`: 98
