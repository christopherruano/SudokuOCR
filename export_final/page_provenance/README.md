# Page Provenance And Completeness

This folder is the corpus-level dictionary for source-image provenance.

Core rule: each rendered PNG hash is recoverable as `sha1(source_pdf_original + ':' + pdf_page_number)`.

Files:

- `page_provenance_dictionary.csv`: one row per unique local PNG hash, with all local paths preserved.
- `ordered_image_sequence.csv`: every image ordered within its source PDF or unresolved doc hash.
- `table_vii_image_sequence.csv`: likely Table VII / age-sex-civil-condition images in source order.
- `table_vii_accessibility.csv`: source-level checklist for whether Table VII candidates are accessible and ordered.
- `table_vii_missing_pages.csv`: missing expected pages for likely Table VII/age-sex-civil-condition ranges.
- `table_range_completeness.csv`: one row per TOC/table range with observed vs missing PDF pages.
- `source_pdf_completeness.csv`: one row per source PDF.
- `pdf_page_inventory.csv`: one row per expected PDF page inside resolved table ranges.
- `missing_pdf_pages.csv`: exact PDF pages that are expected by a resolved range but absent locally.
- `page_provenance_dictionary.json`: image-hash keyed lookup dictionary.
- `page_provenance_workbook.xlsx`: human-browsable version of the same outputs.

Interpretation:

- `provenance_confidence=exact` means the image hash matched a source PDF and PDF page.
- Missing pages are a first-pass worklist from TOC/table ranges. Some may be non-table pages if a range is broad.
- For final OCR targeting, prioritize rows in `missing_pdf_pages.csv` whose neighboring observed pages are relevant Table VII/age tables.

Summary:

- `local_image_paths_indexed`: 31544
- `unique_local_image_hashes_indexed`: 12721
- `manifest_image_hashes`: 8451
- `catalog_image_hashes`: 3603
- `table_ranges_indexed`: 1110
- `page_rows`: 12721
- `exact_page_rows`: 9550
- `unresolved_page_rows`: 3171
- `local_ocr_json_present`: 9874
- `no_local_ocr_json`: 2847
- `table_ranges`: 3850
- `table_ranges_with_missing_pages`: 1993
- `source_pdfs`: 194
- `missing_pdf_pages`: 3570
- `unresolved_source_pdf_doc_hashes`: 79
- `ordered_image_sequence_rows`: 12721
- `table_vii_candidate_images`: 3753
- `table_vii_source_groups`: 177
- `table_vii_missing_expected_pages`: 468
- `provenance_methods`: {'hash_match_bruteforce': 3480, 'hash_match_in_resolved_range': 6070, 'no_pdf_or_hash': 3152, 'hash_no_match': 19}
