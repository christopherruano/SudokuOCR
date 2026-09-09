# PNG Helpfulness Triage Summary

This triage decides which newly rendered Table VII candidate PNGs should be OCRed first, OCRed later, visually screened, or skipped. It combines existing provenance overlap, external render confidence/status, lightweight image QA, and sampled vision-agent review of representative contact sheets.

## Counts

- total_pngs: 4829
- final_action_tier_1: 732
- final_action_tier_2: 902
- final_action_tier_3: 1486
- final_action_tier_4: 10
- final_action_tier_5: 301
- final_action_tier_9: 1398
- decision::vision_screen_before_ocr_generous_broad: 1463
- decision::skip_duplicate_exact_page: 1398
- decision::ocr_next_relocated_discovery: 620
- decision::ocr_now_confirmed_discovery: 395
- decision::ocr_next_generous_near_known_pages: 282
- decision::defer_low_confidence_approx_blind: 280
- decision::ocr_now_exact_table_vii_gap: 164
- decision::ocr_now_existing_gap: 149
- decision::ocr_now_sampled_vision_target: 22
- decision::defer_sampled_non_target: 21
- decision::visual_check_sparse_or_blank_render: 20
- decision::context_only_do_not_data_ocr_first: 10
- decision::visual_check_before_ocr_borderline: 2
- decision::ocr_now_vision_promoted_low_confidence: 2
- decision::visual_check_rotation_then_ocr_if_table: 1
- vision_sampled_pages: 120
- image_qa_likely_blank_separator_or_failed_render: 24

## Recommended Use

- OCR `final_png_ocr_now.csv` first.
- OCR `final_png_ocr_next_if_budget.csv` if first-pass OCR still leaves gaps or budget allows.
- Do not OCR `final_png_visual_screen_before_ocr.csv` wholesale; review/contact-sheet/vision-screen it first.
- Keep `final_png_context_only.csv` for provenance/table-boundary context but do not prioritize data OCR.
- Do not OCR `final_png_defer_or_skip.csv` first; it is mostly duplicates, low-confidence pages, or sampled non-target pages.
