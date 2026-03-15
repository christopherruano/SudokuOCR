# HEB 91r - Historical Indian Census OCR Pipeline

## Purpose
Extract age-by-sex population tables from scanned pages of the Indian Census (1872–1941). Target: 100% cell-level accuracy on every table.

## Architecture
- **pipeline.py** – Main entry point. Defines test cases, API callers (OpenAI, Gemini, Claude), extraction strategies, and scoring against ground truth.
- **ensemble.py** – Voting and ensemble logic: majority_vote, cross_model_ensemble, digit_level_vote, enforce_constraints, digit_level_ensemble, constraint_ensemble (hierarchical CSP solver).
- **image_processing.py** – PIL/numpy-based row detection, cropping, and region extraction. No OpenCV dependency.
- **Data/** – Ground truth Excel files (one per province/state).
- **age_tables/** – Scanned PNG images organized by region and census year.
- **results/** – JSON output from pipeline runs.

## Test Cases
| # | Region | Image | GT Source | Notes |
|---|--------|-------|-----------|-------|
| 0 | Travancore Eastern 1901 | Clean, large digits | Travancore.xlsx | Gemini gets 100% on raw image |
| 1 | Hyderabad State 1901 | Smaller text, wide table | Hyderabad.xlsx | Crop left 33% for Population columns |
| 2 | Coorg 1901 | Multi-year table, small text | Coorg.xlsx | Crop left 65% for 1901 columns; GT has 0-1..4-5 (sum to 0-5) |

## Current Results
All 3 test cases achieve **100% cell-level accuracy** with the `moe` strategy.

## Key Lessons
- With good image quality (like Travancore), models CAN read every digit correctly.
- Preprocessing (crop + upscale + enhance) bridges the gap for harder images.
- Multi-enhancement variants × multi-model = many independent readings.
- Hierarchical CSP with intermediate subtotal locking resolves systematic digit confusions (3↔8, 5↔6) that voting alone cannot fix.
- Locking high-confidence subtotal readings (>50% agreement) as anchor points dramatically narrows the CSP search space, enabling unique constraint-based resolution.
- M+F=Persons constraint and group sum constraints cascade through the hierarchy.

## Ground Truth Corrections
See gt_corrections.md for known errors in the Excel ground truth files.

## Strategies
- `full` – Single full-image pass per model (baseline)
- `multipass` – 3 passes per model with majority voting
- `ensemble` – Multi-pass + cross-model ensemble + verify + constraints
- `moe` – Multi-Enhancement MoE: 4 image variants × 3 models = 12 readings, constraint-propagation ensemble with hierarchical CSP
