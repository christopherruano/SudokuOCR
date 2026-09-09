# Sendout Final Goal

Goal: take the existing SudokuOCR export package and turn it into a final send-out-ready local release with only confirmed, non-redundant, plausibility-checked data; browsable per-table Excel files; explicit confidence/review/exclusion metadata; a clean mortality subcollection; and robust format-aware plausibility checks for tables whose structure differs from the main age/civil-condition pattern.

## Workspace

- Repo root: `/Users/chrisruano/Research/SudokuOCR`
- Current package root: `/Users/chrisruano/Research/SudokuOCR/export_final`
- Current team-ready layer: `/Users/chrisruano/Research/SudokuOCR/export_final/team_ready`
- Local referenced source images:
  - `/Users/chrisruano/Research/SudokuOCR/export/digitized_tables_referenced/digitized_tables/`
- Main datasets:
  - `export_final/datasets/core_curated_v2_final`
  - `export_final/datasets/tail_exploratory_v2`
  - `export_final/datasets/assembled_strong_v2`
- Review/browse material:
  - `export_final/browse/final_submission_review`
  - `export_final/browse/by_topic/mortality`
  - `export_final/browse/failing_tables`
  - `export_final/team_ready/final_clean`

Assume all needed source images and OCR JSONs may already be available locally, especially under:

- `export/digitized_tables_referenced/digitized_tables/`
- relevant `review_assets` folders

## Primary Objective

Create a final release folder at:

- `export_final/sendout_final`

This folder should be clean enough to send to collaborators immediately.

## Constraints

- Be careful with disk and memory. Stream large CSVs. Do not load multi-GB fact files fully into memory.
- Do not duplicate huge datasets unless necessary. Prefer hardlinks or symlinks for large source files.
- Do not delete existing data unless explicitly creating/replacing generated output under `export_final/sendout_final`.
- If a table is suspicious, do not silently include it in the confirmed dataset. Flag it, inspect source image/OCR evidence where available, and either exclude it or put it in a separate review-needed folder.
- The final confirmed fact list must contain no redundant table-level metadata repeated across every fact row. Use a clean fact table plus separate metadata tables.
- The final output should favor being defensible over being large. If in doubt, exclude or send to `review_needed`.

## Required Final Structure

- `export_final/sendout_final/README.md`
- `export_final/sendout_final/DATA_DICTIONARY.md`
- `export_final/sendout_final/RELEASE_NOTES.md`
- `export_final/sendout_final/catalogs/`
- `export_final/sendout_final/data/`
- `export_final/sendout_final/excel_tables/`
- `export_final/sendout_final/mortality/`
- `export_final/sendout_final/review/`
- `export_final/sendout_final/excluded/`
- `export_final/sendout_final/qa/`
- `export_final/sendout_final/showcase/`

## Final Data Model

### `data/facts_confirmed.csv`

One row per confirmed numeric cell only. Keep this lean. Include only fact-level analytical fields:

- `fact_id` if useful
- `table_id`
- `section`
- `row_label_raw`
- `row_label_std`
- `row_dimension`
- `row_group_value`
- `column_group_raw`
- `column_group_std`
- `column_dimension`
- `column_group_value`
- `sub_column_raw`
- `sub_column_std`
- `sex`
- `civil_condition`
- `religion`
- `resolved_age_axis`
- `resolved_age_label_std`
- `resolved_age_lower`
- `resolved_age_upper`
- `resolved_age_open_ended`
- `resolved_age_total`
- `resolved_age_not_stated`
- `measure_unit`
- `value`

### `data/tables_confirmed.csv`

One row per confirmed table. Include table-level context:

- `table_id`
- `layer`
- `province`
- `year`
- `title`
- `table_family`
- `measure_category`
- `constraint_regime`
- `denominator_hint`
- `effective_quality_status`
- `quality_tier`
- `confidence_level`
- `n_checks`
- `failed_checks`
- `qc_flags`
- `source_image_local` if available
- `source_json_local` if available
- `source_image_original`
- `source_json_original`
- `facts_rows`
- `excel_path`

### `data/table_members.csv`

For assembled or multipage tables only. Include:

- `table_id`
- `member_index`
- `member_source_image`
- `member_source_json`
- `member_title`
- `member_page_status` if available

### `data/exclusions.csv`

One row per excluded table with:

- `table_id`
- `layer`
- `province`
- `year`
- `title`
- `reason_category`
- `reason_detail`
- `source_image_local` if available
- `source_json_local` if available

### `data/review_needed.csv`

One row per table that is not confirmed but might be salvageable later.

## Inclusion Policy For `facts_confirmed`

Start from the current strict zero-failure clean set:

- `export_final/team_ready/catalogs/final_clean_tables.csv`

Include only tables with:

- `in_release_scope = 1`
- `quality_tier = strict_validated`
- `failed_checks = 0`
- not flagged for unresolved manual review

Do not include assembled tables automatically. They must pass additional semantic checks and visual/source inspection before inclusion.

Any table with repeated impossible-looking columns, copied totals down age rows, invalid axis interpretation, or implausible zero/rate structure should be excluded or sent to `review_needed`.

## Format-Aware Plausibility Checks

The project includes multiple table formats. Do not apply one constraint regime blindly. Infer the table's likely format from:

- `table_family`
- `title`
- `measure_unit`
- `constraint_regime`
- `denominator_hint`
- `row_dimension` / `column_dimension`
- section names
- column group labels
- source image when needed

For every table, assign a `format_class`, such as:

- `additive_age_sex_counts`
- `age_civil_condition_counts`
- `age_by_religion_counts`
- `distribution_of_1000`
- `distribution_of_10000`
- `rate_or_ratio_table`
- `mortality_counts`
- `mortality_rates`
- `disease_specific_mortality`
- `geography_or_district_counts`
- `education_literacy_table`
- `occupation_caste_tribe_race_table`
- `malformed_or_ambiguous`

Then run plausibility checks appropriate to that format.

## General Checks For All Formats

- values parse as numeric where expected
- no negative values unless explicitly meaningful
- no duplicate conflicting fact coordinates
- table has nonzero useful numeric content
- row/column axes are not obviously collapsed, copied, or swapped
- no one cell family is copied identically down many unrelated row labels unless the original image clearly shows a repeated total
- no columns that should vary by row are constant across age rows or disease rows
- no table has massive male/female imbalance unless the title/context explains it
- no table has `0` males and tens of thousands of females, or vice versa, unless it is explicitly a female-only or male-only table
- `Persons` / `All` should not be less than either Male or Female when sex decomposition exists
- `Persons` / `All` should generally equal Male + Female for count tables
- if the title says female, male, women, men, or of each sex, account for that before flagging sex imbalance

## Additive Count Checks

- Persons/All = Male + Female when those columns exist
- totals/subtotals equal sums where table structure supports it
- no constant copied value repeated down many age rows for marital/civil-condition columns unless the source image proves it is a table total rather than an age-specific cell
- child age bins should not have huge widowed/married totals copied from whole-table totals
- flag any age 0-5 married/widowed values above a small threshold
- small nonzero child married/widowed values may be historically possible, but very large numbers are not credible without image confirmation
- automatic red flags:
  - `0-1` widowed persons = 8,000+
  - `0-5` widowed persons in the thousands
  - `0-1` married persons in the thousands
  - same widowed/unmarried/married total repeated for every age row
  - `Population` constant down all age rows in an age table
  - `Unmarried` constant down all age rows in an age table

## Age/Civil-Condition Checks

- Population = Unmarried + Married + Widowed where the table structure supports it
- Within each civil condition, Persons = Male + Female where available
- For young ages, married/widowed should be small relative to population and must not look like repeated column totals
- For older ages, widowed should generally become more plausible and often rise; a flat repeated widowed value across all ages is suspicious
- If child marriage produces real nonzero values, they should still be numerically plausible and visually confirmed if large

## Age/Sex Population Checks

- Persons = Male + Female
- age totals should align with age-bin sums where totals are present
- no age row should have impossible sex ratios unless context explains it
- if a table is female-only or male-only, ensure the table title/source supports that and avoid forcing Persons = Male + Female

## Age/Religion Or Age/Race/Caste Checks

- Check within each religion/race/caste group separately
- Avoid using cross-group totals unless the table clearly provides them
- Watch for header-level groups being mistaken as row values
- Flag if the same religion/race/caste total is copied down age rows

## Distribution Checks

Distribution-of-1000 and distribution-of-10000 tables need their own logic.

- distribution-of-1000 tables should sum to 1000 within the correct slice
- distribution-of-10000 tables should sum to 10000 within the correct slice
- identify the correct slice before checking: often sex + religion + age or section + year + age
- do not treat rates as distributions just because `1000` appears in the title
- values in a distribution table should generally be in range `0..denominator`
- rows with all zeros are allowed only if the source table clearly represents absent categories
- flag if the same 1000/10000 distribution is repeated across many unrelated rows

## Rate Or Ratio Tables

- Do not force rates to sum to totals.
- Check range plausibility instead.
- Per-mille rates should usually be nonnegative and not absurdly high unless the denominator is very small or the table explains it.
- Repeated 0/1/100 values across many unrelated rows are suspicious.
- If the table is a sex ratio, e.g. females per 1,000 males, values around hundreds or low thousands may be plausible.
- If the table is a death rate per mille, values should be plausible for mortality context and should not be copied totals.

## Mortality Checks

Mortality must be a first-class subcollection.

Separate mortality into:

- `death_counts`
- `death_rates`
- `disease_specific`
- `other_mortality`

Death-count tables:

- Persons = Male + Female if sex decomposition exists
- disease totals should not be copied down unrelated disease rows
- year rows and disease rows must not be mixed as if they are the same axis

Death-rate tables:

- do not sum rates like counts
- rates should be nonnegative
- repeated exact 0, 1, or 100 values across many disease/year rows should be flagged
- if all rates are zero or one, inspect the source image

Disease-specific mortality:

- disease labels should be row categories
- years/geographies should not be mistaken for disease rows
- if the table title says "from certain diseases," disease-specific rows should vary

Exclude or review mortality tables where:

- year rows and disease rows are mixed incorrectly
- same total deaths are copied down disease rows
- ratio per mille values are repeated mechanically
- row labels like Fever/Cholera/Plague inherit the same total-death values from a year row

## Geography Or District Tables

- These may be valid, but only include if they are part of the intended scope.
- Do not apply age-bin plausibility checks unless age is actually an axis.
- Check for district rows being mistaken as ages.
- If the project's final scope is age-structured tables, move pure geography/district tables to `review_needed` or `excluded` unless explicitly intended.

## Education/Literacy, Occupation, Caste/Tribe/Race, Infirmity

- These may appear in the OCR/tail material.
- Include only if they clearly support the project's intended analysis.
- Do not let these in solely because they contain age or sex labels.
- If off-topic, put in `excluded` or `reference_only` with a clear reason.
- If potentially useful, put in `review_needed`.

## Visual/Source Review

For suspicious tables, inspect local source images and OCR JSONs where available.

Primary local source-image root:

- `export/digitized_tables_referenced/digitized_tables/`

Also check:

- `export_final/browse/final_submission_review/review_assets/`
- `export_final/browse/by_topic/mortality/review_assets/`
- any `source_image_local` path already listed in catalogs

Use the source image to decide whether suspicious values are real or extraction artifacts.

If image inspection confirms extraction is wrong, exclude or mark for rerun.

If image inspection confirms the value is real but historically surprising, keep it only with a manual note.

Record evidence in `qa/manual_image_review_notes.csv`:

- `table_id`
- `inspected_image_path`
- `inspected_json_path`
- `issue_seen`
- `decision`
- `notes`

Prioritize image review for:

- assembled tables
- mortality tables
- any table with constraint failures
- any table with repeated constant columns across age rows
- any table with odd 0% or 1-per-mille rates
- any table with extreme sex imbalance
- any table with large married/widowed counts in child ages

## Correction Workflow

When a table fails plausibility:

1. Inspect the source image and OCR JSON locally if available.
2. If the correct interpretation is obvious and can be safely repaired programmatically, create a repaired table artifact and document the repair.
3. If repair requires OCR/model re-extraction, put the table in `review_needed` with recommendation `rerun_ocr_or_gemini`.
4. If the table is off-topic or clearly malformed, exclude it.
5. If the table is plausible but surprising, keep it only with a manual confirmation note.

Do not include in final confirmed output:

- massive child widowed/married artifacts
- copied column totals down age rows
- tables where Male/Female/Persons relationships are nonsensical
- tables where one sex is zero and the other is huge without explicit single-sex context
- mortality tables with copied totals down disease rows
- tables with year rows and disease rows mixed into one axis incorrectly
- tables with unresolved duplicate conflicting cells
- tables with failed constraints unless manually repaired and revalidated

## Per-Table Excel Outputs

Create one Excel workbook per confirmed table under:

- `export_final/sendout_final/excel_tables/`

Use sharded folders so the directory is navigable:

- `excel_tables/core/`
- `excel_tables/tail/`
- `excel_tables/mortality/`
- `excel_tables/by_family/<table_family>/`

Each workbook should include:

- Summary sheet:
  - title, province, year, table_family, quality tier, checks/failures, source paths, review notes if any
- Matrix sheet:
  - human-readable reconstructed table where possible
  - include section explicitly so repeated row labels do not collapse
- Facts sheet:
  - the confirmed fact rows for that table
- Source sheet:
  - source image path, OCR JSON path, original paths, review notes if any

Excel requirements:

- Use memory-efficient write mode.
- Sanitize sheet names.
- Do not create enormous workbooks that Excel cannot open.
- If a table is too wide for a clean matrix, still create Summary and Facts sheets and mark Matrix as "not suitable for reconstruction".
- Produce `catalogs/excel_index.csv` listing every workbook path.

## Preview/Showcase

Create a small showcase folder:

- `export_final/sendout_final/showcase/`

Include:

- 5 clean age/civil-condition tables
- 5 clean age/sex/population tables
- 5 clean distribution-of-1000 tables that pass checks
- 5 clean mortality tables if available
- no assembled examples unless they pass visual/semantic review

For each showcase table, include:

- Excel workbook symlink or copy
- source image if locally available
- short summary text
- why it is trustworthy

## Documentation

Write `README.md` explaining:

- what is in the package
- which file to start with
- what one row in `facts_confirmed.csv` means
- why table metadata is separated from facts
- how to use the Excel workbooks
- what the confidence levels mean
- how mortality is organized
- what was excluded and why
- what remains for future review

Write `DATA_DICTIONARY.md` with concise definitions for every column in:

- `facts_confirmed.csv`
- `tables_confirmed.csv`
- `table_members.csv`
- `exclusions.csv`
- `review_needed.csv`

Write `RELEASE_NOTES.md` with:

- final counts
- included table counts by layer/family/year/province
- mortality counts
- excluded counts by reason
- review-needed counts
- caveats

## QA Outputs

Create:

- `qa/final_counts.json`
- `qa/final_counts.md`
- `qa/number_sense_audit.csv`
- `qa/number_sense_summary.json`
- `qa/schema_audit.json`
- `qa/file_integrity_report.csv`
- `qa/manual_image_review_notes.csv`
- `qa/sample_rows_preview.csv`

## Acceptance Criteria

- `facts_confirmed.csv` contains only confirmed facts.
- `tables_confirmed.csv` has one row per included table.
- `facts_confirmed.csv` does not repeat large table-level metadata fields like title, queue JSON, constraints JSON, metadata JSON, source file, or acceptance JSON.
- Every included table has `failed_checks = 0` or a documented manual override based on image review. Prefer no overrides for the sendout version.
- No included table has obvious copied-total artifacts like the Baroda Jain widowed/unmarried issue.
- No included mortality table has obvious copied totals or fake-looking repeated 0/1 rates.
- No included table has absurd sex imbalances unless explicitly single-sex and documented.
- No included table has massive child widowed/married artifacts.
- Mortality is a clearly visible subcollection.
- Every confirmed table has an Excel workbook, or a documented exception if workbook generation fails.
- The package can be understood by opening `README.md` and `tables_confirmed.csv`.
- The whole process is reproducible via a script, ideally `build_sendout_final.py`.

## Known Problematic Examples

Handle these carefully:

- Baroda 1901 assembled Jain age/civil-condition table should not be trusted unless repaired and visually confirmed. It had repeated Unmarried/Widowed totals down child age rows.
- Baroda 1911 mortality "Reported deaths from certain diseases per mille..." should not be in the clean mortality set. It mixed year/disease rows and repeated totals/rates.

## Implementation Guidance

- First inspect existing files and scripts before editing.
- Reuse existing package/codebook/review-app conventions where possible.
- Add a new builder script rather than destructively changing the current package.
- Prefer streaming CSV readers/writers.
- Use openpyxl or xlsxwriter in write-only/constant-memory mode for Excel.
- Keep generated artifacts under `export_final/sendout_final`.
- After building, run a final QA pass and summarize exactly what was included, excluded, and left for review.

## Final Response Should Include

- path to `export_final/sendout_final`
- final table/fact counts
- where to open the confirmed data
- where to open mortality
- where to open Excel workbooks
- where to inspect exclusions/review-needed tables
- any remaining caveats
