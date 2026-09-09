# Data Dictionary

## facts_confirmed.csv

- `fact_id`: Stable row identifier assigned in this sendout build..
- `table_id`: Stable source table identifier..
- `section`: section.
- `row_label_raw`: row label raw.
- `row_label_std`: row label std.
- `row_dimension`: row dimension.
- `row_group_value`: row group value.
- `column_group_raw`: column group raw.
- `column_group_std`: column group std.
- `column_dimension`: column dimension.
- `column_group_value`: column group value.
- `sub_column_raw`: sub column raw.
- `sub_column_std`: sub column std.
- `sex`: sex.
- `civil_condition`: civil condition.
- `religion`: religion.
- `resolved_age_axis`: resolved age axis.
- `resolved_age_label_std`: resolved age label std.
- `resolved_age_lower`: resolved age lower.
- `resolved_age_upper`: resolved age upper.
- `resolved_age_open_ended`: resolved age open ended.
- `resolved_age_total`: resolved age total.
- `resolved_age_not_stated`: resolved age not stated.
- `measure_unit`: measure unit.
- `value`: value.

## tables_confirmed.csv

- `table_id`: Stable source table identifier..
- `layer`: Source dataset layer: core, tail, or assembled..
- `province`: province.
- `year`: year.
- `title`: title.
- `table_family`: table family.
- `format_class`: Sendout-inferred table format used for plausibility checks..
- `measure_category`: measure category.
- `constraint_regime`: constraint regime.
- `denominator_hint`: denominator hint.
- `effective_quality_status`: effective quality status.
- `quality_tier`: quality tier.
- `confidence_level`: confidence level.
- `n_checks`: n checks.
- `failed_checks`: failed checks.
- `qc_flags`: qc flags.
- `source_image_local`: source image local.
- `source_json_local`: source json local.
- `source_image_original`: source image original.
- `source_json_original`: source json original.
- `facts_rows`: facts rows.
- `excel_path`: Path to the per-table workbook relative to sendout_final..

## table_members.csv

- `table_id`: Stable source table identifier..
- `member_index`: member index.
- `member_source_image`: member source image.
- `member_source_json`: member source json.
- `member_title`: member title.
- `member_page_status`: member page status.

## exclusions.csv

- `table_id`: Stable source table identifier..
- `layer`: Source dataset layer: core, tail, or assembled..
- `province`: province.
- `year`: year.
- `title`: title.
- `reason_category`: Machine-readable reason the table is not confirmed..
- `reason_detail`: Human-readable detail for the exclusion or review decision..
- `source_image_local`: source image local.
- `source_json_local`: source json local.

## review_needed.csv

- `table_id`: Stable source table identifier..
- `layer`: Source dataset layer: core, tail, or assembled..
- `province`: province.
- `year`: year.
- `title`: title.
- `reason_category`: Machine-readable reason the table is not confirmed..
- `reason_detail`: Human-readable detail for the exclusion or review decision..
- `source_image_local`: source image local.
- `source_json_local`: source json local.
- `recommendation`: recommendation.
