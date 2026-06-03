# Historical Indian Census — Query Dataset & Explorer (v1)

Age / sex / civil-condition / religion population tables extracted from scanned
Indian Census volumes (**1872–1961**). One row per numeric cell, fully tidy and
queryable, with canonical geography and topic tags.

## TL;DR — how your research partner uses this
1. `./run_explorer.sh` → open <http://127.0.0.1:5000>
2. **Browse the tree** (🌳): drill **Region → Province/State → District → Year → table**; ⬇ at any node
   downloads the combined CSV for that whole branch. Primary way to navigate.
3. Or **🔎 Query**: filter by **subject / measure_class / region / geography-type / year** (or search titles).
4. Click **⬇ Download combined CSV** → every matching cell across those tables in one file.
5. Click 🔍 on any table → **see the source scan image** beside that table's facts.

## ⚠ The one rule for aggregating
These facts are **NOT additive across tables** — the same people appear in separate age, civil-condition
and religion breakdowns, and coarse age bands mix with fine ones. To total a population, filter to
**`subject=population` + `measure_class=count`** and work **within one table** (`GROUP BY table_id`).
Anything with **`additive=0`** (rates, percentages, distributions, ratios, variations) must never be summed.

## Query axes (refinement)
- **`subject`** — `population` · `civil_condition` · `religion`
- **`measure_class`** — `count` · `distribution` (per 1,000/10,000) · `rate` · `percentage` · `ratio` · `variation`
- **`additive`** — `1` only for counts (safe to sum within a table); `0` otherwise
- plus canonical **geography** (entity_type / broad_region / parent) and **year / decade**.

## Files
| file | what it is |
|---|---|
| `unified_facts.csv` | **785,044 facts** — one tidy row per cell, denormalized with year/geography/topic/measure/provenance/status |
| `catalog_tables.csv` | **the map** — one row per table (full corpus, 3,647), with topic/geography/year/measure/dims/n_facts/status/source image/title |
| `geo_reference.csv` | geography canon: raw name → canonical_name, entity_type, parent, broad_region |
| `census_explorer.sqlite` | the three above loaded + indexed + FTS, powering the app |
| `../../../census_explorer.py` | the Flask browse app (repo root) |
| `run_explorer.sh` | launches the app |

## What's in scope
**2,338 tables / 785,044 facts** are query-ready:
- `confirmed_sent` — 1,818 tables, passed strict validation.
- `recovered` — 520 tables rescued from the review queue (false constraint failures on
  rate/ratio/% tables + topics that were scope-excluded), re-validated under type-aware constraints.

The catalog also lists, but does **not** include facts for, tables still being finished:
- `hold_reocr` (115) — decimals dropped / male-female column data-loss → pending targeted re-OCR
- `needs_reocr` (353) — genuine constraint failures
- `needs_human_review` (245) · `excluded_or_unscoped` (582)

## Fact schema (`unified_facts.csv`)
`fact_id, table_id, section, row_label_std, row_dimension, row_group_value,
column_group_std, column_dimension, column_group_value, sub_column_std, sex,
civil_condition, religion, resolved_age_label_std, resolved_age_lower,
resolved_age_upper, resolved_age_open_ended, measure_unit, value` — plus denormalized
`year, decade, canonical_name, entity_type, parent_geo, broad_region, geo_level,
canonical_topic, table_family, measure_category, source_image, title, status`.

`measure_unit`: `count` (people), `rate`/`per_10000`/`per_1000` (rates), distribution.
`value` is numeric. Variation/"increase or decrease" tables legitimately carry negatives.

## Geography (`entity_type`)
`british_province` · `princely_state` · `agency` (Rajputana, Central India) ·
`division` · `district` · `sub_district` (taluk/township) · `urban` (city/town/ward) ·
`all_india` · `native_states_agg` · `modern_state` (1951/61 successor names) ·
`unspecified` (generic placeholder) · `not_geography` (mis-parsed token) ·
`unresolved` (3.6% long-tail, raw name preserved, flagged for manual geo review).
`broad_region`: North / South / East / West / Central India · Rajputana · Northwest
Frontier · Northeast India · Burma · Andaman & Nicobar · All-India.

## Known caveats (honest)
- **Decimals**: some rate tables had mid-number decimals dropped; those tables are in
  `hold_reocr`, not in the query set. Re-OCR with the decimal-preserving prompt is pending.
- **Synthesized "persons"**: provably-synthesized persons cells (persons == males+females
  where neither was printed) were dropped from recovered tables, matching the confirmed set.
- `topic` is currently age-focused; occupation/education/caste tables from the keeper
  OCR run will extend the taxonomy in a later version.
