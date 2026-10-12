# District atlas, place inventory and census-order views

Scripts that turn the digitized corpus (`census_explorer.sqlite`) into the site's
district atlas, per-place inventories and census-order browser. Every step is a
script; no output is hand-edited.

## Pipeline

| step | script | output |
|---|---|---|
| 1 | `build_geometry.py` | `out/districts_svg.json`: 944 modern districts (GADM 4.1 IND L2, PAK L3, BGD L2, MMR L2), simplified to SVG |
| 2 | `extract_candidates.py` | `out/place_strings.csv`: every string that could name a place (row labels of place×age tables, column groups, section headers, title phrases, table tags), with counts and context |
| 3 | curation (`curation/INSTRUCTIONS.md`, one LLM agent per region chunk) | `curation/out_*.csv`: kind, historical name, parent, modern GADM district(s), confidence, note |
| 4 | `merge_curation.py` | `out/gazetteer.csv`: merged, mechanically validated |
| 5 | `restore_june_facts.py` | restores lost cell facts for 160 tables from the June-13 snapshot (see below) |
| 6 | `build_index.py` | `out/places.csv`, `out/table_places.csv`, `out/place_metrics.csv` |
| 7 | `build_site_data.py` | `census_site/data/{places,volumes,atlas}.json` and CSV downloads |
| 8 | `build_pages.py` | the atlas, place, places, volume and search pages, plus navigation patched into every page (idempotent) |
| 9 | `build_targets.py` | `census_site/data/reextraction_targets.csv`: district-years with partial age tables |
| 10 | `build_hist_geometry.py` | per-year historical outlines (`census_site/data/hist_geo.json`); run after step 7, before step 8 |
| 11 | `build_subdiv.py` | subdivision view (`census_site/data/subdiv.json`): historical taluks on GADM level 3 |

`position.py` places every table at a PDF and PDF page via `page_number_status.csv`
(10,657 of 10,665 site tables); the printed page comes from `provenance.csv`.

## How a table is linked to a place

A table is listed under a place if the place's name, or any recorded spelling of it,
appears as:

- `tag`: the table's own place tag;
- `title`: an "X District/State/Taluk…" phrase in the title;
- `rows`: a row label of a table laid out place × (age | sex | civil condition);
- `colg`: a column group;
- `sect`: a section header.

Tables whose title marks them as birthplace or migration tables are linked as
`birthplace` and never counted as coverage: a Bombay row in a Mysore birthplace table
counts people born in Bombay, not people living there.

## Age indicators (`age_metrics.py`, applied in `build_index.py`)

Only whole-population age tables count. These are the age families (Table VII
age/sex/civil condition, age by religion, age distributions and proportional age
tables), restricted to all-religion, all-caste slices. Infirmity, literacy, caste,
occupation and single-religion tables are excluded. Wide tables print their age
columns across several pages, so consecutive pages of one printed table (same volume,
same title stem, PDF pages ≤3 apart) are merged per place before checking. The parser
handles:

- single-year bands (5 means [5, 6));
- inclusively printed bands (0-4, 5-9);
- open bands ("60 and upwards", "above 55");
- "age not stated" counts, which are included in the check and excluded from shares.

For each (table, place) the facts are split into slices by section, civil condition,
religion and column group. Single civil-condition slices are merged
(unmarried + married + widowed). The largest slice is treated as the whole population.

A figure is published only if the slice's non-overlapping age bands **sum to the
printed all-ages total within 2%**, so the table checks itself. The check is recorded
in `district_age_indicators.csv` (`self_check`). Indicators:

- percent aged 0–9;
- percent aged 60+;
- females per 1,000 males.

Each is computed only when the printed bands include the needed boundary (10, 60).

Proportional tables (per 1,000 or 10,000) are accepted when their bands sum to the base.

Final screen: plausibility bounds (0–9 share 10–45%, 60+ share 0.5–20%, sex ratio
600–1,300). Rejected groups are counted in the build log.

Ranking when several figures exist for a place-year:

1. figures that match a printed total;
2. proportional tables that sum to their base;
3. open-ended bands with no printed total;

then Table VII over other tables, then the largest population.

### Whole-population rules (added after historical review)

- **Single-religion slices are rejected** ("Musalman", "Jain", "Others", …) unless labelled
  all religions or total. 504 figures were removed by this rule.
- **Cities/towns tables never count as district figures.** Madras 1921/1931 Table VII
  Part II is cities, so "Tanjore" there is Tanjore town. The rule applies to rows,
  sections and column groups.
- **District size limits:** a district figure needs at least 30,000 people (the smallest
  real district, Simla 1901, had 40,351) and at most 6,000,000 (catches provincial
  tables carrying a district tag).
- **Relative test:** when a district has several figures in a year, any covering less
  than 40% of its largest is a subset and is dropped.
- `modern_overrides.csv` holds reviewed corrections to the modern mapping, with sources
  (Karachi, Delhi). 

`demography_check.py` compares province medians of the verified district figures with
the provincial sex ratios printed in the census reports. All province-years agree
within 60 except Burma 1901 (3 districts: Akyab, Hanthawaddy, Sagaing). That gap is
historically expected, from heavy male immigration to Lower Burma and the female
surplus in Upper Burma.

An audit of all 399 mapped districts found every modern successor inside the modern
states its province could cover.

## Coverage ladder (atlas "What age data exists")

1. self-checked district age structure;
2. district appears in age tables, but the structure is incomplete;
3. district tables without an age breakdown;
4. province/state-level only;
5. nothing in the corpus.

## Historical boundaries (atlas "Historical districts" view)

Each census year has its own map of historical districts and princely states.

### Sources

- **1941 Census of India district map.** India State Stories, FLAME University; CC0;
  `mapdata/historical_districts/`. It covers present-day India and supplies the 1941
  unit for every place.
- **India State and District Evolution Database, 1872–2025** (Jolad et al.; Harvard
  Dataverse doi:10.7910/DVN/D1AGUR; CC BY 4.0; `mapdata/district_evolution/`). Its 3,220
  links give each unit's predecessor at every census step 1872→1941. It is the primary
  source for which period unit a 1941 unit lay in during each census (`evolution.py`).
  It confirms all 16 of our hand-compiled changes (`district_changes.csv`, now only a
  fallback).
- **GADM 4.1 sub-districts** (tehsils/taluks, upazilas, townships) supply the line
  geometry.

### Construction

1. **Tehsil identity.** Each GADM sub-district takes the identity of the 1941 unit
   covering most of it, or, preferentially, of an overlapping 1941 unit whose name it
   bears (204 tehsils: Gurgaon tehsil → Gurgaon, not Delhi).
2. **Direct drawing.** 64 units smaller than any tehsil are drawn straight from the
   1941 polygon.
3. **Lineage.** In year Y, each 1941 unit belongs to its predecessor in the evolution
   database.
   - With several predecessors (e.g. Amraoti + Ellichpur → 1941 Amraoti), each tehsil
     goes to the predecessor evidenced by a historical taluk located on it
     (`build_subdiv.py`), else to the one whose mapped area covers it most.
   - Period spellings (Jullunder, Ryepoor, Godavery) are matched to registry places by
     name, recorded spellings, the modern GADM name, or closest spelling (≥85).
   - A matched place must lie within 150 km of the polygon, which keeps homonyms apart:
     Nilgiri State (Orissa) and Nilgiri District, Patna State and Patna District,
     Bilaspur State and Bilaspur District, the two Partabgarhs.
   - The 1941 map's unit types are trusted. The database's types may be overridden
     where it mis-types a unit (Tehri Garhwal as a "British district" in 1921–31;
     Patna District as a state in 1931).
4. **Carve-out.** A British district the database lists in year Y that has no 1941
   polygon gets back the tehsils inside its own mapped area, taken only from districts
   that do not themselves contain them. This restores Narsinghpur (merged into
   Hoshangabad in 1932) and Basim (abolished 1905).
5. **Reviewed tehsil overrides** (`tehsil_overrides.csv`) cover units no source
   separates (Merwara = Beawar tehsil until 1911).
6. **Source fixes** (`map1941_fixes.csv`) correct source rows named after their province
   instead of the state (Tehri Garhwal, Rampur, Benares State; the latter dated from
   1911).
7. **Outside present-day India** (Pakistan, Bangladesh, Burma), outlines are unions of
   GADM units assigned through the crosswalk. `unit_periods.csv` holds dated start and
   end years (Attock 1904, Lyallpur 1904, Sheikhupura 1919, Mianwali 1901, Khulna 1882,
   Wun→Yeotmal 1905, …). Unassigned modern districts join the nearest unit in the same
   country.

### Accuracy

- **1941 map position.** Measured on 83 control districts (same name as a modern
  district, area within 0.7–1.4×), the 1941 digitization is displaced from GADM by a
  median 14.4 km. A leave-one-out rubber-sheet (inverse-distance displacement field)
  only reduces this to 11.9 km, so the error is local, and part of it is real boundary
  change. The polygons are therefore not shifted.
- **Border tehsils.** 886 of 2,337 tehsils have less than 75% of their area in their
  assigned 1941 unit (212 below 50%). Border tehsils are the main residual uncertainty,
  mitigated by the name rule.
- **Regression tests.** `test_history.py` asserts 67 documented facts (creations,
  abolitions, renames, homonym locations) and passes.
- **Visual comparison** with the 1909 Imperial Gazetteer sheets (Madras N/S, Punjab,
  United Provinces): the units and their arrangement agree.

### Better sources not used

- No public district-level boundary file covers all of British India for 1872–1941;
  see `out/boundary_sources_report.md`.
- The recommended upgrade for the paper is to trace the 1909 and 1931 Imperial
  Gazetteer atlases (DSAL, full resolution) and carry them across years with the
  evolution database.

## Atlas fill rule

For each modern district and census year the map shows the figure for the historical
district (or princely state) that covered it. Where there is none, it shows the value
for the province/state, drawn pale and labelled "provincial". This is a best-available
hierarchical fill, not a spatial model. A modern district's province comes from the
parents of the historical units mapped to it; districts with no mapped unit take the
province of the nearest mapped district in the same modern state, and are marked as such.

## Verification performed

- Gazetteer: all 10,510 strings curated; 0 GADM codes that do not exist; 25
  region-inconsistent rows, all inspected (birthplace tables or mis-tagged tables, not
  curation errors); a random sample of 30 district mappings was checked by hand and
  all were correct.
- Lost facts: 2,601 confirmed_sent/recovered tables had no facts in the Aug-6 DB.
  - 2,330 are superseded: the same page image is a current keeper/assembled table.
    They are dropped from indices rather than shown as dead links.
  - 160 were restored from the June-13 snapshot, which has the same schema
    (`out/restored_from_june.csv`).
  - 111 have no cell data anywhere in the archive. They are listed as "not yet
    published" and need the cluster copy of `unified_facts.csv`.

## Column-structure audit and correction (October 2026)

Digit accuracy had been verified, but column structure had not. A column can be dropped,
invented, shifted or given the wrong header (e.g. printed Married/Unmarried/Widowed stored
as persons/males/females). Every published table was therefore checked column by column
against its scan, and the faulty ones were re-read under strict gates. All scripts are
in `atlas/`; every decision is in `out/column_fix_log.csv`.

1. **Detection** (`gemini_column_check.py`): Gemini 3.8 Flash compares each table's digitized
   columns, with their first values, to the scan. It uses the page image embedded in the
   source PDF (about 1,400 × 2,300 px) and gives one of eight verdicts.
2. **Calibration.** On 177 tables, the detector was compared with independent Claude
   reviewers who read the scans (`out/column_review_sample.jsonl`). Agreement was 95%
   (κ 0.93); it caught 97% of the tables the reviewers said needed fixing.
3. **Confirmation.** A blind, stratified sample of 190 tables (about 25 per verdict) was
   reviewed by Claude. The verdicts confirmed were: ok 98%, Persons-only 96%, mislabelled
   96%, wrong scan 100%, missing 84%, invented 60%, shifted 12%. A weak verdict only sends a
   table to re-reading; it never changes data by itself.
4. **Corpus estimate** (5,177 tables, reweighted by those confirmation rates):

   | status | share |
   |---|---|
   | columns correct | 45% |
   | only the printed Persons/Total column missing | 32% |
   | other column problems | 16% |
   | linked scan does not contain the table | 7% |

5. **Re-reading** (`reread_columns.py`): flagged tables are re-transcribed from the PDF
   page images with the diagnosis and the existing row labels. Facing pages are offered
   when the table continues onto them. Wrong-scan tables use the correct page, found among
   the neighbouring pages (`find_right_scan.py`).
6. **Persons-only fixes** (`apply_column_fixes.py`): no existing value is changed. A
   Persons cell is added only where it equals the existing Males + Females (count tables),
   or where the re-read's Males and Females equal the existing ones exactly.
7. **Structural fixes** (`vote_readings.py`): a second, independent reading by Gemini 3.1
   Pro. A cell is published only if:
   - both readings give the same value; or
   - they disagree and the table's printed totals decide between them.

   Cells where Persons ≠ Males + Females by one digit are withheld. Any other unresolved
   cell is left blank and counted in the table's note. A table is accepted only if:
   - an independent column check of the new reading passes;
   - its numbers overlap the previous reading (label-free);
   - Persons = Males + Females holds;
   - its own printed totals hold;
   - at least 90% of its cells are confirmed by both readings.
8. **Measured accuracy.** Claude reviewers checked 1,138 random cells from 96 re-read tables
   against the scans:
   - single reading: 97.0% correct;
   - cells both models read identically: 100% (228/228);
   - published cells after voting and arithmetic: 99.7% (337/338).

   The one residual error is a misread shared by both models and consistent with the
   printed totals.
9. **Withdrawn tables.** Wrong-scan tables whose data could not be found on a nearby page
   are taken off the site (`out/withdrawn_wrong_scan.csv`). Each old address shows a
   notice explaining why.

10. **Outcome** (`out/column_fix_log.csv`; 2,380 tables corrected):

    | check verdict | corrected | kept as before, with a page note | withdrawn |
    |---|---|---|---|
    | Persons/Total missing | 1,536 | 70 |  |
    | columns invented | 377 | 75 |  |
    | columns missing | 261 | 82 |  |
    | columns mislabelled | 167 | 14 |  |
    | columns shifted | 20 | 8 |  |
    | wrong scan | 19 (relinked) |  | 358 |

    A second pass recovered tables the first could not correct:
    - **Three-way vote:** a third independent reading (Gemini 3.7 Flash), with each cell
      decided by majority, for tables where Flash and Pro disagreed.
    - **Targeted Persons reading:** each added cell is still verified as Males + Females.
    - **Column withholding:** only the columns whose printed totals do not add up are left
      blank, instead of rejecting the table.
    - **Retries** of failed second readings.

    A table is "kept as before" if its corrected reading did not pass every gate, or if
    no second reading could be obtained. It keeps its original data with a visible note
    and remains a target for a later pass.
11. **Found along the way, not changed here:** 257 tables (`out/duplicated_sections_preexisting.csv`)
    have a section whose non-zero cells repeat another section's. That is a possible copy
    error in the original reading, listed for review.

12. **Third pass: review by reading the scans** (`out/claude_adjudication/`). Claude reviewers read
    the scans for every table still uncorrected (242 packets): disputed cells, Persons values, and
    40-cell spot-checks of large single-reading tables. Two independent auditors then re-read a
    blind sample of 188 items:

    | reviewer decision | auditors agree | rule applied |
    |---|---|---|
    | Persons values | 30/30 | kept, each also equal to Males + Females |
    | disputed cells | 67/70 | kept only if it equals a machine reading and the printed totals confirm it |
    | spot-checked tables (fresh cells) | 77/79 | both errors in one table, which was rejected; the others accepted, noted as a single reading with a spot-check |
    | "no Persons column printed" | 0/8 | discarded; those flags stay |

    Six reviewer verdicts were set aside as unreliable before applying anything (the scans
    had dropped from the reviewer's context, a sample was not checked cell by cell, or the
    scan was of another place).
13. **Frozen rows and copied columns.** Every published table is checked for:
    - a row repeating one value;
    - a column repeating one value;
    - a column copying its neighbour;
    - a serial-number column stored as data.

    Each flag was reviewed against the scan. 32 are printed that way and are listed in
    `out/frozen_exceptions.csv`. The rest were removed in 36 tables (1,896 cells), with a
    reviewed value restored only where Persons = Males + Females confirms it.
    `verify_site.py` now fails if any published table has an unreviewed frozen row or
    copied column. Exceptions confirmed by eye outside the reviewer batches are kept in
    `out/frozen_exceptions_manual.csv`, which is merged in on every run.
14. **Right pages found in the source volumes** (`find_pages.py`, `verify_found_pages.py`).
    The search covered the 451 tables still withdrawn or noted. It took each table's
    distinct numbers (three or more significant digits) and matched them against every page
    within ±12 pages of its linked scan in the same PDF. Pages were read from the PDF text
    layer, or with local OCR where a page has none. A greedy cover then picked the pages
    holding most of the table's numbers.
    - 85 tables had at least 60% of their numbers found on a set of pages.
    - Gemini 3.8 Flash ran the column check against those pages.
    - Tables that checked "ok" (24), or "ok except an undigitized Persons column" (23),
      were relinked to the found pages, with a page note.

    This brought back 40 withdrawn tables, leaving 318 withdrawn, and replaced the
    "use with care" note on 7 others. The remaining 38 are on the right page but their
    columns still disagree with the scan, so they stay as they were.

Originals are never overwritten. Corrections go to the overlay files in `sendout_v1/`
(`corrected_facts.csv`, `corrected_tables.json`, `column_fix_overlay.json`), and the
pre-correction facts of every changed table are kept in
`out/column_fix_original_facts.jsonl`.

## Worked example (checked against the scan by hand)

Gurgaon District, 1901, Punjab Table VII Part II (scan `census_site/img/25363739_p50.jpg`).
All 38 POPULATION cells (male/female × 18 age bands plus totals) match the printed page.

| | value | from the scan |
|---|---|---|
| % aged 0–9 | 27.86 | (99,518 + 108,369) / 746,208 |
| % aged 60+ | 4.85 | 36,196 / 746,208 |
| females per 1,000 males | 911.2 | 355,765 / 390,443 × 1,000 |
| self-check | pass | the 18 bands sum to the printed 746,208 |

## Verification suite

`python3 atlas/verify_site.py [base-url]` checks:

- every table link resolves to a real page, not a host fallback;
- internal consistency of all data files;
- that each published figure, recomputed from the raw cells, is identical (a
  determinism check on a 150-figure sample);
- that every figure is within bounds and carries a self-check;
- that every page type renders in headless Chrome with no JavaScript errors.

Only figures whose check passes (bands = printed total, or a proportional table
summing to its base) are called verified or drawn on the map. 850 of 1,858 figures
come from tables that print no total. They appear on place pages labelled "not
checkable" and are flagged `verified=no` in `district_age_indicators.csv`.
