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

- **Composition** comes from the 1941 Census of India district map (India State Stories,
  FLAME University; CC0; `mapdata/historical_districts/`). It covers present-day India.
  The digitization places units correctly relative to one another, but its lines are
  positionally coarse: the median centroid displacement against GADM is about 15 km, in
  no consistent direction. So it is used for composition only.
- **Lines** come from GADM 4.1 sub-districts (tehsils, upazilas, townships). Each takes
  the identity of the 1941 unit covering most of it.
- **Small units:** the 77 units smaller than any modern sub-district are drawn straight
  from the 1941 polygon, clipped to land.
- **Ownership in year Y** is decided once per 1941 record:
  - a princely state is always itself;
  - a British district is itself, except before a dated split or rename in
    `district_changes.csv` (e.g. Guntur in Kistna until 1904);
  - districts of a princely state may fold into their state when it was tabulated as
    one unit (e.g. Hyderabad in 1881).
- **Name matching** requires the same unit type (Benares State ≠ Benares District) and
  overlap with the unit's mapped area, so homonyms in other provinces cannot match.
- **Outside the 1941 map** (Pakistan, Bangladesh, Burma), outlines are unions of GADM
  units assigned through the crosswalk. `unit_periods.csv` holds dated start and end
  years (Attock 1904, Lyallpur 1904, Sheikhupura 1919, Wun→Yeotmal 1905, …). Unassigned
  modern districts join the nearest unit in the same country.
- **Source fixes:** `map1941_fixes.csv` corrects source rows that carry a province name
  instead of the state's (Tehri Garhwal, Rampur, Benares State), each identified by
  location and area.
- **Checked against** the 1909 Imperial Gazetteer sheets for Madras (north and south),
  the Punjab and the United Provinces (Wikimedia Commons). The units and their
  arrangement agree. Limitation: changes that never crossed a 1941 line are invisible
  (e.g. Basim, abolished 1905, has no polygon of its own and shows within Akola).

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
