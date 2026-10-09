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

## Coverage ladder (atlas "What age data exists")

1. self-checked district age structure;
2. district appears in age tables, but the structure is incomplete;
3. district tables without an age breakdown;
4. province/state-level only;
5. nothing in the corpus.

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
