# Place-string curation — instructions (shared by all chunks)

You are resolving strings extracted from digitized British-India census tables (1871–1951) to
historical administrative units and their modern equivalents. This feeds a research crosswalk that
will be published, so be accurate and honest: `unsure` is always better than a confident guess.

## Input
`atlas/curation/in_<chunk>.csv`, columns:
- `key` — normalized string (copy it unchanged into your output; it is the join key)
- `label` — most common raw spelling; `variants` — other raw spellings seen
- `n_tables` — how many tables contain it
- `sources` — where it was found: `row` (a row label of a table whose rows are places and columns are
  ages/sex/civil condition), `colg` (a column group), `sect` (a section header), `title`
  (from the table title), `tag` (the table's own place tag)
- `context` — up to 3 examples of `broad_region | table's place tag | census year | table title`

## Reference
`atlas/curation/gadm_reference.tsv` — every modern district (GADM 4.1): `country, admin1, district,
variants`. Countries: IND, PAK, BGD, MMR. Names are CamelCase without spaces, e.g. `UttarPradesh`,
`KanpurNagar`. Search it (grep) for every modern district you assign.

## Output
Write `atlas/curation/out_<chunk>.csv` with exactly these columns, one row per input row, same order:

`key,kind,hist_name,hist_parent,modern,confidence,note`

- `kind` — one of:
  `district` · `sub_district` (taluk, tahsil, pargana, subdivision, township, mahal, prant) ·
  `town` (a town/city/municipality/cantonment) · `city_section` (ward/section of a city, e.g. Bombay's
  Girgaum) · `princely_state` · `province` (British province or presidency, e.g. Bengal, Berar,
  Central Provinces) · `division` (revenue/administrative division, e.g. Meerut Division) ·
  `agency` (political agency, e.g. Bhopawar Agency) · `natural_division` (census natural/geographic
  region, e.g. "Brahmaputra Valley", "Western Malwa") · `aggregate` (several units in one string,
  e.g. "Hoshangabad, Nimar") · `country` (a modern or historical foreign country, e.g. France, Nepal) ·
  `not_place` (age band, caste, religion, occupation, heading, OCR noise) · `unsure`
- `hist_name` — canonical historical name in contemporary census spelling plus unit word, e.g.
  `Cawnpore District`, `Bhilsa Pargana`, `Sakti State`, `Berar`, `Nowgong District`. Blank for
  not_place/country/unsure.
- `hist_parent` — the province/presidency/agency/state it sat in, e.g. `United Provinces`,
  `Gwalior State`, `Central Provinces`, `Assam`, `Burma`, `Madras`. Use the period name
  ("North-Western Provinces" before 1902 is fine as `United Provinces` — pick `United Provinces`).
- `modern` — the modern GADM district(s) covering the unit's territory, `;`-separated, each written
  `COUNTRY/admin1/district` copied EXACTLY from gadm_reference.tsv. For a district that split, list all
  successors. For a sub-district/town/state, give the single modern district containing it (or the
  few it straddles). For provinces/agencies/natural divisions/aggregates/countries leave it blank.
- `confidence` — `high` (you know this unit and its successor), `medium` (likely), `low` (best guess).
- `note` — short: rename/split evidence or why unsure, e.g. "Cawnpore=Kanpur; split Kanpur Nagar+Dehat".

## Rules
- Use the context: the same string can be different places in different provinces (e.g. Partabgarh
  is a UP district and a Rajputana state). Decide by the region/table in `context`.
- Rows from birth-place/migration tables (title mentions birth-place, born, immigrants) are still
  places — classify them normally; we filter those tables elsewhere.
- Strings that combine a place with a subset ("Kolar District Urban", "Shimoga District-Brahmin
  Males") → kind of the underlying place, note "subset".
- OCR mangling is common ("Burmah", "Allygurh", "Moozuffernuggur", "Itáwá") — resolve it when clear.
- Do not invent GADM names. If you cannot find the successor in gadm_reference.tsv, leave `modern`
  blank and say why in `note`.
- Work through ALL rows. Write the CSV with a proper CSV writer (quote fields containing commas).
  When done, reply with only: rows written, and counts by kind.
