# QA Report — Historical Indian Census Dataset (dogfooding, 11 query verdicts)

## 1) Overall verdict
Note: the brief says 12 agents but only 11 verdicts were supplied; this report covers the 11.

- **ok: 6** — child marriage by sex/region; southern age structure (Mysore); Burma age pyramid; female widowhood 40+; Travancore taluks; Bombay city wards.
- **suspicious: 4** — North-India/Punjab sex ratio; rate-table magnitudes; age-by-religion column sums; variation-table negatives.
- **broken: 2** — "distribution-of-1000" slice; all-India decade totals.

Bottom line: the dataset is **largely reasonable and human-readable**. Every agent that inspected individual tables found real titles, real `source_image` PNGs, ordered age bands, and internal additivity (M+F=All; Unmarried+Married+Widowed=Population; single-year bands sum to subtotals) holding **exactly** in the clean tables. The two "broken" verdicts are about *slice-level labeling* and *missing aggregate rows*, not corrupted cell data. Most "suspicious" verdicts are driven by a small number of misclassified tables polluting otherwise-sound slices.

## 2) Most credible REAL data problems (not query artifacts)

These are genuine extraction/metadata defects with table_ids, ranked by impact:

1. **`measure_class` mislabeling is systemic and the single biggest problem.** Multiple slices are contaminated with the wrong measure class:
   - `measure_class=rate` contains ~86,070 age-sex population **counts** + ~34,443 age_distribution facts; max value 2,483,451 (impossible as a rate). Includes a **negative** rate min=-2,323 (Mysore 1901 -9.8) — impossible.
   - `measure_class=distribution` is almost entirely raw counts (`measure_unit` uniformly 'count', group sums to 6.2e8); only 6/353 tables are genuine proportion tables.
   - `subject=population/measure_class=count` includes a **registered-deaths** table: `United_Provinces/1901/9b499bab257d184d_t0005_...` (SUBSIDIARY TABLE V, deaths 1891-1900) whose row labels are calendar years, not ages.
   This is the highest-value fix because it pollutes population/rate/distribution analytics directly.

2. **OCR digit error (single confirmed):** caste variation table `.../be071bfafde60e80_t0009_TABLE-L...` — "Muhammadan unspecified" Net variation = **-1,981,626** but Persons (1872=1,102,371 → 1901=20,745) imply ~-1,081,626. Off by **exactly 900,000** — a leading-digit misread. Sign correct, magnitude wrong.

3. **`measure_unit` tagging bug:** Madras Ports 1881 `.../7d0e6450a5c3685c_t0038_Table-No.-146/...` — absolute "Difference" counts (-1900, -1132, -210) tagged `measure_unit=rate` mixed with the genuine Percentage column.

4. **Broken structure / non-reconciling subtotals:** `Baroda/1891/58e2514a48091e11_t0001_TABLE-VII__...` ("Ages by Religions") — `row_label_std` are bare row indices ('17'..'25'); Christian Total=0 but age bands sum to 2,114; Parsi Total=7,192 but bands sum to 8,273; Parsi sex ratio 1,284 F/1000M (implausible). Column sums untrustworthy.

5. **Broken per-mille table:** 1891 "Table No.35, proportion per mille" (source `...bb2e4d04dd6f2b17177f91844d.png`) — one column value copied into every age band (All=10000×7, M=5007×7, F=4993×7); sums to 70000 instead of ~1000/10000.

6. **Geography mislabel:** `Bengal/1931/1e17a9869c8b088_t0044_Table-7__...` has path prefix "Bengal/1931" but `canonical_name='Central India Agency'`, `broad_region='Central India'`. Path region contradicts resolved place.

7. **Title/segmentation attribution errors:** `India/.../0ea4111808aa3dbf_t0003_K-E-Percentage-of-Illiteracy...` is a 1881 Population-by-Languages table whose table_id slug comes from an unrelated illiteracy page. Madras table `6d8c200b9ae1049d_t0015` title says "MALABAR DISTRICT" but slug reads "MADURA D-STRIOT."

8. **Field-shift / leakage (low volume, but indicates header drift):** stray `civil_condition='3.0'` (1 row), `broad_region='age_civil_condition'` (1 row), `sex='F.-Minor and Unspecified Religions.'` (1 row). ~45k `broad_region` NaN values.

## 3) Readability issues in individual tables
The data is human-readable overall; specific issues:
- **Multi-section/multi-religion tables encode dimensions inconsistently** — Burma tables carry religion in BOTH a `religion` column (frequently NaN even on religion-specific rows) AND `column_group_std`. Rangoon Cantonment table has 2 sections (Cantonment + Port). Readable, but ambiguous without section awareness.
- **OCR-corrupted titles inside `table_id` slugs** (the clean `title`/`canonical_name` columns are fine — recommend always displaying those, never the slug).
- **`(Contd.)` pages expose partial schemas** — e.g. Shencottah taluk `Travancore/.../t0028` shows only Population+Unmarried; Married/Widowed are on the partner page.
- **Merged geography labels:** 22.2% of Travancore taluk facts carry "A Taluk / B Taluk" canonical_names, plus spelling-variant duplicates (Kartikapalli vs Kartikappalli) that fragment per-taluk aggregation.
- **`resolved_age_label_std` 100% empty** in some rate tables (e.g. Bengal 1931 IV-17); 1891 township tables report pctU15=100 because upper bands didn't resolve into `resolved_age_lower/upper`.
- **Duplicated rows:** Bombay 1931 death-rate table `d20e8cfca692791a_t0016` repeats identical rows across years 1921-1930.

## 4) Demographic patterns that VALIDATE the extraction
The data reproduces well-documented census-era demography — strong evidence the digits are right:
- **North-Indian female deficit:** Punjab 1901 districts Lahore 771, Amritsar 815, Gurdaspur 841; Delhi 869; UP ~913-918 — all below parity, Punjab most severe, matching history.
- **Child marriage skew:** ever-married share at ages 0-15 is ~4× higher for females (F 1.24% vs M 0.32% on clean buckets); highest in Central India (F 14.85%) and Rajputana (F 13.34%), lowest Burma (F 0.41%) — correct geography.
- **Young age structure:** Mysore 1921 under-15 = 38.3%, 15-59 = 55.4%, 60+ = 6.3% (bands sum exactly to grand total 5,859,952).
- **Labour-migration male bulge, Burma:** Rangoon Cantonment 1891 working-age sex ratios crash (20-24 SR 206, 25-29 SR 183) while rural Tharrawaddy is a balanced pyramid (SR ~983) — textbook colonial port vs rural contrast.
- **Female widowhood rises monotonically with age:** Madras 1901 Malabar females 40-60 = 53%, 60+ = 86%; Burma lower (consistent with remarriage norms).
- **Internal additivity holds exactly** across many inspected tables (M+F=All; U+M+W=Population; single-year bands = 5-year subtotals; religion parts = 99.92% of "All Religions").
- **Variation negatives are mostly legitimate** decreases (Jain -47/-49% on tiny base during plague/famine decade; districts -1.3 to -6%).

## 5) Concrete fixes to queue (specific, prioritized)

1. **Re-derive / audit `measure_class` and `measure_unit` per table.** Highest priority. Reclassify the deaths/literacy/education tables out of `subject=population/measure_class=count`; move count columns out of `measure_class=rate` and `=distribution`. Gate: no `measure_class=rate` fact should exceed ~1,000 or be negative-as-a-magnitude unless the column header explicitly says "increase(+)/decrease(-)". Flag the -2,323 Mysore 1901 rate.
2. **Fix the confirmed OCR digit error** in `be071bfafde60e80_t0009` ("Muhammadan unspecified" -1,981,626 → ~-1,081,626).
3. **Add an automated additivity validator** that recomputes Net variation vs Persons deltas, U+M+W vs Population, M+F vs All, and section-Total vs band sums; auto-flag any table where they disagree (would have caught Baroda 1891 t0001 and the per-mille table `...bb2e4d04...`).
4. **Build an overlapping-band flag.** Tag aggregate rows ("Total 0-5", "Total 1-5", "Total 5-60") distinctly from detail bands so consumers don't double-count. This single fix de-risks most "naive sum" errors agents hit.
5. **Reconcile geography:** make `canonical_name`/`broad_region` authoritative; flag and correct cases where the path-region prefix disagrees (e.g. `Bengal/1931/...t0044` → Central India). Normalize taluk spelling variants and split merged "A / B Taluk" labels.
6. **Decouple `table_id` slug from title attribution** — the slug is OCR-noisy and sometimes lifted from an adjacent page (the Languages/illiteracy and Malabar/Madura cases). Surface `title`/`canonical_name` in the app, never the slug.
7. **Clean field-shift leakage:** drop/repair the stray `civil_condition='3.0'`, `broad_region='age_civil_condition'`, `sex='F.-Minor...'` rows; investigate the ~45k `broad_region` NaNs.
8. **Fix the `table_id=` filter endpoint** — it ignored the URL-encoded ID and returned the full ~791k-row dataset for at least two agents, forcing client-side isolation. This is a query-tool bug, not data, but it blocks per-table inspection.
9. **Dedupe Bombay 1931 rate rows** repeated across 1921-1930 (`d20e8cfca692791a_t0016`).
10. **Document the sparse temporal coverage** (1921/1931/1941/1961 frequently absent) and the absence of true all-India grand-total rows so users don't attempt unreconstructable decade totals.

**Honest caveat:** several "broken/suspicious" verdicts are partly query-method artifacts (naive cross-section/cross-religion summing, overlapping subtotal double-counting, the broken `table_id=` endpoint). The underlying *cell-level* extraction is in good shape; the real backlog is **metadata/classification hygiene**, with only one confirmed digit-level OCR error found across all 11 deep dives.