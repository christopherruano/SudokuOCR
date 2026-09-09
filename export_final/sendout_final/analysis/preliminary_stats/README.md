# Preliminary Confirmed-Table Stats

These files are generated from `data/tables_confirmed.csv` and `data/facts_confirmed.csv`.

## Scope

- Confirmed tables: 1,933
- Confirmed facts scanned: 734,157
- Primary age-over-time facts used: 20,791
- Age profile units built: 2,382
- Complete `All`-sex broad age profiles: 73

## Important Interpretation Note

The age-over-time summaries are preliminary profile-level summaries. They are useful for
seeing broad temporal patterns in the confirmed age tables, but they are **not** a
deduplicated population series. Multiple confirmed tables can cover overlapping geographies
or subgroup sections, so do not sum `assigned_population_observed` across rows as if it were
an all-India total.

The primary series uses only strict-validated additive age/sex count tables, excludes rate
and per-1,000 tables, and keeps only age bins that can be assigned cleanly to:

- `child_0_14`
- `working_15_59`
- `older_60_plus`

## Median Broad Age Shares By Year

| year | profile units | tables | child 0-14 | working 15-59 | older 60+ |
|---|---:|---:|---:|---:|---:|
| 1881 | 8 | 8 | 28.1% | 69.9% | 2.8% |
| 1891 | 21 | 6 | 12.7% | 75.0% | 5.4% |
| 1901 | 3 | 2 | 2.3% | 84.0% | 11.4% |
| 1911 | 2 | 2 | 47.6% | 50.8% | 1.6% |
| 1931 | 2 | 1 | 32.7% | 61.7% | 5.5% |
| 1951 | 12 | 6 | 38.6% | 58.7% | 2.1% |
| 1961 | 25 | 4 | 2.0% | 94.4% | 3.5% |

## Files

- `age_profile_units.csv`: one table/section/entity/sex profile with broad age counts and shares.
- `age_over_time_by_year.csv`: median/mean/weighted broad age shares by census year for `sex=All`.
- `age_over_time_quick_descriptive_by_year.csv`: stricter quick-look version excluding tiny, extreme, and special-topic profiles.
- `age_over_time_by_year_sex.csv`: same summary split by sex.
- `age_over_time_by_province_year.csv`: same summary by province-year.
- `age_profile_plausibility_flags.csv`: profile units with small totals, extreme shares, or special-topic keywords.
- `coverage_by_*.csv`: confirmed table coverage by family, format, measure, year, and province.
- `preliminary_confirmed_stats.xlsx`: browseable workbook version.
