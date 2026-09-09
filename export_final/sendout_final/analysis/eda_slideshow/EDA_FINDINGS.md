# Confirmed Census-Table Corpus: EDA Findings

## Dataset Summary

- **Confirmed tables:** 1,933
- **Confirmed facts:** 734,157
- **Provinces:** 39
- **Year range:** 1871–2001
- **Mortality tables:** 26
- **Mortality facts:** 4,576
- **Review-needed:** 1,132
- **Excluded:** 582
- **Plausibility-filtered snapshots:** 34 from 18 provinces

## Analysis Method

Age-structure analyses bypass the profile-unit system and compute shares
directly from cell-level facts. Cross-boundary age bins (e.g. 10--20) are
proportionally split across broad groups (0--14, 15--59, 60+). This recovers
usable data from tables the profile pipeline cannot bucket.

Plausibility filter: pop >= 5,000, child share 10--55%.
Life tables are excluded by title keyword matching.

## Population Structure

The facts-based analysis recovers 34 province-year snapshots from 18 provinces,
capturing a pre-transitional demographic regime: child shares around 30--48% (aged 0--14),
elderly shares around 1--6% (aged 60+), and correspondingly high youth dependency ratios.
Population pyramids recovered from OCR-extracted cell facts show the classic broad-base,
rapid-taper shape expected for high-fertility, high-mortality populations.

## Sex Ratios

Sex ratios (males per 1,000 females) vary across age groups and provinces.
Age-varying patterns can reveal differential mortality, migration, and
enumeration bias. Historical India's well-documented male-biased sex ratios
are visible in the data. Bins with fewer than 500 persons per sex are excluded.

## Mortality

The mortality subcollection (26 tables, 4,576 facts)
is mostly death rates, not counts. The classic U-shaped age-mortality curve is recoverable
from several tables, particularly the Bengal 1931 age-sex-specific death rate series.

**Important caveat:** Mortality tables vary in denominator, scale, and rate definition.
Tables must be classified (analysis-ready, scale-review, ratio-only) and normalized
before pooled analysis.

## Research Directions

1. Construct a curated analysis layer: one preferred table per province-year, geographic crosswalk.
2. Age-heaping analysis (Whipple's index) to assess enumeration quality.
3. Mortality scale harmonization and external benchmarking.
4. Link to vital statistics and demographic estimates for validation.
5. Systematic sex-ratio-at-age analysis for evidence of gender-differential mortality.

---
*Generated from confirmed sendout files. Preliminary EDA, not a final demographic series.*
