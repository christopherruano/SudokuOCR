# Team-Ready Unified Export

This folder is the fast handoff layer for the full release.

It includes:
- the three shipped release datasets
- one unified catalog covering all read tables
- separate release-scope and reference-only splits

Release datasets:
- `datasets/core_curated_v2_final`
- `datasets/tail_exploratory_v2`
- `datasets/assembled_strong_v2`

Use these entrypoints:
- `catalogs/all_tables_master.csv`: concise one-row-per-table index for the whole release
- `catalogs/all_tables_full_metadata.csv`: full source-catalog metadata plus derived QC/review fields
- `catalogs/release_scope_tables.csv`: the shipped release-scope tables
- `catalogs/reference_only_tables.csv`: read tables outside the final release scope
- `catalogs/final_clean_tables.csv`: strict-only, no-manual-review demo subset
- `catalogs/manual_review_queue.csv`: all tables that still warrant later manual review
- `catalogs/mortality_tables.csv`: mortality/death-related subcollection
- `browse/mortality`: dedicated mortality browsing folder
- `browse/review_queue_all`: full review queue with preview folders and app-backed review flow
- `apps/launch_final_submission_review.sh`: launch the general review app from the repo root
- `apps/launch_mortality_review.sh`: launch the mortality review app from the repo root

Confidence tiers:
- `high`: strict validated
- `medium`: contextual unvalidated
- `low`: needs review
- `reference_only`: not in the shipped release scope, but still included here for browsing and follow-up

Manual review signals:
- `needs_manual_review=1` marks tables that are in the review queue or assembled-review overlay
- `assistant_suggested_decision`, `manual_review_decision`, and `manual_final_decision` preserve the current review state

Important note:
- This is a showable unified export, not a deduplicated analytical union.
- Assembled tables can overlap page-level core/tail tables.
- Keep `layer` in any team presentation so people do not assume the three layers are already merged.
- For the safest team demo, start in `final_clean/`.
