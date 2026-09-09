# final_validated

This package is the current implementation of the full-validation export plan.

## Readiness status

`readiness_status`: **not_ready_pending_actions**

The package is intentionally honest: current confirmed facts are exported as validated,
but tables still requiring rerun, repair, scope decisions, assembled review, or mortality
scale annotation remain in explicit queues.

## Main files

- `data/tables_all_validated.csv`: currently validated table catalog.
- `data/facts_all_validated.csv`: facts for currently validated tables.
- `analysis/age_profile_panel/tables_age_profile_panel_candidates.csv`: a small conservative subset used for quick age-profile EDA panels. This is not a correctness boundary.
- `analysis/age_profile_panel/facts_age_profile_panel_candidates.csv`: facts for that EDA subset.
- `data/tables_pending_action.csv`: all tables blocking full readiness.
- `review/final_table_registry.csv`: one row for every audited table.
- `review/queues/`: action-specific queues for rerun, manual review, repair, scope, assembled review, and mortality scale review.
- `qa/final_validation_report.json`: machine-readable readiness report.
- `qa/final_validation_report.md`: human-readable readiness report.

## Tier meanings

- `validated_confirmed`: current validated facts, source-faithful and passing release checks.
- `validated_manual_accept`: manually accepted facts after visual review.
- `excluded_*`: manually excluded or unextractable after process closure.
- `pending_*`: not fully ready yet; see `final_action` and `blocking_reasons`.

The age-profile panel candidate files are intentionally narrow, because they are meant
for one clean descriptive plot family. They do not mean that only those tables are correct.
The full validated export is `data/tables_all_validated.csv` plus `data/facts_all_validated.csv`.
