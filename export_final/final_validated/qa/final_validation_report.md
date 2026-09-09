# Final Validation Report

- readiness_status: `not_ready_pending_actions`
- audited_tables_total: `3647`
- tables_all_validated: `1933`
- facts_all_validated: `734157`
- tables_age_profile_panel_candidates: `9`
- facts_age_profile_panel_candidates: `6600`
- tables_pending_action: `1714`
- tables_excluded_or_unextractable: `0`

## Pending By Tier

- `pending_assembled_visual_review`: `29`
- `pending_exclude_or_rerun_decision`: `290`
- `pending_manual_review`: `73`
- `pending_repair_or_rerun`: `263`
- `pending_rerun_no_output`: `582`
- `pending_scope_decision`: `477`
- `validated_confirmed`: `1933`

## Pending By Action

- `decide_scope_then_tier`: `477`
- `inspect_then_exclude_or_rerun`: `290`
- `manual_visual_review`: `73`
- `none`: `1933`
- `repair_or_rerun_then_revalidate`: `263`
- `rerun_ocr_or_extraction`: `582`
- `review_assembled_table`: `29`

## Interpretation

This package is not yet fully ready for a claim that every audited table has validated facts. It is ready as a controlled validation workspace: all current validated facts are exported, all pending tables are accounted for, and the remaining blockers are explicit.
