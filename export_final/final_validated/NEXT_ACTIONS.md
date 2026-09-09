# Next Actions To Reach Full Readiness

Current status: `not_ready_pending_actions`.

## Required closure order

1. Rerun no-output tables.
   - Queue: `review/queues/no_output_rerun_queue.csv`
   - Count: `582`
   - Run on the cluster/API environment, then rebuild sendout and this package.

2. Fast manual review.
   - Queue: `review/queues/manual_review_queue.csv`
   - Count: `73`
   - Use the review app and mark `keep` or `drop`.

3. Assembled review.
   - Queue: `review/queues/assembled_visual_review_queue.csv`
   - Count: `29`
   - Use the assembled renderer/workbook; accept or reject whole assembled tables.

4. Repair/rerun candidates.
   - Queue: `review/queues/repair_or_rerun_queue.csv`
   - Count: `263`
   - Repair extraction or rerun, then revalidate constraints.

5. Scope decisions.
   - Queue: `review/queues/scope_decision_queue.csv`
   - Count: `477`
   - Decide whether each table belongs in the project. In-scope tables need validation; out-of-scope tables move to an `excluded_*` tier.

6. Exclude-or-rerun decisions.
   - Queue: `review/queues/exclude_or_rerun_queue.csv`
   - Count: `290`
   - Inspect and either rerun or permanently exclude.

7. Mortality scale annotations.
   - Template: `review/mortality_scale_annotation_template.csv`
   - Count: `14`
   - Fill `scale_annotation` before using mortality rates in pooled analysis.

## Rebuild command

```bash
python3 build_final_validated_export.py
```

## Done condition

Full readiness requires:

- `tables_pending_action == 0`
- every audited table has a terminal tier: `validated_confirmed`, `validated_manual_accept`, an `excluded_*` tier, or another explicit validated/excluded terminal status
- no facts refer to excluded tables
- full plausibility audit has zero unresolved fail/warn rows for validated tables
- mortality scale review queue is empty for any mortality table used in pooled rate analysis
