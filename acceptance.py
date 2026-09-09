"""Trust and acceptance logic for the oneshot extraction pipeline.

- `accepted_high_confidence` means safe to finalize automatically
- `accepted_with_risk` means structurally plausible but not safe to finalize
- `rejected_constraints_failed` means extraction is not internally consistent
"""

from copy import deepcopy


def build_acceptance_assessment(report, age_warnings, signals):
    """Build a conservative acceptance assessment for oneshot pipeline output."""
    signals = deepcopy(signals)
    score = 100
    reasons = []

    def flag(reason, penalty):
        nonlocal score
        reasons.append(reason)
        score -= penalty

    if not report.get("all_passed", False):
        flag("constraints_failed", 100)
        decision = "rejected_constraints_failed"
        next_step = "review_or_escalate"
    else:
        if age_warnings:
            flag("age_ordering_warnings", 35)
        if signals.get("initial_failures", 0) > 0:
            flag("initial_constraint_failures", 30)
        if signals.get("l1_repair_count", 0) > 0:
            flag("direct_l1_repair_used", 15)
        if signals.get("targeted_recheck_used"):
            flag("targeted_recheck_used", 20)
        if signals.get("repair_entries_count", 0) > 0:
            flag("repair_cascade_used", 25)
        if signals.get("repair_api_calls", 0) > 0:
            flag("repair_api_calls_used", 15)
        if signals.get("extraction_mode") == "per_group":
            flag("per_group_extraction_fallback", 10)
        if signals.get("extraction_attempt_failures", 0) > 0:
            flag("top_level_retry_needed", 15)
        if signals.get("rows_extracted", 0) == 0:
            flag("no_data_rows_extracted", 55)
        if signals.get("final_total_checks", signals.get("initial_total_checks", 0)) == 0:
            flag("zero_constraint_checks", 35)

        decision = ("accepted_high_confidence"
                    if not reasons else
                    "accepted_with_risk")
        next_step = "finalize" if decision == "accepted_high_confidence" else "review_or_escalate"

    return {
        "version": "v2_oneshot",
        "decision": decision,
        "next_step": next_step,
        "trust_score": max(score, 0),
        "reasons": reasons,
        "signals": signals,
    }
