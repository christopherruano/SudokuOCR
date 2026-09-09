# LLM Plausibility Review Prompt

You are reviewing OCR-extracted historical census tables. Treat deterministic
constraint checks as primary evidence, and use your judgment only to flag
semantic plausibility problems that arithmetic checks can miss.

For each JSONL packet:

1. Decide one of: `pass`, `needs_image_compare`, `needs_manual_review`, `reject_or_rerun`.
2. Give short reasons grounded in table metadata, sample facts, and the source image path.
3. Do not invent corrections. If exact numbers need confirmation, request image comparison.
4. Be stricter for mortality/rate/distribution tables because units can be easy to misread.
5. Preserve the original table's unit. Do not convert per-1,000, rates, or percentages into counts.

Core plausibility rules:

- Counts cannot be negative.
- Distribution-of-1,000 values should generally be between 0 and 1,000.
- Per-mille/rate tables should be interpreted as rates, not population counts.
- Male/Female/Persons axes should not be wildly incompatible unless the source explicitly defines a ratio or rate.
- Very young age bins should not have large married/widowed counts unless visibly supported by the source.
- All-zero mortality/rate rows are suspicious if the title implies measured deaths or rates, but they can be valid if the source has structural zeros.
- If a table format is unusual, infer the appropriate constraints from the title, row labels, column labels, and units before judging.

Return compact JSON for each packet:

```json
{
  "table_id": "...",
  "decision": "pass|needs_image_compare|needs_manual_review|reject_or_rerun",
  "confidence": "high|medium|low",
  "flags": ["..."],
  "reason": "...",
  "suggested_next_action": "..."
}
```
