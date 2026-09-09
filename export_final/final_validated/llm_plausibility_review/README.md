# LLM Plausibility Review

Offline packet export for table-by-table LLM sense checks.

Files:

- `validated_table_packets.jsonl`: one compact packet for every currently validated table.
- `priority_review_packets.jsonl`: packets with rate/mortality/unit/heuristic priority signals.
- `llm_review_queue.csv`: sortable table-level queue.
- `llm_review_results_template.csv`: place to store model or human decisions.
- `llm_review_prompt.md`: prompt/schema for review.
- `gemini_flash_full/gemini_plausibility_results.csv`: completed Gemini Flash review for all validated tables.
- `gemini_flash_full/decision_summary.json`: compact model-decision summary.

Commands:

```bash
python3 run_gemini_plausibility_review.py \
  --model gemini-2.5-flash \
  --output-dir export_final/final_validated/llm_plausibility_review/gemini_flash_full \
  --exclude-key-slots 1,2,5

python3 llm_plausibility_review_app.py
```

Use this as a triage layer. A `pass` from an LLM is not a replacement for the deterministic audit or visual image comparison.
The completed run is intentionally conservative: non-pass results mean "review this",
not automatically "bad table".
