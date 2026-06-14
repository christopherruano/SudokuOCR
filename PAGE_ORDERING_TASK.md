# Task for a separate window: fix page ordering of assembled multi-page Table VII

## Why this matters (read first)
We assemble multi-page census Table VII (Age, Sex & Civil Condition) from individual
scanned pages. The pages MUST be in printed-page order, and the assembled facts must
follow that order, or you can't tell which fact came from which scan. Right now the
member pages are ordered by an earlier, unreliable page-number read, so **very few
assembled tables are actually in page order**. The printed page number is **always at
the TOP of each scan**. Your job: read those page numbers accurately (use **Opus**
vision — these can be faint/handwritten and Haiku misreads them), then re-order.

Repo: `/Users/chrisruano/Research/SudokuOCR`  (work from here)

## Inputs
- `export_final/sendout_final/sendout_v1/page_reads_fixed.json` — list of entries,
  each `{table_id, dochash, seq, page_number, page_visible, ...}`. This is what the
  assembler sorts by. **You will update `page_number` (+ set `page_visible: true`).**
- `export_final/sendout_final/sendout_v1/assembled_members.json` — `{asm_table_id: [member_table_ids]}`.
- `export_final/sendout_final/sendout_v1/tableVII_verdicts.csv` — the review set;
  start with the `status in (unrecoverable, retrying)` assembled tables (~94 tables,
  ~479 member scans), then expand to ALL assembled tables if time allows.
- The imghash is embedded in each member `table_id` as the 40-hex before `_oneshot`
  (e.g. `digitized_tables/<dochash>_t0013_.../<40hex>_oneshot`). Resolve the local scan
  with `census_explorer.resolve_img(<imghash>.png)` (it returns an absolute path or None).

## Steps
1. **Back up first:** `cp page_reads_fixed.json page_reads_fixed.json.bak_pageorder`
2. **Build the scan list:** for each assembled table in the review set, for each member
   table_id, extract the imghash and resolve the scan. Keep `(member_table_id, scan_path)`.
   Skip members whose scan won't resolve (e.g. age_tables "existing" members with no imghash).
3. **Read page numbers with Opus vision.** For each scan, open the image and read the
   **printed page number at the TOP of the page** (header/running head). Return the integer.
   Also note any "—concluded"/"—continued"/"(2)" continuation markers (helps tie-break).
   - Recommended: fan out with parallel Opus subagents (Task/Agent tool, `model: opus`),
     ~15–20 scans per agent, each writing its batch to a part file like
     `export_final/sendout_final/sendout_v1/pageno_parts/part_<n>.json` as
     `[{"table_id": "...", "page_number": 140, "confidence": "high"}]`.
     Subagents CAN write files; have each validate its JSON before writing.
   - If a number genuinely isn't legible, set page_number to null and leave it.
4. **Apply:** load all part files; for each `{table_id, page_number}` update the matching
   entry in `page_reads_fixed.json` (`page_number = <int>`, `page_visible = true`).
   Write the file back (same JSON list shape).
5. **Re-assemble + rebuild.** Run the full pipeline (it sorts members by the new
   page_number AND captures the parallel sibling-label re-OCR that the other window
   is producing into `export/ocr_results_referenced/`):
   ```bash
   bash reingest.sh                          # ~15-20 min: extract -> assemble(by page#) -> DB -> verdicts
   python3 prep_failing_plausibility.py      # regenerate the review queue (now in page order)
   ```
   COORDINATION: only ONE window may run `reingest.sh` / `build_explorer_db.py` at a
   time (they rebuild the same SQLite). Tell the other window before you start; it will
   hold its rebuild. This single run captures both the page-ordering and the re-OCR.
6. **Restart the review app** so the reordered pages show:
   ```bash
   pkill -f llm_plausibility_review_app.py
   F=export_final/final_validated/llm_plausibility_review/failing
   python3 llm_plausibility_review_app.py --results-csv $F/failing_plausibility_results.csv \
     --packets-jsonl $F/failing_packets.jsonl --decisions-csv $F/failing_decisions.csv \
     --host 127.0.0.1 --port 5002 &
   ```
7. **Verify:** for a sample of assembled tables, confirm `assembled_members.json` member
   order now corresponds to ascending page_number, and spot-check 3–4 in the app at
   http://127.0.0.1:5002 — the "Page 1 of N … Page N of N" scans should read in true
   printed-page order, and the facts matrix should track them.

## Guardrails
- ONLY edit `page_number`/`page_visible` in page_reads_fixed.json — don't touch other fields.
- Don't invent page numbers; null is better than a guess (the assembler keeps original
  order for ties/zeros).
- This is independent of the OCR content — you're only fixing ORDER. Don't re-OCR facts.
- A `build_explorer_db.py` run takes a few minutes (large SQLite); that's normal.
