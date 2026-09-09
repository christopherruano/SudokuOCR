# RESUME HERE — 2026-06-29 (while you were at the store)

## What I did autonomously
1. **Constraint-guided re-OCR pass** (your "re-OCR with the broken cells called out" idea) → **applied 6 clean fixes** (5 to 0 failures, 1 reduced) on the simple horizontal tables. Backups: `*_oneshot.preconstraint.json`. Only 7 targets existed — most failures live in transposed/dense tables the horizontal tool correctly skips.
2. **Full reingest** (re-extract → re-merge → re-QA → re-assemble → rebuild explorer DB → re-validate). Folded in the 6 fixes + the 29 stitched tables + recovered reads. ✅ Clean, all 10 steps.
3. **Verified your manual assembly work is untouched** — `tableVII_runs.json` is byte-identical to the pre-run backup (`.beforeAFK.bak`).
4. **Recomputed severity** + refreshed first-cut + rebuilt the assembled view.
5. **Restarted the review app** on fresh data (pid running, all routes 200, every queue position spot-checked).

## ⚠️ Important: the reingest re-labeled ~900 tables "unrecoverable"
The verdict ledger swung hard: **retrying 493→40, unrecoverable 468→914.** But we already proved (with scans) that **"unrecoverable" is mostly a FALSE label** — it's driven by `few_age_bands:0` (right-half spread pages whose age labels are on the facing page) and `missing_cols:MUW` (Race/Caste & distribution tables that correctly have no civil-condition columns). **Real source-limited count is small (~10).**

Because of that, I **un-hid the unrecoverable tables from `/tables`** (I'd hidden them earlier when I thought the label was real). Your `/tables` queue is back to **901 individual tables, worst-first** — real recoverable work, not an empty queue.

## Current state
| | |
|---|---|
| Verdict ledger | perfect **1,341** · retrying **40** · unrecoverable **914** (mostly false) |
| Assembled first-cut | verified **295** · minor **52** · needs-review **39** · incomplete **2** |
| `/tables` queue | **901** individual tables, **sorted worst-first** (severity = magnitude + 200·failing-rows) |
| `/assembly` queue | **363 / 397** runs, **sorted worst-first** |
| Applied this session | 29 left/right stitches + 6 constraint-guided re-OCR fixes (all backed up) |

## Recommended next steps (highest leverage first)
1. **Work `/assembly` worst-first** — it's your main surface, full, ranked. Top run has 109 failing rows.
2. **The big recovery lever: OCR the 155 missing right-halves + re-stitch.** The stitch pilot proved 29 clean recoveries; 155 more left-halves are waiting on their facing right page to be OCR'd. This is the single biggest dent in the 914 "unrecoverable." (I did NOT run this autonomously — it's a large OCR job + page-pairing; say the word and I'll launch it.)
3. **Mark distribution / Race-Caste tables `∅ Constraints N/A`** as you hit them — they're correctly read, just don't obey the identities.
4. **Re-run `python3 compute_severity.py`** after any re-OCR/edit batch to refresh the worst-first order.

## Pending / offered (not done)
- Wire `metadata.constraints_irrelevant` into the validation pipeline so Constraints-N/A survives a future reingest (currently the verdict keeps them out of your queue, but a reingest's CSV wouldn't know). ~quick.
- Mistral pass on the 34 "column-drop" unrecoverables.
- Fresh Cloudflare deploy (site is rebuilt locally via reingest; I did NOT push — left for your review).

## Handy commands
- Review app: already running at http://localhost:5058 (`/tables`, `/assembly`)
- Refresh ranking: `python3 compute_severity.py`
- Stitch more (after right-halves OCR'd): `python3 stitch_pilot.py --apply`
- Constraint-guided re-OCR: `python3 constraint_guided_reocr.py --max-fail-rows 8 --apply`
