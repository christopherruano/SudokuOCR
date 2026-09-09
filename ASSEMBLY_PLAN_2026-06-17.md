# First-Principles Assembly Plan

Snapshot used:
`export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_overnight_priority_20260617c_20260617_085848`

Primary workbook:
`export_final/pdf_recovery_incoming_2026-06-16_first_principles_assembly_release_snapshot_overnight_priority_20260617c_20260617_085848/assembly_release_snapshot_workbook.xlsx`

## Current State

| Item | Count | Meaning |
| --- | ---: | --- |
| Materialized physical assemblies | 41 | Pairwise consecutive page assemblies produced from current OCR evidence. |
| Structure-ready physical assemblies | 37 | Safe to use as assembled structure, with values still subject to normal value review. |
| Physical assemblies needing review | 4 | Already materialized, but row/block or value confidence needs eyes before promotion. |
| Adjacent manual review candidates | 36 | Consecutive candidate pairs held as preview-only, not physical merges. |
| Core Table VII manual candidates | 15 | Most important manual-review set for row-label continuation decisions. |
| Missing OCR adjacent edges | 29 | Candidate edges blocked until OCR artifact exists, then retest. |
| Missing page bridges | 110 | Nonconsecutive likely bridges; never merge until intermediate pages are recovered and adjacent edges retest. |
| Structural/semantic conflict edges | 171 | Explicit keep-separate records. |
| Site-ready logical sibling sets | 13 | Logical navigation groups only, never physical merges. |
| Logical sibling sets needing review | 133 | Possible same-table/same-topic collections held for review. |
| Row-label potential missed count | 0 | Current audit does not see an uncovered row-label continuation candidate outside tracked queues. |
| Release audit failures | 0 | No release-blocking invariant failures. |
| Release audit warnings | 2 | Warnings are carried into review queues, not silently promoted. |

## Assembly Rules

Physical assembly is allowed only when all of these are true:

| Rule | Requirement |
| --- | --- |
| Same source | Pages must share the same source PDF/source key. |
| Consecutive pages | PDF pages must be adjacent. A gap can be marked likely, but cannot be merged. |
| Pairwise first | Build only one edge at a time. Multi-page components must be chains of verified pairwise edges. |
| Compatible table identity | No table-number, table-subject, derivative selected-caste/race/tribe, or context/title-only switch. |
| Continuation evidence | Right page must look like continuation: no unrelated new title, repeated/continued column schema, row labels inherited or clearly next block. |
| Column order preserved | Assembled matrix must keep left-page columns before right-page continuation columns unless evidence proves otherwise. |
| Row labels resolved | No-label/right-side rows should inherit only when row counts, row order, and labels are compatible. |
| Values not decisive alone | Zero checks can block value trust, but a clear image-level continuation can still be structure-only after review. |

Logical grouping is separate:

| Logical case | Policy |
| --- | --- |
| Same table/topic/religion sibling pages | Group for navigation only. |
| Muslim/Christian/Jew/Hindu subtables in one district/source | Keep as logical sibling collection unless physically consecutive continuation rules also pass. |
| Same title across gaps | Do not physically merge. Use as a recovery/retest hint only. |
| Large or mixed logical sets | Hold for review before static-site use. |

## Review Now

These four materialized pairs should be checked first because they are already assembled but not fully clean:

| Rank | Source | PDF pages | Current evidence | Decision needed |
| ---: | --- | --- | --- | --- |
| 7 | `saoa.crl.25352804.pdf` | 65-66 | `label_overlap=1`, zero checks pass | Confirm continuation structure or keep values under review. |
| 14 | `saoa.crl.25352817.pdf` | 77-78 | `label_overlap=1`, zero checks pass | Confirm row/block continuity. |
| 20 | `saoa.crl.25352822.pdf` | 66-67 | `label_overlap=1`, zero checks pass | Confirm row/block continuity. |
| 39 | `saoa.crl.25851854.pdf` | 157-158 | `position=2`, zero checks pass | Confirm row labels/columns align visually. |

Then review the 15 core Table VII preview-only candidates:

| Priority | Source | PDF pages | Zero status | Matrix CSV |
| ---: | --- | --- | --- | --- |
| 25 | `saoa.crl.25366892.pdf` | 36-37 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/012_v46preview_9a1a43db28e92e9d.csv` |
| 26 | `saoa.crl.25366892.pdf` | 56-57 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/013_v46preview_ddfb3be8d7c778df.csv` |
| 27 | `saoa.crl.25793215.pdf` | 199-200 | `no_zero_check_available` | `.../manual_preview_matrix_by_candidate/017_v46preview_c47a7f18a5b3d62c.csv` |
| 28 | `saoa.crl.25851854.pdf` | 112-113 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/026_v46preview_5073ce7cd00c9ac4.csv` |
| 29 | `saoa.crl.25851854.pdf` | 132-133 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/028_v46preview_57edce1a8c74b65e.csv` |
| 30 | `saoa.crl.25851854.pdf` | 138-139 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/029_v46preview_40428e7adbc63d6e.csv` |
| 31 | `saoa.crl.25851854.pdf` | 156-157 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/032_v46preview_2260b312a6bd8813.csv` |
| 32 | `saoa.crl.25851854.pdf` | 168-169 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/034_v46preview_1c80c9c09ff893ba.csv` |
| 33 | `saoa.crl.25851854.pdf` | 186-187 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/035_v46preview_1544e416ea4bd02a.csv` |
| 34 | `saoa.crl.25851854.pdf` | 38-39 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/019_v46preview_c399078c96f37816.csv` |
| 35 | `saoa.crl.25851854.pdf` | 44-45 | `zero_check_failures_review` | `.../manual_preview_matrix_by_candidate/020_v46preview_9fae6af96485a711.csv` |
| 36 | `saoa.crl.25851854.pdf` | 48-49 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/021_v46preview_f31c30e74cac4410.csv` |
| 37 | `saoa.crl.25851854.pdf` | 60-61 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/022_v46preview_4e4dd55db9e224c8.csv` |
| 38 | `saoa.crl.25851854.pdf` | 62-63 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/023_v46preview_de186ce5d395d673.csv` |
| 39 | `saoa.crl.25851854.pdf` | 78-79 | `zero_checks_pass` | `.../manual_preview_matrix_by_candidate/024_v46preview_0bdda60a546bfb1d.csv` |

For each preview pair, record one of:

| Decision | Meaning |
| --- | --- |
| `promote_structure` | The two adjacent pages are one continuous table; values can still be separately flagged. |
| `keep_separate` | There is a table/subtable/row-universe switch or the right page is not continuation. |
| `needs_image_review` | Matrix/OCR evidence is insufficient; check scan image before deciding. |

## Do Not Review Yet

| Queue | Count | Why held |
| --- | ---: | --- |
| Core Table VII missing-page bridges | 23 | Recover missing intermediate page(s), then retest adjacent pairs. |
| Core Table VII missing OCR edge | 1 | Wait for OCR artifact, then retest. |
| All missing-page bridges | 110 | Nonconsecutive by definition, so no physical merge now. |
| All missing-OCR adjacent edges | 29 | At least one side lacks parseable OCR evidence. |

## Infrastructure Work Remaining

The next reporting improvement should add a canonical grouping record to every release snapshot:

| Output | Purpose |
| --- | --- |
| `assembly_grouping_record.csv` | One row per grouping proposal with a single vocabulary for physical, logical, blocked, manual, conflict, and recovery states. |
| `assembly_rule_policy.csv` | Static policy table documenting which rules permit physical merge, logical-only grouping, review-only status, or no-merge. |
| Workbook sheets `Grouping Record` and `Rule Policy` | Human-readable version for static-site planning and review. |
| Audit checks | Prove physical merges are only pairwise consecutive, logical groups never imply physical merge, missing-page bridges never merge, and conflicts stay separate. |

## Static Site Update Path

Use these layers in order:

| Layer | Static-site treatment |
| --- | --- |
| 37 structure-ready physical pairs | Show as assembled table structures, values still flagged according to value policy. |
| 4 review-required materialized pairs | Hide or mark review-required until accepted. |
| 15 core manual preview pairs | Do not publish as assembled until manually accepted. |
| 13 site-ready logical sibling sets | Add as navigation/logical grouping only. |
| 133 review logical sibling sets | Hold until review. |
| Missing OCR/page queues | Use only as recovery worklists. |
| Conflict edges | Keep separate and preserve conflict reason for traceability. |

## Verification Gates

Before any release is treated as static-site ready, these gates must pass:

| Gate | Current state |
| --- | --- |
| Release audit failures | `0` |
| Grouping safety failures | `0` |
| Safe physical chains larger than one edge | `0` |
| Nonconsecutive physical merges | `0` |
| Row-label potential missed count | `0` |
| Missing-page candidates physically merged | `0` |
| Logical sibling records physically merged | `0` |

The goal is not to maximize assemblies. The goal is to avoid false assemblies while keeping every plausible continuation visible in a review or recovery queue.
