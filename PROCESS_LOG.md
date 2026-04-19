# Census OCR Pipeline — Full Process Log

## Project Goal
Extract age-by-sex population tables from scanned Indian Census volumes (1871–1961). Target: 100% cell-level accuracy on every table across ~724 PDFs totaling ~380GB.

---

## Phase 1: Pipeline Development & Local Testing (Mar 30 – Apr 4)

### FASRC Setup
- Account: cruano@login.rc.fas.harvard.edu
- Lab: shleifer_lab
- Home: /n/home11/cruano (95GB quota)
- No holyscratch or netscratch write access — using home dir with batch rotation
- Venv: ~/.envs/sudokuocr
- Project: ~/SudokuOCR

### Harvard API Portal
- Portal: https://portal.apis.huit.harvard.edu/
- Endpoint: https://go.apis.huit.harvard.edu/ais-google-gemini/
- Auth: API key via x-api-key header (secret key NOT needed for Gemini)
- 3 API keys provisioned for parallel throughput
- Keys stored in .env as HARVARD_GEMINI_API_KEYS (comma-separated)
- pipeline.py modified: _get_gemini_client() checks HARVARD_GEMINI_API_KEY/BASE_URL first, falls back to GEMINI_API_KEY

### Code Changes
1. **batch_production.py**: Added --chunk/--total-chunks for Slurm array jobs, --workers for threading, --max-cost for spending caps, graceful SIGTERM shutdown, per-chunk JSONL logs, AtomicCounters, periodic checkpoints
2. **pipeline.py**: HARVARD_GEMINI_API_KEY/BASE_URL support, 300s timeout for Harvard proxy (vs 120s for Google direct)
3. **oneshot.py**: Fixed verify_all_constraints — was using row.get("age") but rows used schema label name (e.g. "AGE."). Added row_label_name to constraints dict. This was breaking resumability (all results classified as "corrupt" due to total_checks=0)
4. **fasrc/submit.sh**: Multi-key rotation (HARVARD_GEMINI_API_KEYS comma-separated, round-robin by SLURM_ARRAY_TASK_ID % N_KEYS), configurable IMAGE_ROOT/RESULTS_DIR via env vars, MAX_COST_PER_CHUNK, time limit increased from 8h to 24h, memory reduced from 4G to 1G

### Smoke Tests (7 tests, all passed)
1. Chunking correctness: all array sizes (1-300) cover all images, no gaps/overlaps
2. Result file collisions: 269 unique output paths, no collisions across chunks
3. Thread safety: BatchLogger 800/800 lines, AtomicCounters 80000/80000
4. Resumability: skips accepted_high_confidence on restart (after bug fix)
5. Graceful shutdown: SIGTERM finishes in-flight work, writes summary
6. Edge cases: empty chunks, boundary IDs, invalid IDs all handled
7. Live end-to-end: Harvard endpoint works, 164 checks, accepted_high_confidence

### Concurrency Finding
- Harvard proxy handles ~2 concurrent connections per API key reliably
- 3+ concurrent per key causes HTTP timeouts ("write operation timed out")
- Production config: %6 concurrency limit (2 per key × 3 keys)
- In practice, 9 concurrent (from overlapping jobs) worked without issues

---

## Phase 2: PDF Classification (Apr 7)

### Flash Drive Inventory
- Drive: /Volumes/Expansion/census_pdfs/
- 724 total PDFs across 10 decade directories
- Files named with CRL archive IDs: saoa.crl.NUMBER.pdf (no human-readable names)
- Individual files up to 1.6GB (entire census volumes, not individual tables)

### Classification Method
- Tool: pdftotext (poppler) — reads first 5 pages per PDF
- Zero LLM calls — pure text extraction + regex pattern matching
- JSTOR/CRL cover page has standardized title: "Census of India YEAR. Vol. N, PROVINCE : description"
- Script: classify_census_pdfs.py

### Classification Results
| Category | Count |
|----------|-------|
| age_table_confirmed | 39 |
| likely_has_age_tables | 293 |
| census_other | 62 |
| unknown | 330 |
| **Total** | **724** |

### PDF Sizes by Decade
| Decade | Size |
|--------|------|
| 1871 | 17GB |
| 1881 | 23GB |
| 1891 | 40GB |
| 1901 | 44GB |
| 1911 | 27GB |
| 1921 | 28GB |
| 1931 | 36GB |
| 1941 | 11GB |
| 1951 | 46GB |
| 1961 | 7.3GB |
| **Total** | **~279GB** |

---

## Phase 3: ToC Pipeline on FASRC (Apr 9–10)

### Pipeline Architecture
5-stage Slurm pipeline: manifest → merge → range resolution → merge → digitize

Scripts:
- slurm/toc_manifest_array.sbatch — reads ToC from each PDF
- slurm/resolve_ranges_array.sbatch — resolves page ranges for age tables
- slurm/merge_outputs.sbatch — combines task outputs
- slurm/digitize_table_pages_array.sbatch — renders page ranges as PNGs
- process_census_batch.sh — one-command orchestrator for all 5 stages

### Storage Rotation Strategy
Home dir: 95GB, so PDFs processed in batches:
1. Upload decade(s) fitting in ~80GB
2. Run process_census_batch.sh (all 5 stages)
3. Script auto-deletes source PDFs after digitization
4. PNGs accumulate (small: ~1-2GB per decade)
5. Upload next batch

### Batch Processing Results

**Batch: 1901** (69 PDFs, 44GB)
- Manifest job: 4429393 — all 69 COMPLETED
- Range job: 4432171 — all 69 COMPLETED
- Digitize job: completed — 0 failures
- Digitization queue: 366 page ranges
- Manual review queue: 16 items
- PNGs produced: 2,640
- PNG disk usage: 2.8GB

**Batch: 1891** (45 PDFs, 40GB)
- Digitization queue: 283 page ranges
- Manual review queue: 23 items (cumulative)
- PNGs produced: 2,757 (new) → 5,397 total

**Batch: 1871 + 1881 + 1941** (87 PDFs, 51GB)
- Manifest job: 4592415
- Range job: 4594843
- Digitize job: 4595066
- PDFs processed: 87
- Documents found: 87
- Pages digitized: 156 ranges → 7,157 total PNGs
- Manual review: 20 items

**Batch: 1911** (47 PDFs, 27GB)
- Processed via process_census_batch.sh

**Batch: 1921** (55 PDFs, 28GB)
- Processed via process_census_batch.sh

**Batch: 1931** (65 PDFs, 36GB)
- Processed via process_census_batch.sh

**Batch: 1951** (59 PDFs, 46GB)
- Processed via process_census_batch.sh

**Batch: 1961** (46 PDFs, 7.3GB)
- Manifest job: 4856551
- Range job: 4858155
- Digitize job: 4858315
- PDFs processed: 46
- Pages digitized: 25 ranges → 15,776 total PNGs
- Manual review: 3 items
- Note: 1961 data is mixed quality — includes US Census, linguistic surveys, district handbooks alongside Indian Census volumes

### Total Digitization Output
- **15,777 PNGs** across all decades
- **~70+ manual review items** saved to manual_review_log.txt (deferred)

---

## Phase 4: OCR Processing (Apr 9–ongoing)

### OCR Job History
1. Job 4438731: 1901 only, --array=0-19%6, 8h limit — 6 tasks TIMED OUT
2. Job 4598617: all accumulated PNGs, --array=0-49%6, 8h limit — running
3. Job 4869085: all PNGs, --array=0-99%6, 24h limit — running

### Progress Snapshot (Apr 11, latest)
| Metric | Count |
|--------|-------|
| Total PNGs | 15,777 |
| OCR results produced | 4,637 |
| Progress | 29% |

### Quality Audit (at 4,637 results)
| Category | Count | % |
|----------|-------|---|
| **Age tables found** | **1,262** | |
| — Validated (checks passed) | 835 | 66% of age tables |
| — Unvalidated (no checks) | 160 | 13% |
| — Failed (checks failed) | 229 | 18% |
| Non-age tables (filtered out) | 3,372 | 73% of all results |
| Plausibility issues | 15 | |
| Corrupt | 3 | |

### Quality Notes
- 77% of all results are accepted_high_confidence
- 14.5% rejected_constraints_failed (can be rerun with MoE)
- 8.6% accepted_with_risk (math checks out but took extra effort)
- 1,774 zero-check results = mostly non-age tables (education, birth-place, etc.)
- Negative values in "Variation" and "Percentage" columns are legitimate (population decline between censuses)
- 3 corrupt JSON files out of 4,637

### Cost
- Spent so far: ~$37 (at 4,637 results)
- Projected total: ~$130 for all 15,777 PNGs
- Per-image cost: ~$0.008

### Output Structure
- Results at: ~/census_toc/ocr_results/digitized_tables/{hash_tableslug}/{hash}_oneshot.json
- Not organized by year/province in directory structure
- Year, region, province ARE in each JSON's metadata — can reorganize later
- Manual review backlog at: ~/SudokuOCR/manual_review_log.txt

---

## Remaining Work

### Immediate
- [ ] OCR job completes (~71% remaining, est. 3-4 days)
- [ ] Final audit of all results

### Post-OCR
- [ ] Rerun 229+ constraint failures with MoE strategy (--rerun-failures)
- [ ] Review 160+ unvalidated age tables (no P=M+F structure)
- [ ] Reorganize results by year/province/table
- [ ] Address 70+ manual review items from ToC pipeline
- [ ] Filter out 3,372+ non-age table results
- [ ] Validate against known census bibliography (province × year coverage)

### Quality Improvements
- [ ] Investigate why 73% of digitized pages are non-age tables (ToC pipeline too broad)
- [ ] Add validation for age tables without M/F structure (proportional, religion-by-age, etc.)
- [ ] Cross-decade plausibility checks (population growth trends)
