# SudokuOCR

**Constraint-driven OCR that achieves 100% cell-level accuracy on historical census tables.**

SudokuOCR extracts numeric data from scanned pages of the Indian Census (1872–1941) by treating each table as a constraint satisfaction problem — like Sudoku. Vision LLMs read the digits; arithmetic constraints verify and correct them.

**492 cells across 3 ground-truth-verified tables. Zero errors.**

## Why "Sudoku"?

Census tables aren't grids of independent numbers. They're constraint systems:

```
                    Persons   Males   Females
    Age 0-1          33,217   14,919   18,298   ← P = M + F  ✓
    Age 1-2          30,401   14,637   15,764   ← P = M + F  ✓
    ...
    ──────────────────────────────────────────
    Total 0-5       169,500   81,224   88,276   ← = Σ rows above  ✓
                                                 ← AND P = M + F  ✓
```

A table with 240 cells and 4 column groups has **~400 constraint equations**. Every cell participates in 3–5 independent checks. If you know enough cells correctly, the constraints uniquely determine the rest — exactly like Sudoku.

### Five levels of constraints

| Level | Constraint | Example |
|-------|-----------|---------|
| L1 | Row balance | Persons = Males + Females |
| L2 | Vertical sums | Total 0–5 = (0–1) + (1–2) + ... + (4–5) |
| L3 | Cross-group | Population = Unmarried + Married + Widowed |
| L4 | Cross-section | All Communities = Hindus + Muslims + ... |
| L5 | Non-negativity | All values ≥ 0 |

## The Key Insight

AI is best at tasks where generation is hard but verification is easy.

Reading degraded digits from a 120-year-old scan is **hard**. Checking whether `Persons = Males + Females` is **trivial**. The constraint system transforms OCR from an unverifiable perception task into a constraint satisfaction problem — and CSPs have efficient solvers.

When multiple readings disagree, we don't just vote. We **reason deductively**: if fixing `53,818 → 53,318` makes all 400+ constraints pass, it's almost certainly correct. The minority reading was right; the constraint overrides the vote.

## Architecture

```
Phase 1: UNDERSTAND — Schema Discovery (1 API call)
  Image → "What kind of table is this?"
  → column groups, row labels, subtotal hierarchy, data type

Phase 2: EXTRACT — Tailored Extraction (1–4 API calls)
  Image + Schema → structured JSON
  Prompt is GENERATED from the schema (prevents hallucination)

Phase 3: VERIFY + REPAIR — Constraint Engine (0–1 API calls)
  Check L1–L5 constraints → blame scoring → targeted recheck
  Hierarchical CSP solver locks subtotals as anchors, solves top-down
```

Total: **2–6 API calls per table** at ~$0.08 each.

## Results

### Ground-truth accuracy (3 verified test cases)

| Table | Cells | Accuracy |
|-------|-------|----------|
| Travancore Eastern Division 1901 | 114 | 100.0% |
| Hyderabad State Summary 1901 | 240 | 100.0% |
| Coorg 1901 (proportional, per-1000) | 138 | 100.0% |
| **Total** | **492** | **100.0%** |

### Production scale (7,500+ tables processed)

At scale on Harvard's FASRC cluster using Slurm array jobs with multi-key API rotation:

- **4,553 age tables** extracted and constraint-validated
- **3,318 fully validated** (all constraint checks pass)
- **840 validated with warnings** (minor constraint failures)
- **395 flagged for review** (significant constraint failures)
- Self-verification via constraint pass rate — no ground truth needed

### How it compares

| Method | Cell Accuracy | Self-Verifying? | Cost/table |
|--------|-------------|-----------------|------------|
| Tesseract 5 | 70–85% | No | Free |
| Google Document AI | 90–95% | No | $0.065 |
| GPT-4o (single pass) | 92–96% | No | ~$0.05 |
| Gemini 2.5 Pro (single pass) | 93–97% | No | ~$0.04 |
| **SudokuOCR** | **100.0%** | **Yes** | **~$0.08** |

## The pipeline found errors in the ground truth

28 human transcription errors across 3 ground-truth Excel files. In every case, the constraint-verified pipeline output was correct and the human-entered data was wrong.

## Project Structure

```
pipeline.py              Core pipeline: API callers, extraction strategies, scoring
oneshot.py               Schema-aware extraction + constraint verification
ensemble.py              Voting, CSP solver, constraint propagation
schema_discovery.py      Automatic table structure detection
acceptance.py            Acceptance criteria and quality scoring
image_processing.py      Row detection, cropping, enhancement (PIL/numpy)
batch_production.py      Threaded batch runner with cost tracking and Slurm chunking
score_comprehensive.py   Scoring against ground truth

fasrc/                   Harvard FASRC cluster setup and Slurm submission
slurm/                   Slurm array job scripts for the ToC pipeline
process_census_batch.sh  One-command orchestrator: PDF → ToC → page ranges → PNGs

Data/                    Ground truth Excel files (one per province/state)
results/                 Sample JSON/CSV/XLSX outputs
figures/                 Presentation figures
```

## Quick Start

```bash
# Install dependencies
pip install google-genai openpyxl pillow numpy

# Set API key
export GEMINI_API_KEY=your_key_here

# Run on a single image
python pipeline.py

# Run batch production
python batch_production.py age_tables/ --results results/ --workers 4
```

## The Broader Principle

SudokuOCR demonstrates a general pattern: **AI + verifiable constraints = reliable AI**.

Use AI for the hard part (perception). Use math for the easy part (verification). Let verification guide correction. The constraints don't need to directly check whether a "3" looks like an "8" — they just need to be violated when the digit is wrong and satisfied when it's right.

This is the same principle behind test-driven code generation, proof-carrying code, and simulation-verified scientific predictions.
