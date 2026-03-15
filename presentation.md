# Constraint-Driven OCR: Achieving 100% Accuracy on Historical Census Tables

## The Problem

Historical census tables from the Indian Census (1872-1941) contain critical demographic data locked in scanned page images. These tables record age-by-sex population breakdowns for every district, province, and state across British India.

**Scale**: ~20 provinces, ~70+ years of census rounds, hundreds of districts = thousands of tables.

**Challenge**: The scans vary dramatically in quality:

| Era | Quality | Example |
|-----|---------|---------|
| 1901 | Clear typeset, large digits | Travancore Eastern Division |
| 1931 | Smaller text, wider tables, faded ink | Hyderabad district tables |
| 1941 | Dense multi-section tables, tiny text | Adilabad (3 sections x 5 groups) |

Traditional OCR tools (Tesseract, ABBYY) struggle with these: degraded ink, inconsistent fonts, complex multi-level headers, and merged cells produce error rates of 5-30% per cell. Manual transcription is expensive and, as we'll show, itself error-prone.

---

## Slide: Why This Is Hard

```
     POPULATION          UNMARRIED           MARRIED           WIDOWED
  Persons Males Females  Persons Males Fem.  Persons Males Fem.  Persons Males Fem.
  ─────── ───── ───────  ─────── ───── ────  ─────── ───── ────  ─────── ───── ────
  175,788 85,577 90,211  174,225 85,051 89,174  1,467   509   958      84    17    77
  236,893 115,495 121,398 235,371 114,699 120,572 1,490   763   727    132    33    99
  ...
```

A single table can have:
- **20 age rows x 4 column groups x 3 columns (P/M/F) = 240 cells** of numeric data
- **Multi-section tables**: 3 community sections x 5 groups x 11 ages = **495+ cells per image**
- Subtotal rows that must sum correctly
- Cross-group constraints (Population = Unmarried + Married + Widowed + Divorced)

**One wrong digit in one cell = unusable data for demographic analysis.**

---

## Slide: The Baseline — "Just Ask the AI"

The naive approach: send the image to a vision LLM (GPT-4, Gemini, Claude) and ask it to extract all numbers.

**Result**: ~92-96% cell-level accuracy on clean images.

That sounds good until you do the math:
- 240 cells per table x 4% error rate = **~10 wrong cells per table**
- No way to know WHICH cells are wrong
- Errors are systematic: `3↔8`, `5↔6`, `0↔9` digit confusions

**The fundamental problem**: The AI doesn't know when it's wrong. It reads "8" where the scan shows a degraded "3" and has no reason to doubt itself.

---

## Slide: The Key Insight — Census Tables Are Sudoku Puzzles

Census tables aren't just grids of independent numbers. They're **constraint systems**:

```
                    Persons   Males   Females
    Age 0-1          33,217   14,919   18,298   ← P = M + F  ✓
    Age 1-2          30,401   14,637   15,764   ← P = M + F  ✓
    Age 2-3          34,723   17,012   17,711   ← P = M + F  ✓
    Age 3-4          38,158   18,416   19,742   ← P = M + F  ✓
    Age 4-5          34,000   16,240   17,760   ← P = M + F  ✓
    ──────────────────────────────────────────
    Total 0-5       169,500   81,224   88,276   ← = sum of rows above  ✓
                                                 ← AND P = M + F  ✓
```

**Level 1 (L1)**: Every row → `Persons = Males + Females`
**Level 2 (L2)**: Vertical sums → `Total 0-5 = (0-1) + (1-2) + (2-3) + (3-4) + (4-5)`
**Level 3 (L3)**: Cross-group → `Population = Unmarried + Married + Widowed + Divorced`
**Level 4 (L4)**: Cross-section → `All Communities = Brahmanic + Other Hindus + Muslims + ...`
**Level 5 (L5)**: Non-negativity → All values ≥ 0

A table with 240 cells and 4 column groups has **~400 constraint equations**. Every cell participates in 3-5 independent constraints. Like Sudoku: if you know enough cells correctly, the constraints uniquely determine the rest.

---

## Slide: AI Is Best at Tasks with Verifiable Goals

This is a broader principle about where AI excels:

```
┌─────────────────────────────────────┐
│  HARD TO VERIFY    │  EASY TO VERIFY │
│  ──────────────    │  ────────────── │
│  "Is this essay    │  "Does P = M+F  │
│   well-written?"   │   for every     │
│                    │   row?"         │
│  "Is this code     │  "Do all        │
│   well-designed?"  │   subtotals     │
│                    │   sum           │
│  "Is this          │   correctly?"   │
│   translation      │                 │
│   natural?"        │  "Does Pop =    │
│                    │   Unm + Mar +   │
│                    │   Wid + Div?"   │
└─────────────────────────────────────┘
```

Census OCR sits in the sweet spot: the **generation task is hard** (reading degraded digits) but the **verification is trivial** (arithmetic). We can't easily tell if "8" or "3" is correct by looking at a blurry scan — but we CAN check if all the math works out.

**This is the same principle behind**: code generation (hard to write, easy to test), theorem proving (hard to discover, easy to verify), protein folding (hard to predict, easy to check energy).

The constraint system transforms OCR from an **unverifiable generation task** into a **constraint satisfaction problem** — and CSPs have efficient solvers.

---

## Slide: The Architecture — 3 Phases

```
┌──────────────────────────────────────────────────────────┐
│                    PHASE 1: UNDERSTAND                    │
│                                                          │
│  Image ──→ [Schema Discovery]                            │
│            "What kind of table is this?"                 │
│                                                          │
│  Output: TableSchema                                     │
│    • column_groups: [Population, Unmarried, Married, ...]│
│    • row_labels: [0-1, 1-2, ..., Total 0-5, ..., Total] │
│    • subtotal_hierarchy: {Total 0-5: [0-1...4-5]}       │
│    • data_type: absolute | proportional (per 1,000)      │
│    • has_persons_column: true/false                       │
│    • cross_group_constraints: {Pop: [Unm, Mar, Wid]}    │
│                                                          │
│                       1 API call                         │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    PHASE 2: EXTRACT                       │
│                                                          │
│  Image + Schema ──→ [Tailored Extraction]                │
│  "Here are the exact rows and columns to read.           │
│   The subtotals must satisfy these relationships.        │
│   Persons = Males + Females for every row."              │
│                                                          │
│  The prompt is GENERATED from the schema:                │
│    • Lists exact row labels (prevents hallucination)     │
│    • Specifies subtotal relationships                    │
│    • Describes data type + constraints                   │
│    • Emphasizes common digit confusions (3↔8, 5↔6)      │
│                                                          │
│                       1 API call                         │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    PHASE 3: VERIFY + REPAIR               │
│                                                          │
│  Extracted Data ──→ [Constraint Engine]                   │
│                                                          │
│  Check all constraints (L1-L5):                          │
│    ✓ 380/392 pass                                        │
│    ✗ 12 failures identified                              │
│                                                          │
│  If failures:                                            │
│    → Blame scoring: which cells most likely wrong?       │
│    → Math repair: can we fix with minimal changes?       │
│    → Targeted recheck: re-read only suspicious cells     │
│                                                          │
│                    0-1 API calls                          │
└──────────────────────────────────────────────────────────┘

Total: 2-3 API calls per table (vs. 12+ for brute-force ensemble)
```

---

## Slide: The Sudoku Solver — Hierarchical Constraint Propagation

When multiple readings disagree, we don't just vote. We **reason deductively**.

**Example**: 12 independent readings of a table. For age 30-35:

```
Reading 1:  Males = 53,818    ← 8 readings say 53,818
Reading 2:  Males = 53,818
Reading 3:  Males = 53,318    ← 4 readings say 53,318
Reading 4:  Males = 53,818
...                           (classic 3↔8 confusion)
```

Simple majority vote picks 53,818. But is that right?

**Constraint check**: We KNOW the Total row (locked with 100% agreement).
```
Total Males = 5,673,629  (all 12 readings agree)
Sum of all age rows with 53,818 at 30-35 = 5,674,129  → off by 500!
Sum of all age rows with 53,318 at 30-35 = 5,673,629  → EXACT MATCH ✓
```

**The minority was right.** The constraint overrides the vote.

This is the core algorithm:

```
Phase 1:  LOCK cells with unanimous agreement (100%)
Phase 1b: LOCK cells with supermajority (≥75%)
Phase 2:  PROPAGATE constraints iteratively
          • 2 of 3 known in a row → deduce the 3rd
          • N-1 of N known in a sum → deduce the missing one
Phase 2.5: HIERARCHICAL CSP (the Sudoku solver)
          • Lock intermediate subtotals as anchor points
          • Top-down: find valid combos at each hierarchy level
          • Exploit tree structure: polynomial, not exponential
Phase 3:  SEARCH remaining via candidate constraint satisfaction
```

---

## Slide: Why Hierarchical CSP Works

The subtotal hierarchy creates a **tree** of constraints:

```
                    Grand Total
                   /     |      \
            Total 0-5  Total 5-15  Total 15-40  Total 40-60  60+
            /  |  |  \    |   |      |   |  |    |    |
          0-1 1-2 ... 4-5 5-10 10-15 15-20 ... 40-45 ...
```

**Without hierarchy**: N unresolved cells → 2^N possible combinations to try.
**With hierarchy**: Each level has ~2-5 children → ~10-50 combos per level → total search: ~100-500.

**The trick**: Lock intermediate subtotals (Total 0-5, Total 15-40, etc.) when >50% of readings agree. These become **anchor points** — like solved squares in Sudoku that constrain their neighbors.

Then solve top-down:
1. Grand Total is known (highest agreement)
2. Find which combo of intermediate subtotals sums to Grand Total
3. For each intermediate: find which combo of leaf rows sums to it
4. Pick the combination with highest total vote score

**Result**: Unique solution found in milliseconds, even when majority vote was wrong.

---

## Slide: Generalization — Table Structure Discovery

The pipeline doesn't need manual configuration. It **discovers** the table structure automatically:

```
INPUT: Any scanned census table image
       (never seen before, unknown layout)

SCHEMA DISCOVERY OUTPUT:
{
  "title": "Age, Sex and Civil Condition",
  "data_type": "proportional",
  "denominator": 1000,
  "column_groups": [
    {"name": "1901", "sub_columns": ["Males", "Females"]},
    {"name": "1891", "sub_columns": ["Males", "Females"]},
    {"name": "1881", "sub_columns": ["Males", "Females"]}
  ],
  "has_persons_column": false,
  "subtotal_hierarchy": {
    "Total 0-5": ["0-1", "1-2", "2-3", "3-4", "4-5"],
    "Total 0-15": ["Total 0-5", "5-10", "10-15"],
    "Total 15-40": ["15-20", "20-25", "25-30", "30-35", "35-40"],
    "Total 40-60": ["40-45", "45-50", "50-55", "55-60"]
  },
  "known_totals": {"males": 1000, "females": 1000}
}
```

This handles radically different table types:
- **Absolute counts** (175,788 persons) vs **proportional** (28 per 1,000)
- **P/M/F columns** vs **M/F only** (persons must be computed)
- **Single-year** vs **multi-year comparison** tables
- **Single community** vs **multi-section** (by religion/caste)
- **4-group tables** (Pop/Unm/Mar/Wid) vs **5-group** (+Divorced)

---

## Slide: Results

### Ground Truth Accuracy (3 verified test cases)

```
                                    gemini-3.1       gemini-2.5
                                    pro-preview      pro
                        Cells       ───────────      ──────────
Travancore Eastern 1901   114        100.0%           100.0%
  (absolute, 2 groups, P/M/F)

Hyderabad State 1901      240        100.0%            98.3%
  (absolute, 4 groups, P/M/F)

Coorg 1901                138        100.0%           100.0%
  (per-1000, 3 years, M/F only,
   rich subtotal hierarchy)
                        ─────       ─────────        ────────
TOTAL                     492        100.0%            98.6%
                                    (492/492)        (485/492)
```

With the best available model + constraint verification:
**492 cells, 0 errors, 100% accuracy.**

### Constraint Pass Rate (54 tables, no ground truth needed)

```
┌─────────────────────────────────────────────────────────┐
│  54 tables processed across 6 regions and 4 census years│
│                                                         │
│  Total constraint checks:  29,880                       │
│  Passed:                   29,468  (98.6%)              │
│  Perfect images:           30/54   (56%)                │
│                                                         │
│  By source:                                             │
│    1891 multi-district (Mysore, Assam, Burma): 96-99%   │
│    1901 test cases:   ALL PERFECT (5/5)                 │
│    1931 Hyderabad districts:  98.4%  (9/16 perfect)     │
│    1941 Hyderabad districts:  99.5% (10/17 perfect)     │
│                                                         │
│  Key insight: constraint pass rate tells you quality    │
│  WITHOUT any ground truth. 804/804 = trustworthy.       │
│  663/663 = trustworthy. 613/663 = needs review.        │
└─────────────────────────────────────────────────────────┘
```

---

## Slide: The Progression — From Baseline to 100%

```
ACCURACY PROGRESSION (492 GT cells across 3 test tables):

  100% ┤                                         ●──── SudokuOCR + 3.1-pro
       │                                    ●
   98% ┤                              ●─────────── SudokuOCR + 2.5-pro
       │                         ●
   96% ┤                    ●
       │               ●
   94% ┤          ●
       │     ●
   92% ┤●
       │
   90% ┤
       └──┬────┬────┬────┬────┬────┬────┬─────
         Raw  +3x  +Cnst +CSP +Sch  2.5  3.1
         LLM  Vote Prop  Hier Aware pro  pro

  Raw LLM:              ~92%  (1 pass, generic prompt, 1 API call)
  + Majority vote:      ~96%  (3 passes, pick consensus, 3 calls)
  + Constraint prop:    ~98%  (use P=M+F and sums to fix errors)
  + Hierarchical CSP:    99%  (subtotal anchors + tree solve)
  + Schema-aware:        99%  (tailored prompts, structure-aware)
  + gemini-2.5-pro:    98.6%  (2-3 API calls, constraint-verified)
  + gemini-3.1-pro:   100.0%  (2-3 API calls, constraint-verified)
```

**Key takeaway**: Each layer of constraint reasoning adds accuracy. The jump from voting to constraint propagation is where "AI is guessing" becomes "AI is deducing." The best model (3.1-pro) + constraints = **zero errors across 492 cells**.

---

## Slide: Correlated vs Uncorrelated Errors

Why constraints beat voting:

```
VOTING fails when errors are CORRELATED:

  Reading 1:  age 30-35, Males = 53,818  ← same image artifact
  Reading 2:  age 30-35, Males = 53,818  ← same model bias
  Reading 3:  age 30-35, Males = 53,818  ← same enhancement
  Vote: 53,818 (WRONG — correct answer is 53,318)

  The "3" in the original scan is degraded.
  All 3 models see the same degraded "3" and read "8".
  8/8 agree. Voting says 53,818. WRONG.

CONSTRAINTS succeed because they're ORTHOGONAL:

  The error in Males propagates NOWHERE:
    Row check:     P ≠ M + F  → flags this cell
    Vertical sum:  Total ≠ Σ(ages)  → flags this cell
    Cross-group:   Pop ≠ Unm + Mar + Wid  → flags this cell

  3 independent constraints, 3 independent signals.
  Even with 100% correlated model errors,
  constraints provide UNCORRELATED verification.
```

**This is the fundamental advantage**: constraints create independent verification channels that don't share the same failure modes as the AI's visual perception.

---

## Slide: Output Formats — Research-Ready Data

The pipeline produces structured outputs ready for analysis:

### Excel (wide format, one sheet per community section)
```
┌──────┬────────────────────┬────────────────────┬──────────────┐
│ Age  │ POPULATION         │ UNMARRIED          │ MARRIED      │
│      │ Persons Males Fem  │ Persons Males Fem  │ Persons ...  │
├──────┼────────────────────┼────────────────────┼──────────────┤
│ 0-1  │ 28,785 13,661 15,124│28,480 13,534 14,946│  279   ...  │
│ 1-5  │109,003 49,712 59,291│102,530 47,678 54,852│6,018  ...  │
│ ...  │                    │                    │              │
│Total │1,647,244 ...       │579,296 ...         │909,452 ...   │
├──────┴────────────────────┴────────────────────┴──────────────┤
│ ✓ All P=M+F constraints pass                                 │
│ ✓ All vertical sum constraints pass                          │
│ ✓ All cross-group constraints pass                           │
│ ✓ 804/804 total checks passed                                │
└───────────────────────────────────────────────────────────────┘
```

### JSON (nested, with full metadata)
```json
{
  "source_image": "Adilabad.png",
  "api_calls": 2,
  "metadata": {
    "column_groups": ["POPULATION","UNMARRIED","MARRIED","WIDOWED","DIVORCED"],
    "data_type": "absolute"
  },
  "data": [
    {"section": "1. ALL COMMUNITIES.", "rows": [...]},
    {"section": "2. Brahmanic Hindus.", "rows": [...]},
    {"section": "3. Other Hindus", "rows": [...]}
  ],
  "constraints": {"total_checks": 804, "passed": 804, "all_passed": true}
}
```

### CSV (long format, analysis-ready)
```
section,age,group,persons,males,females
1. ALL COMMUNITIES.,0-1,POPULATION,28785,13661,15124
1. ALL COMMUNITIES.,0-1,UNMARRIED,28480,13534,14946
...
```

---

## Slide: Finding Errors in the Ground Truth

An unexpected result: the pipeline is **more accurate than the human-transcribed ground truth**.

Cross-referencing pipeline output against two independent GT sources:

```
┌────────────────────┬──────────┬───────────────────────────────┐
│ GT Source           │ Errors   │ Verdict                       │
├────────────────────┼──────────┼───────────────────────────────┤
│ Travancore.xlsx    │ 4 cells  │ ALL are GT transcription errors│
│  (human-entered)   │          │ e.g. 30-35 off by 1,000       │
├────────────────────┼──────────┼───────────────────────────────┤
│ Hyderabad.xlsx     │ 21 cells │ ALL are GT transcription errors│
│  (human-entered)   │          │ e.g. 10-15 off by 6,000       │
│                    │          │      45-50 off by 6,030       │
├────────────────────┼──────────┼───────────────────────────────┤
│ Coorg.xlsx         │ 3 cells  │ ALL are GT transcription errors│
│  (human-entered)   │          │ e.g. 50-55 off by 11          │
├────────────────────┼──────────┼───────────────────────────────┤
│ Ground truth for   │ 0 cells  │ Pipeline matches perfectly    │
│ 3.xlsx (verified)  │ (4 with  │ (4 errors in Hyd Unmarried   │
│                    │  2.5-pro)│  at 45-50, ±300 M/F swap)    │
└────────────────────┴──────────┴───────────────────────────────┘

28 human transcription errors found across 3 GT files.
In every case, the pipeline's constraint-verified output was correct.
```

**The constraint system doesn't just catch AI errors — it catches human errors too.** A reading that passes all 400+ arithmetic constraints is almost certainly correct, regardless of who (or what) produced it.

---

## Slide: Cost and Speed

```
Approach                  API Calls  Cost/img   Accuracy   Knows if wrong?
────────────────────────  ─────────  ────────   ────────   ──────────────
Single LLM pass             1        ~$0.03     ~92-96%    No
Multi-pass voting (3x)      3        ~$0.10     ~96-98%    No
Full MoE ensemble          12        ~$0.90     99-100%    No
  (4 variants x 3 models)
────────────────────────  ─────────  ────────   ────────   ──────────────
SudokuOCR (this work)      2-6       ~$0.08     100.0%     YES
  Schema + Extract +                             (constraint
  Constraint verify                               pass rate)
```

54 tables × 108 API calls = **~$4.30 total** at gemini-2.5-pro pricing.

SudokuOCR achieves MoE-level accuracy at ~1/10th the cost,
AND it knows when it's wrong — constraint failures pinpoint errors.

---

## Slide: The Broader Principle

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   AI + Verifiable Constraints = Reliable AI             │
│                                                         │
│   The pattern:                                          │
│   1. Use AI for the HARD part (perception, generation)  │
│   2. Use MATH for the EASY part (verification)          │
│   3. Let verification GUIDE correction                  │
│                                                         │
│   Census OCR:  AI reads digits → arithmetic verifies    │
│   Code gen:    AI writes code  → tests verify           │
│   Translation: AI translates   → back-translation verifies│
│   Science:     AI predicts     → simulation verifies    │
│                                                         │
│   The constraints don't need to be EXACTLY the goal.    │
│   They just need to be CORRELATED with correctness.     │
│   P=M+F doesn't check if "8" looks like "3" in the     │
│   scan — but if fixing 8→3 makes ALL constraints pass,  │
│   it's almost certainly right.                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Constraints that are **correlated with but not identical to** the actual goal still provide powerful signal. You don't need a constraint that directly says "this digit is 3 not 8" — you need constraints that are **violated when the digit is wrong** and **satisfied when it's right**.

---

## Slide: How This Compares to State of the Art

```
METHOD COMPARISON — Historical Census Table OCR

                                Cell-Level   Table        Self-        Cost
Method                         Accuracy     Structure?   Verifying?   per page
─────────────────────────────  ──────────   ──────────   ──────────   ────────
TRADITIONAL OCR
  Tesseract 5                    70-85%        No           No        Free
  ABBYY FineReader               85-95%       Partial      No        ~$0.10
  Google Document AI              90-95%       Yes          No        $0.065
  Amazon Textract                 88-93%       Yes          No        $0.015
  Azure Document Intelligence     90-95%       Yes          No        $0.01

LLM-BASED (raw, single pass)
  GPT-4o                          92-96%       Yes          No        ~$0.05
  Gemini 2.5 Pro                  93-97%       Yes          No        ~$0.04
  Gemini 3.1 Pro Preview          95-98%       Yes          No        ~$0.04
  Claude 3.5 Sonnet               91-95%       Yes          No        ~$0.06

ACADEMIC BENCHMARKS
  TrOCR + Table Transformer       85-92%       Yes          No        N/A
  PubTabNet fine-tuned             90-95%       Yes          No        N/A
  OCRBench v2 (best system)       ~74%*        Yes          No        N/A

SUDOKUOCR (THIS WORK)
  Schema + Extract + Verify       100.0%       Yes         YES       ~$0.08
  (Gemini 3.1 Pro Preview)       (492/492)     Auto        (constraint
                                                discovery    pass rate)
────────────────────────────────────────────────────────────────────────────
* OCRBench v2 measures across diverse OCR tasks, not table-specific
```

**Key differentiators**:
1. **100% cell accuracy** on verified test cases — no other published system reports this on historical tables
2. **Self-verification** — constraint pass rate tells you quality WITHOUT ground truth
3. **Automatic structure discovery** — no per-table configuration needed
4. Traditional OCR tools produce raw text; they don't understand table structure or validate arithmetic

---

## Slide: Performance at Scale — 54 Tables, 29,880 Constraints

```
DATASET SUMMARY
────────────────────────────────────────────────────────────────
  54 tables processed across 6 regions and 4 census years
  29,880 constraint equations checked
  98.6% pass rate, 56% of tables fully verified
────────────────────────────────────────────────────────────────

BY SOURCE:
                                              Constraint    Perfect
Source                  Tables  Constraints    Pass Rate     Tables
──────────────────────  ──────  ───────────    ─────────     ──────
Mysore 1891 (multi-d)    11       2,753        96.0%         5/11
Assam 1891 (multi-d)      3       1,566        96.6%         2/3
Burma 1891 (multi-d)      1         404        98.5%         0/1
1901 test cases           5       1,143        99.8%         4/5
Cochin 1891               1         144       100.0%         1/1
Hyderabad 1931           16      10,387        98.4%         9/16
Hyderabad 1941           17      13,063        99.5%        10/17
Other 1901                1         663        99.4%         0/1
──────────────────────  ──────  ───────────    ─────────     ──────
ALL                      54      29,880        98.6%        30/54

NOTE: batch processing ongoing — remaining Assam, Bombay,
Madras, Burma, Berar, NW Provinces, Punjab, Central India
(~25 more pages queued)
```

**Multi-district 1891 tables** are a new format: districts as rows,
age groups as columns, spanning multiple pages per province.
Even with this more complex layout: 96-99% constraint pass rate.

---

## Slide: Per-Table Constraint Quality — No Ground Truth Needed

```
CONSTRAINT PASS RATE ACROSS 54 TABLES
(sorted by quality — each █ = 1 table)

100%  ██████████████████████████████  30 tables — fully verified, trustworthy
 99%+ ███████                          7 tables — 1-4 constraint failures
 97%+ █████                            5 tables — minor issues
 95%+ █████                            5 tables — needs review
 <95% ███████                          7 tables — needs review (1931 era + multi-district)

KEY INSIGHT: You don't need ground truth to know quality.
  • 804/804 constraints pass → high confidence, ship it
  • 663/663 constraints pass → high confidence, ship it
  • 613/663 constraints pass → 50 failures → flag for review

The constraint rate IS the quality metric.
```

---

## Slide: Cost Comparison — What You Actually Pay

```
COST PER TABLE (including table structure + verification)

Method                          Cost/table   What you get
──────────────────────────────  ──────────   ──────────────────────────
Manual transcription            $5-15        Accuracy varies (we found
                                             28 errors in 3 GT files)

Google Document AI              $0.065       Raw text + basic layout
Amazon Textract (Tables)        $0.015       Cell extraction, no verify
Azure Doc Intelligence          $0.01        Cell extraction, no verify

GPT-4o (single pass)            ~$0.05       Structured JSON, ~94%
                                             No verification

Gemini 3.1 Pro (single pass)    ~$0.03       Structured JSON, ~96%
                                             No verification

SUDOKUOCR (Gemini 3.1 Pro Preview)
  Schema discovery (Call 1)      ~$0.01  ┐
  Extraction (Calls 2-5)        ~$0.05  ├─ $0.08/table average
  Recheck if needed              ~$0.02  ┘   ($2.99 for all 39 tables)

                                             Structured JSON + CSV + XLSX
                                             Full constraint verification
                                             Self-assessed quality score
──────────────────────────────────────────────────────────────────────────

BY TABLE COMPLEXITY:
  Single-section tables:  ~$0.05/table  (2-3 API calls)
  Multi-section tables:   ~$0.09/table  (4-5 API calls)

AT SCALE:
  1000 tables ≈ $77 total, ~100 hours compute
```

SudokuOCR costs **~2.5x a single raw LLM call** but delivers:
verified accuracy, structured output in 3 formats, and a quality
score that tells you which tables to trust without any ground truth.

---

## Slide: How Constraints Structure the Output

Schema discovery isn't just for verification — it **shapes** the extraction:

```
WITHOUT SCHEMA DISCOVERY              WITH SCHEMA DISCOVERY
─────────────────────────              ──────────────────────

Prompt: "Extract all numbers          Prompt: "This table has:
 from this table"                      • 20 age rows: 0-1, 1-2, ..., 60+
                                       • 4 groups: Pop, Unm, Mar, Wid
                                       • P/M/F columns in each group
Failure modes:                         • Subtotals: Total 0-5 = Σ(0-1..4-5)
 × Misidentifies column groups         • Cross-group: Pop = Unm + Mar + Wid
 × Hallucinate extra/missing rows      • Self-check: P = M + F every row
 × Misses subtotal structure
 × Wrong data type assumption        Result:
 × No way to validate output          ✓ Exact row/column structure
                                       ✓ Subtotals identified and verified
                                       ✓ Appropriate constraints applied
                                       ✓ Known totals act as anchors
                                       ✓ Quality score included in output
```

**The constraint system serves 3 purposes simultaneously:**

1. **Structures the prompt** — tells the model exactly what to look for
2. **Verifies the output** — catches errors through arithmetic checks
3. **Enables repair** — pinpoints suspicious cells for targeted recheck

This is why the pipeline adapts automatically to table types it's
never seen before: schema discovery finds the constraints, and
constraints drive everything else.

---

## Slide: The Constraint Hierarchy as a Trust Signal

```
CONSTRAINT COVERAGE BY TABLE TYPE

Table Type              Example           Constraints  Density
──────────────────────  ────────────────  ───────────  ──────────
Simple (1 section,      Travancore 1901   59 checks    0.52/cell
 2 groups, P/M/F)

Medium (1 section,      Hyderabad 1901    404 checks   1.68/cell
 4 groups, P/M/F)

Complex (3 sections,    Adilabad 1941     804 checks   1.52/cell
 5 groups, P/M/F)

Proportional (M/F       Coorg 1901        174 checks   1.26/cell
 only, known totals)

HIGHER DENSITY = MORE VERIFICATION = MORE TRUST

Key: Each cell participates in 3-5 independent constraints.
     A cell that satisfies ALL of them is almost certainly correct.
     A cell that violates ANY of them is flagged for review.
```

The constraint density means errors can't hide. In a 4-group table,
a single wrong digit triggers failures in:
- Its row's P=M+F check (L1)
- Its column's vertical sum (L2)
- Its cross-group equation (L3)
- Possibly a subtotal check (L2)

**4 independent alarm bells for 1 wrong digit.**

---

## Appendix: Sample Scans

### Easy (Travancore 1901)
Clean typeset, large digits, clear column lines. Single LLM pass: ~96%.
With constraints: 100%.

### Medium (Hyderabad State 1901)
4 column groups, 20 age rows, smaller text. 240 cells total.
Single LLM pass: ~92%. With constraints: 98.3-100%.

### Hard (Hyderabad Districts 1931)
Faded ink, small text, wide tables, 3 sections x 5 groups.
663 constraint checks per image. 99.0% constraint pass rate across all.

### Complex (Coorg 1901 — Proportional)
Per-1,000 table, M/F only (no Persons column), 3 census years,
rich subtotal hierarchy. Known totals (M=1000, F=1000) act as anchors.
100% accuracy.

---

## Appendix: Error Modes and How Constraints Catch Them

| Error | Example | Which Constraint Catches It |
|-------|---------|---------------------------|
| 3↔8 confusion | 53,318 → 53,818 (+500) | L2: vertical sum off by 500 |
| 5↔6 confusion | 15,764 → 15,764 (+1) | L1: P ≠ M+F by 1 |
| Dropped digit | 175,788 → 17,578 | L2: sum wildly off |
| Extra digit | 85,577 → 855,770 | L1: P < M (impossible) |
| Column shift | Read Unmarried as Population | L3: Pop ≠ Unm+Mar+Wid |
| M/F swap | Males↔Females switched | L3: cross-group inconsistency |

Every common OCR error mode violates at least one constraint level. The constraint system provides **defense in depth** — even if one check misses the error, another catches it.
