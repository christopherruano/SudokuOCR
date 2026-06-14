#!/usr/bin/env python3
"""Categorize each imperfect Table VII by ROOT CAUSE so we fix the right way:
  A) half_page_split  – columns missing here exist on a SIBLING scan we already
     have (same doc, adjacent page/seq). FIX = assembly merge (free, no re-OCR).
  B) transposed       – age is in the COLUMN headers, caste/district in rows
     (C-III cross-tabs). FIX = orientation-aware handling (free).
  C) genuine_reocr    – a single wide scan that truly dropped a column / digit
     errors. FIX = cluster re-OCR with schema forced.
  D) partial_page     – legitimately few age bands on this page, nothing missing.
Writes imperfect_categorized.csv + prints the breakdown."""
import os, re, csv, sqlite3
from collections import defaultdict
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
HEX = re.compile(r"([0-9a-f]{40})")
DOC = re.compile(r"([0-9a-f]{16})_t(\d+)")
AGE_IN = re.compile(r"\b\d+\s*[-–]\s*\d+\b|\bunder\s*\d|\d+\s*and over", re.I)
CIV = {"U": re.compile(r"unmarr|single|never", re.I), "M": re.compile(r"marri", re.I),
       "W": re.compile(r"widow", re.I), "P": re.compile(r"popul|persons|total", re.I)}


def civ_of(s):
    s = s or ""
    if CIV["U"].search(s): return "U"
    if CIV["W"].search(s): return "W"
    if CIV["M"].search(s): return "M"
    if CIV["P"].search(s): return "P"
    return ""


def main():
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    imp = list(csv.DictReader(open(os.path.join(V1, "tableVII_imperfect.csv"))))

    # index: dochash -> [(seq, table_id, cols_present, ages)] for sibling lookups
    doc_tables = defaultdict(list)
    allrows = con.execute("SELECT table_id,source_image FROM tables").fetchall()
    tcols = {}
    for r in allrows:
        m = DOC.search((r["source_image"] or "") + " " + r["table_id"])
        if not m:
            continue
        doc_tables[m.group(1)].append((int(m.group(2)), r["table_id"]))

    def cols_and_ages(tid):
        cols, ages, age_in_col = set(), set(), 0
        for f in con.execute("SELECT column_group_std,sub_column_std,row_label_std FROM facts WHERE table_id=?", (tid,)):
            for fld in (f["column_group_std"], f["sub_column_std"], f["row_label_std"]):
                c = civ_of(fld)
                if c: cols.add(c)
            if AGE_IN.search(f["row_label_std"] or ""): ages.add(re.sub(r"\s+", "", (f["row_label_std"] or "").lower())[:12])
            if AGE_IN.search(f["column_group_std"] or ""): age_in_col += 1
        return cols, ages, age_in_col

    out = []
    cat = defaultdict(int)
    for r in imp:
        tid = r["table_id"]; issues = r["issues"]
        miss = set("".join(re.findall(r"missing_cols:([UMWP]+)", issues)))
        cols, ages, age_in_col = cols_and_ages(tid)
        nfacts = con.execute("SELECT COUNT(*) c FROM facts WHERE table_id=?", (tid,)).fetchone()["c"]
        category = ""
        detail = ""
        # B) transposed: lots of age tokens live in the column header
        if age_in_col >= max(4, 0.3 * max(1, nfacts)):
            category = "transposed"; detail = f"age_in_col={age_in_col}"
        # A) half-page split: a sibling table in the same doc has the columns we're missing
        if not category and miss:
            m = DOC.search((con.execute("SELECT source_image FROM tables WHERE table_id=?", (tid,)).fetchone()["source_image"] or "") + " " + tid)
            dh = m.group(1) if m else None; seq = int(m.group(2)) if m else None
            if dh:
                for sseq, stid in doc_tables.get(dh, []):
                    if stid == tid or abs(sseq - seq) > 2:
                        continue
                    scols, sages, _ = cols_and_ages(stid)
                    if miss <= scols and len(ages & sages) >= 3:
                        category = "half_page_split"; detail = f"sibling::{stid}::seq{sseq}::shares{len(ages & sages)}"
                        break
        # D) partial page only
        if not category and issues.startswith("few_age_bands") and not miss:
            category = "partial_page"; detail = f"ages={len(ages)}"
        # C) everything else needs a genuine re-OCR
        if not category:
            category = "genuine_reocr"; detail = f"cols={''.join(sorted(cols))} miss={''.join(sorted(miss))}"
        cat[category] += 1
        out.append({"table_id": tid, "category": category, "issues": issues, "detail": detail,
                    "title": r["title"], "is_assembled": tid.startswith("asm_")})

    with open(os.path.join(V1, "imperfect_categorized.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "category", "issues", "detail", "title", "is_assembled"])
        w.writeheader(); [w.writerow(x) for x in out]
    print(f"imperfect categorized: {len(out)}")
    for k in ("half_page_split", "transposed", "partial_page", "genuine_reocr"):
        print(f"  {k}: {cat[k]}")
    print("wrote imperfect_categorized.csv")


if __name__ == "__main__":
    main()
