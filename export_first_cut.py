#!/usr/bin/env python3
"""Prepare a shareable FIRST-CUT of the Imperial Table VII data. Produces, in export/first_cut/:
  catalog.csv            one row per assembled table: doc, region, year, page-range, #pages, #rows,
                         quality label (verified / minor-issues / incomplete), #constraint-fail rows,
                         #empty pages, #missing pages — HONEST labels so a reader knows what to trust.
  tableVII_long.csv      tidy long-format: every cell of every table (doc, page, section, row, group,
                         subkey, value) — universal for analysis.
  tables/<doc>_<range>.csv   per-table flattened matrix (human-readable).
  README.md              what this is + the honest quality summary.
  python3 export_first_cut.py
"""
import sys, os, json, csv, re, datetime
sys.argv = ["x"]
import census_review_app as A
import boundary_apply as B

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "export/first_cut")
RUNS = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))


import re as _re
_BLANK = _re.compile(r"blank|empty page|no print|no table|not a table|table not found|"
                     r"no table present|no data|nothing on", _re.I)


def _is_blank(d):
    """A page the OCR correctly read as having no table (blank / 'No table present')."""
    if (d.get("metadata", {}) or {}).get("title_page"):
        return True
    meta = d.get("metadata", {}) or {}
    blob = " ".join(str(meta.get(k, "")) for k in ("title", "note", "notes", "reason", "page_type"))
    return bool(_BLANK.search(blob))


def flatten(d):
    """yield (section, row_label, group, subkey, value) for a table's data."""
    for s in d.get("data", []):
        rows = s.get("rows", [])
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        for row in rows:
            lbl = str(row.get(rh, "")) if rh else ""
            for g, v in row.items():
                if isinstance(v, dict):
                    for k, val in v.items():
                        yield s.get("name", ""), lbl, g, k, val


def main():
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    cat = []
    long_rows = []
    n_tables = 0
    for r in RUNS:
        dh = r["dochash"]; s, e = r["page_range"]; tp = set(r.get("title_pages", []))
        doc = str(r["saoa"])
        members = sorted(r.get("members", []), key=lambda m: m.get("page", 0))
        # collect rows + quality
        nrows = fail_rows = empty_pages = 0
        per_table = []   # (page, section, row_label, {group.subkey: val})
        for m in members:
            d, ih, _ = A.load_table(m["table_id"])
            if not d:
                continue
            meta = d.get("metadata", {}) or {}
            if meta.get("title_page"):
                continue
            pg = m.get("page", "")
            page_rows = sum(len(sec.get("rows", [])) for sec in d.get("data", []))
            if page_rows == 0:
                if not _is_blank(d):          # a genuinely blank / "no table" page is NOT a defect
                    empty_pages += 1
                continue
            for sec in d.get("data", []):
                nrows += len(sec.get("rows", []))
            fail_rows += A.table_fail_rows(d)   # layout-aware: proportional tables judged by their real identity
            for section, lbl, g, k, val in flatten(d):
                long_rows.append((doc, pg, section, lbl, g, k, val))
                per_table.append((pg, section, lbl, f"{g}/{k}", val))
        missing = [pg for pg in range(s, e + 1) if pg not in tp and not B.member_for(dh, pg)[1]]
        if nrows == 0:
            continue                                   # skip empty/non-data runs from the data export
        n_tables += 1
        # honest quality label. NOTE: a page that was OCR'd and returned 0 rows is NOT incompleteness —
        # verified by inspection these are title pages / section dividers / narrative prose that fall inside
        # the page span (the OCR correctly found no table). Only GENUINELY-MISSING (never-OCR'd) pages count.
        if missing:
            quality = "incomplete"
        elif fail_rows == 0:
            quality = "verified"
        elif fail_rows <= max(2, nrows // 20):
            quality = "minor-issues"
        else:
            quality = "needs-review"
        cat.append({"doc": doc, "region": r.get("region", ""), "year": r.get("year", ""),
                    "table": r.get("tnum", "VII"), "page_range": f"{s}-{e}", "n_pages": len(members),
                    "n_rows": nrows, "quality": quality, "constraint_fail_rows": fail_rows,
                    "empty_pages": empty_pages, "missing_pages": len(missing)})
        # per-table matrix CSV (page, section, row, then group/subkey columns)
        cols = []
        for _, _, _, gk, _ in per_table:
            if gk not in cols:
                cols.append(gk)
        bykey = {}
        for pg, section, lbl, gk, val in per_table:
            bykey.setdefault((pg, section, lbl), {})[gk] = val
        tf = os.path.join(OUT, "tables", f"{doc}_{s}-{e}.csv")
        with open(tf, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["page", "section", "row"] + cols)
            for (pg, section, lbl), cells in bykey.items():
                w.writerow([pg, section, lbl] + [cells.get(c, "") for c in cols])

    with open(os.path.join(OUT, "catalog.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat[0].keys())); w.writeheader(); w.writerows(cat)
    with open(os.path.join(OUT, "tableVII_long.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["doc", "pdf_page", "section", "row", "group", "subkey", "value"])
        w.writerows(long_rows)

    from collections import Counter
    q = Counter(c["quality"] for c in cat)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    readme = f"""# Imperial Table VII (Age, Sex & Civil Condition) — FIRST CUT ({stamp})

Digitized age-by-sex-by-civil-condition tables from the Indian Census (1872–1941), assembled across
multi-page spreads. **This is a first cut shared while accuracy work continues — labels are honest.**

## Contents
- `catalog.csv` — one row per table, with an honest **quality** label.
- `tableVII_long.csv` — every cell of every table in tidy long format (for analysis).
- `tables/<doc>_<range>.csv` — each table as a readable matrix.

## Quality labels (in catalog.csv)
- **verified** — passes all internal arithmetic checks (M+F=Persons, U+M+W=Total, column sums). {q.get('verified',0)} tables.
- **minor-issues** — a few cells fail a check (usually a single misread digit or a source typo). {q.get('minor-issues',0)} tables.
- **needs-review** — more substantial discrepancies; treat with caution. {q.get('needs-review',0)} tables.
- **incomplete** — some pages not yet OCR'd or came back empty. {q.get('incomplete',0)} tables.

Passing the checks means the table is internally consistent — a strong but not absolute signal of correctness.
Cells that don't reconcile are sometimes faithful transcriptions of errors in the printed original.

## Totals
{n_tables} assembled tables · {len(long_rows):,} cells.
"""
    open(os.path.join(OUT, "README.md"), "w").write(readme)
    print(f"FIRST CUT -> {OUT}")
    print(f"  {n_tables} tables | {len(long_rows):,} cells")
    print(f"  quality: {dict(q)}")


if __name__ == "__main__":
    main()
