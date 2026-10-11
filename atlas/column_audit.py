#!/usr/bin/env python3
"""Column- and row-level misread detector for digitized census tables.

Misreads concentrate in columns: a mis-parsed header shifts, invents or copies whole columns.
Each column (section x column group x sub-column) and each row is tested for:

  ungrounded   its numbers are (almost) not printed on the page, relative to the table's other
               columns: invented columns, columns taken from another page, synthesized totals
  frozen       one non-zero value repeated down the column (>= 80% of cells)
  duplicate    identical to another column in >= 90% of rows (copy artifact)
  dead         empty while the table's other columns are filled
and rows for frozen / dead / duplicate. Zero is never evidence of freezing (all-zero columns,
e.g. widowed aged 0-1, are genuine).

Page numbers come from atlas/out/page_ocr_numbers.jsonl (written by scan_grounding.py).
Output atlas/out/column_audit.csv (one row per table) and column_audit_detail.csv.
"""
import os, re, sys, csv, json, sqlite3, hashlib, glob, collections, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, ".."); SITE = os.path.join(ROOT, "census_site")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
from grounding import norm_value


BASES = {100.0, 1000.0, 10000.0, 100000.0}      # proportional-table bases repeat legitimately
TOTALISH = re.compile(r"(?i)total|population|all ages|all religions|persons|both sexes")


def digits(t):
    """digit string without separators: census prints decimals with a raised dot (2425·02)
    that the OCR drops or misreads, so 2425.02 and '242502' compare equal"""
    return re.sub(r"\D", "", str(t)).lstrip("0")


def sig(v):
    n = digits(norm_value(v)) if v is not None else ""
    return n if len(n) >= 2 else None


class Pool:
    """numbers OCR'd from the page; long tokens are run-together neighbouring cells, so a
    value of >= 3 digits also matches inside a token >= 7 digits long"""
    def __init__(self, page_nums):
        self.exact = {digits(k) for k in page_nums} - {""}
        self.long = [t for t in self.exact if len(t) >= 7]

    def __contains__(self, v):
        return v in self.exact or (len(v) >= 3 and any(v in t for t in self.long))

    def __bool__(self):
        return bool(self.exact)


def audit_table(facts, page_nums):
    cols = collections.defaultdict(dict)       # col -> {row: value}
    rows = collections.defaultdict(dict)       # row -> {col: value}
    for f in facts:
        c = (f["section"] or "", f["column_group_std"] or "", f["sub_column_std"] or "")
        r = (f["section"] or "", f["row_label_std"] or "")
        cols[c][r] = f["value"]
        rows[r][c] = f["value"]
    if len(cols) < 2:
        return None, []
    pool = Pool(page_nums)
    detail, flags = [], collections.Counter()
    cg = {}
    for c, cells in cols.items():
        vals = [sig(v) for v in cells.values() if v is not None]
        vals = [v for v in vals if v]
        cg[c] = (sum(v in pool for v in vals) / len(vals)) if len(vals) >= 3 and pool else None
    ref = sorted(g for g in cg.values() if g is not None)
    med = st.median(ref) if ref else None
    # reference = how well the best-read columns are grounded (80th percentile), so a table whose
    # columns are mostly invented is still judged against the columns that are real
    top = ref[int(0.8 * (len(ref) - 1))] if ref else None
    allv = [sig(v) for cells in cols.values() for v in cells.values()]
    allv = [v for v in allv if v]
    n_vals = len(allv)
    recall = round(sum(v in pool for v in allv) / n_vals, 3) if n_vals and pool else None
    # the page OCR must itself be usable: enough multi-digit numbers relative to the table
    n_ocr = sum(c for k, c in page_nums.items() if len(digits(k)) >= 2)
    page_ok = n_ocr >= 0.4 * n_vals if n_vals else False
    n_rows = len(rows)
    for c, cells in cols.items():
        nz = [v for v in cells.values() if v not in (None, 0) and v not in BASES and abs(v) >= 10]
        reasons = []
        if page_ok and top is not None and cg[c] is not None and top >= 0.45 and cg[c] <= min(0.15, top * 0.3):
            reasons.append(f"ungrounded ({cg[c]:.2f} vs best columns {top:.2f})")
        if len(nz) >= 5:
            mode, k = collections.Counter(nz).most_common(1)[0]
            if k / len(nz) >= 0.8:
                reasons.append(f"frozen (value {norm_value(mode)} in {k}/{len(nz)} cells)")
        if not cells or all(v is None for v in cells.values()):
            if n_rows >= 3:
                reasons.append("dead")
        for reason in reasons:
            flags[reason.split(" ")[0]] += 1
            detail.append(["col", " | ".join(x for x in c if x), reason])
    keys = list(cols)           # facts are in reading order, so neighbours are adjacent columns
    for i, a in enumerate(keys):
        for b in keys[i + 1:i + 3]:
            same_group = a[:2] == b[:2]
            label = " ".join(a + b)
            sexcopy = same_group and {a[2].lower(), b[2].lower()} & {"persons", "total"} and {a[2].lower(), b[2].lower()} & {"males", "females"}
            if TOTALISH.search(label) and not sexcopy:
                continue
            common = [r for r in cols[a] if r in cols[b] and cols[a][r] not in (None, 0) and cols[b][r] not in (None, 0)
                      and abs(cols[a][r]) >= 10 and cols[a][r] not in BASES]
            if len(common) >= 5 and sum(cols[a][r] == cols[b][r] for r in common) / len(common) >= 0.9:
                flags["duplicate"] += 1
                detail.append(["col", " | ".join(x for x in b if x), f"duplicate of {' | '.join(x for x in a if x)}"])
    for r, cells in rows.items():
        if TOTALISH.search(r[1]):
            continue
        nz = [v for v in cells.values() if v not in (None, 0) and v not in BASES and abs(v) >= 10]
        if len(cells) >= 5 and len(nz) >= 5:
            mode, k = collections.Counter(nz).most_common(1)[0]
            if k / len(nz) >= 0.8:
                flags["frozen_row"] += 1
                detail.append(["row", " | ".join(x for x in r if x), f"frozen (value {norm_value(mode)} in {k}/{len(nz)})"])
    if med is not None and med < 0.35 and page_ok:
        flags["table_ungrounded"] += 1
        detail.append(["table", "", f"most columns ungrounded (median {med:.2f}) though the page OCR found {n_ocr} multi-digit numbers"])
    n_bad_cols = len({d[1] for d in detail if d[0] == "col"})
    summary = dict(n_cols=len(cols), n_rows=n_rows, table_grounding=recall, page_ocr_ok=page_ok, bad_cols=n_bad_cols,
                   bad_col_share=round(n_bad_cols / len(cols), 3), median_col_grounding=None if med is None else round(med, 3),
                   **{k: flags.get(k, 0) for k in ("ungrounded", "frozen", "duplicate", "dead", "frozen_row", "table_ungrounded")})
    return summary, detail


def load_page_nums():
    nums = collections.defaultdict(collections.Counter)
    p = os.path.join(HERE, "out", "page_ocr_numbers.jsonl")
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            nums[d["img"]].update(d["nums"])
    return nums


def main(only=None):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    h = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    ocr = load_page_nums()
    out, det = [], []
    pages = only or [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(SITE, "tables", "*.html"))]
    for page in pages:
        tid = h.get(page.rsplit("_", 1)[1])
        if not tid:
            continue
        s = open(os.path.join(SITE, "tables", page + ".html"), errors="ignore").read()
        imgs = re.findall(r'src="\.\./img/([^"]+)"', s)
        pn = collections.Counter()
        complete = bool(imgs) and all(im in ocr for im in imgs)
        if complete:                 # score only against the full set of the table's pages
            for im in imgs:
                pn.update(ocr[im])
        facts = [dict(r) for r in con.execute(
            "select section,row_label_std,column_group_std,sub_column_std,value from facts where table_id=?", (tid,))]
        summ, detail = audit_table(facts, pn)
        if summ is None:
            continue
        out.append(dict(page=page, table_id=tid, ocr_available=complete, **summ))
        det += [[page] + d for d in detail]
    with open(os.path.join(HERE, "out", "column_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    with open(os.path.join(HERE, "out", "column_audit_detail.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["page", "level", "column_or_row", "reason"]); w.writerows(det)
    return out, det


if __name__ == "__main__":
    out, det = main(sys.argv[1:] or None)
    flagged = [r for r in out if r["bad_cols"] or r["frozen_row"] or r["table_ungrounded"]]
    print(f"tables audited: {len(out)} (with page OCR: {sum(r['ocr_available'] for r in out)}); flagged: {len(flagged)}")
