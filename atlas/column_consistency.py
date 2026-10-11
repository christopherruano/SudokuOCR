#!/usr/bin/env python3
"""Column-structure check (no OCR): pages of one table series, and sections of one table,
should carry the same columns. A table whose columns are a strict subset of its series'
usual columns has probably lost columns in reading; one with columns its siblings lack has
probably invented or mislabelled them.

Series = same source volume + same normalized title (Table VII, "continued" pages, ...).
Output atlas/out/column_consistency.csv
"""
import os, re, csv, sqlite3, hashlib, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site")


def ntitle(t):
    t = (t or "").lower()
    t = re.sub(r"[-–—(]?\s*contin\w*\.?\)?", " ", t)
    t = re.sub(r"\bpart\s+[a-z0-9]+\b|\b\d{4}\b", " ", t)
    return re.sub(r"[^a-z]+", " ", t).strip()[:60]


def ncol(g, s):
    g = re.sub(r"[^a-z0-9]+", " ", (g or "").lower()).strip()
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    return (g, s)


KINDS = [("married", r"\bmarried"), ("unmarried", r"unmarried|single"), ("widowed", r"widow"),
         ("males", r"\bmales?\b|\bmale\b|\bm\b"), ("females", r"females?|\bf\b"), ("persons", r"persons|total|both")]


def kind(c):
    """the column's kind from its sub-column, falling back to its group: civil condition
    or sex. Age ranges are deliberately ignored."""
    for text in (c[1], c[0]):
        for k, rx in KINDS:
            if k == "married" and re.search(r"unmarried", text):
                continue
            if re.search(rx, text):
                return k
    return ""


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pub = {}
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        pub.setdefault(os.path.basename(f)[:-5].rsplit("_", 1)[1], os.path.basename(f)[:-5])
    T = {}
    for tid, title, year, place in con.execute("select table_id,title,year,canonical_name from tables"):
        h = hashlib.md5(tid.encode()).hexdigest()[:7]
        if h not in pub:
            continue
        m = re.search(r"digitized_tables/([0-9a-f]{16})_", tid)
        T[tid] = dict(page=pub[h], vol=m.group(1) if m else f"{place}|{year}", title=title, series=ntitle(title))
    cols = collections.defaultdict(lambda: collections.defaultdict(set))   # tid -> section -> cols
    for tid, sec, g, s in con.execute("select table_id,section,column_group_std,sub_column_std from facts"):
        if tid in T:
            cols[tid][sec or ""].add(ncol(g, s))
    series = collections.defaultdict(list)
    for tid, t in T.items():
        series[(t["vol"], t["series"])].append(tid)
    out = []
    for tid, t in T.items():
        secs = cols.get(tid)
        if not secs:
            continue
        allc = set().union(*secs.values())
        issues = []
        # 1. sections of this table disagree
        if len(secs) >= 3:
            cnt = collections.Counter(c for v in secs.values() for c in v)
            core = {c for c, k in cnt.items() if k >= 0.8 * len(secs)}
            short = [s for s, v in secs.items() if core and len(core - v) >= max(3, 0.5 * len(core))]
            if short:
                issues.append(f"{len(short)}/{len(secs)} sections lack columns the other sections have (e.g. {short[0][:40]!r} missing {len(core - secs[short[0]])} of {len(core)})")
        # 2. the KIND of columns differs from sibling pages of the same series. Wide tables
        # are split across pages (ages 0-14 on one, 15-39 on the next), so differing age
        # ranges are normal; but every page of a series uses the same column vocabulary
        # (e.g. married/unmarried/widowed under each age), so a page whose vocabulary
        # differs had its header misread.
        sib = [x for x in series[(t["vol"], t["series"])] if x != tid and x in cols]
        if len(sib) >= 2:
            vocab = lambda x: {kind(c) for v in cols[x].values() for c in v} - {""}
            sv = [vocab(x) for x in sib]
            cnt = collections.Counter(k for v in sv for k in v)
            usual = {k for k, n in cnt.items() if n >= 0.6 * len(sib)}
            mine = vocab(tid)
            agree = sum(len(v & usual) >= 0.7 * len(usual) for v in sv) / len(sv)
            # a page with no sex/civil-condition columns at all carries them in its rows
            # (alternating Male/Female lines), a layout choice rather than a misread
            if mine and usual and agree >= 0.6 and len(usual & mine) < 0.5 * len(usual):
                issues.append(f"column kinds differ from its {len(sib)} sibling pages: siblings use {sorted(usual)[:6]}, this page {sorted(mine)[:6]}")
        if issues:
            out.append(dict(page=t["page"], title=t["title"], n_cols=len(allc), n_sections=len(secs), issues=" || ".join(issues)))
    with open(os.path.join(HERE, "out", "column_consistency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "title", "n_cols", "n_sections", "issues"]); w.writeheader(); w.writerows(out)
    print(f"published tables checked: {len(T)}; with column-structure issues: {len(out)}")
    k = collections.Counter()
    for r in out:
        for i in r["issues"].split(" || "):
            k["sections disagree" if "sections lack" in i else "column kinds differ from siblings"] += 1
    print(dict(k))


if __name__ == "__main__":
    main()
