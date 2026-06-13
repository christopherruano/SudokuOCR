#!/usr/bin/env python3
"""Make assembled multi-page tables read as ONE cohesive spreadsheet, PERFECTLY:
sections in census tables are hierarchical — a geographic header (taluk/district)
applies to the religion sub-blocks (A-Total, B-Hindu, C-Musalman...) that follow
it across pages. We walk facts in reading order, carry the current geo context,
and build a clean section = "<Geo> · <Religion>" (or just <Religion> when the
table covers a single place). Total-block first, then religions in census order,
then districts; age rows ordered with subtotals after their range, grand total
last. Rewrites assembled_facts.csv (original snapshotted)."""
import os, re, csv, shutil
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")

CONT = re.compile(r"[—\-,:;\(\s]*(continued|cont\b|contd|concld|concluded|concl)\.?\)?\.?\s*$", re.I)
GEO_RE = re.compile(r"\(?\d{0,2}\)?\s*([A-Z][A-Za-z'’]+(?:[ -][A-Z][A-Za-z'’]+){0,2})\s+(Taluk|Taluq|District|Distt|State|Division|Tahsil|Zila|City|Municipality|Agency)", re.I)
RELIG = [("All / Total", re.compile(r"all\s*(religion|communit|race)|\btotal\s*popul|\btotal\b|provincial|whole\s*state|grand", re.I)),
         ("Hindu", re.compile(r"hindu", re.I)), ("Musalman", re.compile(r"musalman|musulman|mahomedan|muslim|moslem|mohammedan", re.I)),
         ("Christian", re.compile(r"christian", re.I)), ("Jain", re.compile(r"jain", re.I)),
         ("Sikh", re.compile(r"sikh", re.I)), ("Buddhist", re.compile(r"buddhis", re.I)),
         ("Animist", re.compile(r"animis|tribal", re.I)), ("Parsi", re.compile(r"parsi|zoroastr", re.I)),
         ("Jew", re.compile(r"\bjew", re.I))]
RELIG_ORDER = {n: i for i, (n, _) in enumerate(RELIG)}
AGE_AS_SEC = re.compile(r"^\s*(age\s+)?\d+\s*[-–]\s*\d+\s*$|^\s*age\s+\d", re.I)

RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
ONE_RE = re.compile(r"(\d+)")


def religion_of(s):
    low = s.lower()
    for name, rx in RELIG:
        if rx.search(low):
            return name
    return None


def geo_of(s):
    m = GEO_RE.search(s or "")
    return m.group(1).strip().title() if m else None


def age_key(a):
    a = (a or "").strip(); low = a.lower()
    is_sub = "total" in low or "all ages" in low or "grand" in low
    m = RANGE_RE.search(a)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (hi - 0.1) if is_sub else float(lo)
    if low in ("total", "grand total", "all ages", "all"):
        return 100000.0
    m = ONE_RE.search(a)
    if m:
        n = float(m.group(1))
        return n + 0.5 if ("and over" in low or "+" in a) else n
    return 99998.0


def main():
    p = os.path.join(V1, "assembled_facts.csv")
    rows = list(csv.DictReader(open(p)))
    cols = list(rows[0].keys())
    os.makedirs(os.path.join(V1, "versions"), exist_ok=True)
    shutil.copy(p, os.path.join(V1, "versions", "assembled_facts_presection.csv"))

    by_tab = defaultdict(list)
    for r in rows:
        by_tab[r["table_id"]].append(r)

    out = []
    for tid, frs in by_tab.items():
        # pass 1 (reading order): carry geo context, assign (geo, religion, clean-name)
        cur_geo = None
        for r in frs:
            raw = CONT.sub("", (r.get("section", "") or "").strip())
            g = geo_of(raw)
            if g:
                cur_geo = g
            rel = religion_of(raw)
            r["_geo"] = cur_geo or ""
            r["_rel"] = rel
        geos = {r["_geo"] for r in frs if r["_geo"]}
        multi_geo = len(geos) >= 2
        for r in frs:
            rel = r["_rel"]
            if rel is None:
                # not a religion block: keep a cleaned named section (district/"Age Returns"), or blank
                base = CONT.sub("", (r.get("section", "") or "").strip())
                if AGE_AS_SEC.match(base):
                    base = ""
                elif base.isupper():
                    base = base.title()
                name = base[:40]
                order = (1, 99, name)
            else:
                name = (f"{r['_geo']} · {rel}" if multi_geo and r["_geo"] else rel)
                order = (0, RELIG_ORDER[rel] if not multi_geo else 0, r["_geo"])
            r["section"] = name
            r["_ord"] = order
        # order: geo-block (appearance) then religion(total first) then age
        geo_seq = []
        for r in frs:
            if r["_geo"] and r["_geo"] not in geo_seq:
                geo_seq.append(r["_geo"])
        geo_idx = {g: i for i, g in enumerate(geo_seq)}
        frs.sort(key=lambda r: (geo_idx.get(r["_geo"], 999), RELIG_ORDER.get(r["_rel"], 99),
                                age_key(r.get("row_label_std", "")), r.get("column_group_std", "")))
        for r in frs:
            for k in ("_geo", "_rel", "_ord"):
                r.pop(k, None)
        out.extend(frs)

    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in out]

    # verify: any remaining duplicate coords?
    from collections import Counter
    dup_tables = 0
    for tid, frs in by_tab.items():
        c = Counter((r["section"], r["row_label_std"], r["column_group_std"], r["sub_column_std"]) for r in frs)
        if any(n > 1 for n in c.values()):
            dup_tables += 1
    print(f"cleaned {len(by_tab)} assembled tables; multi-geo handled")
    print(f"  tables still with duplicate coords: {dup_tables}")
    secs = Counter(r["section"] for r in out)
    print("  top sections:", dict(secs.most_common(8)))


if __name__ == "__main__":
    main()
