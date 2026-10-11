#!/usr/bin/env python3
"""Printed vs digitized column counts per table: the direct test of whether columns were read.

printed   per scan page, the larger of two independent counts, both of which can only
          undercount: right-aligned number bands (scan_columns.py, out/scan_columns.jsonl)
          and printed vertical rules (scan_rules.py; rules - 1)
digitized distinct (column group, sub-column) pairs in the table's widest section

not read  digitized <= widest page - max(2, 30%)   (the counter only undercounts, so this
          direction is reliable)
invented  digitized >= sum over pages + max(2, 30%), only where every page is dense
          (>= 15 entries in its busiest column band), since sparse pages undercount
Writes out/column_count_check.csv (every table) and prints the counts.
"""
import os, re, csv, json, sqlite3, hashlib, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site")


def _rules(args):
    import scan_rules
    img, rot = args
    try:
        from PIL import Image
        p = os.path.join(SITE, "img", img)
        if rot:
            q = os.path.join("/tmp", f"_rot_{os.getpid()}.png")
            Image.open(p).rotate(rot, expand=True).save(q); p = q
        return img, max(0, len(scan_rules.rules(p, 0.10)) - 1)
    except Exception:
        return img, 0


RULES = {}


def main():
    import sys
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, HERE)
    sc = {}
    for l in open(os.path.join(HERE, "out", "scan_columns.jsonl")):
        d = json.loads(l); sc[d["img"]] = d
    cache = os.path.join(HERE, "out", "scan_rules.json")
    RULES.update(json.load(open(cache)) if os.path.exists(cache) else {})
    todo = [(i, d.get("rot", 0)) for i, d in sc.items() if i not in RULES]
    with ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for img, n in ex.map(_rules, todo, chunksize=16):
            RULES[img] = n
    json.dump(RULES, open(cache, "w"))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pages = {}
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        p = os.path.basename(f)[:-5]
        pages.setdefault(p.rsplit("_", 1)[1], (p, re.findall(r'src="\.\./img/([^"]+)"', open(f, errors="ignore").read())))
    tid_of = {hashlib.md5(t.encode()).hexdigest()[:7]: t for (t,) in con.execute("select table_id from tables")}
    seccols = collections.defaultdict(lambda: collections.defaultdict(set))
    want = {tid_of[h] for h in pages if h in tid_of}
    for tid, sec, g, s in con.execute("select table_id,section,column_group_std,sub_column_std from facts"):
        if tid in want:
            seccols[tid][sec or ""].add(((g or "").strip().lower(), (s or "").strip().lower()))
    out = []
    for h, (page, imgs) in pages.items():
        tid = tid_of.get(h)
        if not tid or tid not in seccols or not imgs or not all(i in sc and sc[i].get("n") is not None for i in imgs):
            continue
        dig = max(len(v) for v in seccols[tid].values())
        per = [max(sc[i]["n"], RULES.get(i, 0)) for i in imgs]
        dense = all(sc[i].get("density", 0) >= 15 or RULES.get(i, 0) >= 4 for i in imgs)
        verdict = ""
        if dig <= max(per) - max(2, 0.3 * max(per)):
            verdict = "columns not read"
        elif dense and dig >= sum(per) + max(2, 0.3 * sum(per)):
            verdict = "columns invented"
        out.append(dict(page=page, printed_per_page=" ".join(map(str, per)), digitized=dig, dense=dense, verdict=verdict))
    with open(os.path.join(HERE, "out", "column_count_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    c = collections.Counter(r["verdict"] or "ok" for r in out)
    print(f"tables compared: {len(out)}; {dict(c)}")


if __name__ == "__main__":
    main()
