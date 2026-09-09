#!/usr/bin/env python3
"""Turn the AI boundary verdicts into a confirmed-Table-VII override the pipeline can apply.

For every HIGH-confidence 'extend' verdict, the boundary page IS part of its run's multi-page table.
We emit export/confirmed_vii_pages.json = [{saoa, dochash, page, imghash, title}] so:
  - build_money_corpus_adapter folds these reads even when 0-row (continuation/divider pages), and
  - update_page_master forces is_tableVII=1 for them (overriding the regex, e.g. 'Table E' continuations)
so the run detector links them and the runs grow to their AI-judged true extent.
  python3 build_confirmed_extensions.py
"""
import json, os, re, csv, hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERD = os.path.join(ROOT, "export/boundary_verdicts.json")


def vol_map():
    m = {}
    for fn, col in (("cc_volumes.csv", "cluster_path"), ("documents.csv", "source_pdf"),
                    ("provenance.csv", "source_pdf")):
        p = os.path.join(V1, fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cp = r.get(col)
                if cp and "census_pdfs" in cp:
                    m.setdefault(hashlib.sha1(cp.encode()).hexdigest()[:16], cp)
    return m


def main():
    runs = json.load(open(RUNS))
    vol = vol_map()
    # index runs by "saoa:s-e"
    bykey = {f"{r['saoa']}:{r['page_range'][0]}-{r['page_range'][1]}": r for r in runs}
    verds = json.load(open(VERD))

    out = []
    miss = 0
    for v in verds:
        if not (v["verdict"] == "extend" and v["confidence"] == "high"):
            continue
        run = bykey.get(v["run"])
        if not run:
            miss += 1; continue
        dh = run["dochash"]
        cp = vol.get(dh)
        if not cp:
            miss += 1; continue
        pg = int(v["boundary_page"])
        out.append({"saoa": run["saoa"], "dochash": dh, "page": pg,
                    "imghash": hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest(),
                    "title": v.get("reason", "")[:60]})
    # dedup by imghash
    seen = {}
    for o in out:
        seen[o["imghash"]] = o
    out = list(seen.values())
    json.dump(out, open(os.path.join(ROOT, "export/confirmed_vii_pages.json"), "w"), indent=0)
    print(f"confirmed-VII override: {len(out)} pages from high-confidence extends (unmatched runs: {miss})")


if __name__ == "__main__":
    main()
