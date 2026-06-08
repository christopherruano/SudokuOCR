#!/usr/bin/env python3
"""Prep for AI-driven multi-page stitching: per source volume (dochash), list its
tables in PRINTED-PAGE order with titles, so an AI can 'chop' the sequence into
logical multi-page tables (a run of consecutive pages that are the same table —
e.g. Imperial Table VII across districts). Writes stitch_chop_input.json.
"""
import os, re, json, csv
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
AGEFAM = {"age_civil_condition", "age_sex_population", "age_by_religion", "proportional_age",
          "caste_class_age", "single_year_age", "age_sex_civil_condition", "infirmity_age"}


def main():
    reads = {r["table_id"]: r for r in json.load(open(os.path.join(V1, "page_reads_fixed.json")))}
    prov = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "provenance.csv")))}
    fam = {}
    # family from catalogs (cheap: read provenance family)
    for t, r in prov.items():
        fam[t] = r.get("family", "")

    bydoc = defaultdict(list)
    for t, rd in reads.items():
        pg = str(rd.get("page_number", ""))
        if not pg.isdigit():
            continue
        p = prov.get(t, {})
        bydoc[rd["dochash"]].append({
            "table_id": t, "page": int(pg),
            "table_number": (rd.get("table_number", "") or "").strip(),
            "title": (p.get("title", "") or rd.get("section", "") or "")[:90],
            "family": p.get("family", ""),
        })

    docs = []
    for dh, items in bydoc.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x["page"])
        age_n = sum(1 for i in items if i["family"] in AGEFAM)
        if age_n < 2:                       # only volumes where multi-page AGE stitching matters
            continue
        info = next((prov[i["table_id"]] for i in items if i["table_id"] in prov), {})
        # compact ordered list (index by position so AI echoes a short int)
        seq = [{"i": k, "page": it["page"], "tno": it["table_number"], "fam": it["family"],
                "title": it["title"]} for k, it in enumerate(items)]
        docs.append({"dochash": dh, "year": info.get("year", ""), "region": info.get("region", ""),
                     "n_tables": len(items), "n_age": age_n,
                     "table_ids": [it["table_id"] for it in items], "sequence": seq})

    docs.sort(key=lambda d: -d["n_age"])
    json.dump(docs, open(os.path.join(V1, "stitch_chop_input.json"), "w"))
    # per-doc files so each chop agent reads only its own volume (token-efficient)
    dd = os.path.join(V1, "stitch_chop_docs")
    os.makedirs(dd, exist_ok=True)
    for i, d in enumerate(docs):
        json.dump({"index": i, "dochash": d["dochash"], "year": d["year"], "region": d["region"],
                   "sequence": d["sequence"]}, open(os.path.join(dd, f"doc_{i}.json"), "w"))
    print(f"documents to chop: {len(docs)}  (per-doc files in stitch_chop_docs/)")
    print(f"  total tables in them: {sum(d['n_tables'] for d in docs)}")
    print(f"  age tables: {sum(d['n_age'] for d in docs)}")
    big = ", ".join("{} {}({})".format(d["region"][:10], d["year"], d["n_tables"]) for d in docs[:5])
    print("  biggest volumes: " + big)


if __name__ == "__main__":
    main()
