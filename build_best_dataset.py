#!/usr/bin/env python3
"""BEST-PER-PLACE-YEAR Table VII dataset — the data-science-ready bundle for the site.
For every (province, census year): pick the BEST source among our holdings (user-confirmed tables volume
first, then by run quality/completeness), and export its assembled Table VII. Produces:
  census_site/data/tableVII_best.zip     README + catalog + tidy long.csv + per-table CSVs
  census_site/data/coverage_matrix.csv   province x year: BEST / MARGINAL / NO_SOURCE  (the priority check)
  python3 build_best_dataset.py
"""
import sys, os, json, csv, re, zipfile, io, datetime
sys.argv = ["x"]
import census_review_app as A
from canonical_places import canonicalize

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "census_site/data")
QRANK = {"verified": 0, "minor-issues": 1, "needs-review": 2, "incomplete": 3, "": 4}


def province_of(title, region):
    m = re.search(r"vol(?:ume)?\.?\s*[IVXLC\d]+[-A-B]*\s*[,.]\s*([^.:;]+?)(?:\s*[.:;]|\s+pt\b|\s+part\b|$)",
                  title or "", re.I)
    if m:
        return re.sub(r"^the\s+", "", m.group(1).strip())
    return (region or "").strip() or "(unknown)"


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    confirmed = {k.split(":")[1] for k, v in verd.items() if v.get("range") or v.get("ranges")}
    titles = {r["saoa"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/document_titles.csv")))}
    quality = {}
    fp = os.path.join(ROOT, "export/first_cut/catalog.csv")
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp)):
            quality[(r["doc"], r["page_range"])] = r["quality"]

    # candidate runs per (province, year)
    cand = {}
    for run in runs:
        sa = str(run["saoa"])
        t = titles.get(sa, {})
        raw = province_of(t.get("official_title", ""), run.get("region", ""))
        prov, kind = canonicalize(raw)
        if kind == "junk" or not prov:
            continue
        yr = str(t.get("census_year", "") or run.get("year", ""))
        if not yr:
            continue
        q = quality.get((sa, f"{run['page_range'][0]}-{run['page_range'][1]}"), "")
        # IRONCLAD FIRST: runs whose bounds the user set by hand (via /documents confirm or /assembly
        # set-bounds) outrank everything for their place-year — they are human ground truth.
        human_bounds = run.get("start_type") in ("human_confirmed_documents", "human_set") or \
                       run.get("end_type") in ("human_confirmed_documents", "human_set")
        score = (0 if human_bounds else 1,
                 0 if sa in confirmed else 1,
                 0 if t.get("part_type") == "tables" else 1,
                 QRANK.get(q, 4), -run.get("n_pages", 0))
        cand.setdefault((prov.lower(), yr), []).append((score, prov, sa, run, q, t.get("part_type", ""), kind))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    catalog, longbuf, tables = [], io.StringIO(), {}
    lw = csv.writer(longbuf)
    lw.writerow(["province", "year", "doc", "pdf_page", "section", "row", "group", "subkey", "value"])
    cov_rows = []
    for (pk, yr), lst in sorted(cand.items()):
        lst.sort(key=lambda x: x[0])
        best = lst[0]
        _, prov, sa, run, q, ptype, kind = best
        s, e = run["page_range"]
        nrows = 0
        cells_wide = {}
        colorder = []
        for m in sorted(run.get("members", []), key=lambda m: int(m.get("page", 0) or 0)):
            d, ih, _ = A.load_table(m["table_id"])
            if not d or (d.get("metadata", {}) or {}).get("title_page"):
                continue
            for sec in d.get("data", []):
                for row in sec.get("rows", []):
                    rh = next((k for k in row if not isinstance(row[k], dict)), None)
                    lbl = str(row.get(rh, "")) if rh else ""
                    for g, v in row.items():
                        if isinstance(v, dict):
                            for k2, val in v.items():
                                lw.writerow([prov, yr, sa, m.get("page", ""), sec.get("name", "")[:40], lbl, g, k2, val])
                                ck = f"{g} | {k2}"
                                if ck not in colorder:
                                    colorder.append(ck)
                                cells_wide.setdefault((m.get("page", ""), sec.get("name", "")[:40], lbl), {})[ck] = val
                                nrows += 1
        if nrows == 0:
            continue
        tbuf = io.StringIO(); tw = csv.writer(tbuf)
        tw.writerow(["pdf_page", "section", "age_or_row"] + colorder)      # NORMAL table shape
        for (pg2, sec2, lbl2), cc in cells_wide.items():
            tw.writerow([pg2, sec2, lbl2] + [cc.get(c, "") for c in colorder])
        fn = re.sub(r"[^A-Za-z0-9]+", "_", f"{prov}_{yr}_{sa}")[:60] + ".csv"
        tables[fn] = tbuf.getvalue()
        _hb = run.get("start_type") in ("human_confirmed_documents", "human_set") or \
              run.get("end_type") in ("human_confirmed_documents", "human_set")
        best_kind = ("IRONCLAD" if _hb else
                     ("BEST" if (sa in confirmed and ptype == "tables") else
                      ("CONFIRMED" if sa in confirmed else "MARGINAL")))
        _sitekey = re.sub(r"[^A-Za-z0-9]+", "_", f"{sa}_{s}_{e}")
        catalog.append(dict(province=prov, year=yr, place_kind=kind, doc=sa, page_range=f"{s}-{e}",
                            site_page=f"assembled/{_sitekey}.html",
                            quality=q or "unlabeled", source_kind=best_kind, volume_type=ptype or "?",
                            n_cells=nrows, file=f"tables/{fn}",
                            alternatives="|".join(x[2] for x in lst[1:4])))
        cov_rows.append(dict(province=prov, year=yr, status=best_kind, doc=sa, quality=q or ""))
    # VIA-NATIONAL synthesis: a province-year with no direct volume is still covered if a national
    # volume exists that year (All-India Table VII breaks down BY PROVINCE).
    nat_by_year = {}
    for c in catalog:
        if c["place_kind"] in ("national", "region"):
            nat_by_year.setdefault(c["year"], c["doc"])
    provs_all = sorted({c["province"] for c in catalog if c["place_kind"] == "province"})
    direct = {(r["province"].lower(), r["year"]) for r in cov_rows}
    for yr, natdoc in nat_by_year.items():
        for p in provs_all:
            if (p.lower(), yr) not in direct:
                cov_rows.append(dict(province=p, year=yr, status="VIA_NATIONAL", doc=natdoc,
                                     quality="see national volume"))

    zpath = os.path.join(OUT, "tableVII_best.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        cb = io.StringIO(); w = csv.DictWriter(cb, fieldnames=list(catalog[0].keys())); w.writeheader(); w.writerows(catalog)
        z.writestr("catalog.csv", cb.getvalue())
        z.writestr("tableVII_long.csv", longbuf.getvalue())
        for fn, content in tables.items():
            z.writestr("tables/" + fn, content)
        z.writestr("README.md", f"""# Imperial Table VII — BEST table per place-year ({stamp})
One age/sex/civil-condition table per (province, census year): the best source among our holdings —
human-verified tables volumes first ('BEST'/'CONFIRMED'), else the only available source ('MARGINAL').
- catalog.csv: what was chosen for each place-year, its quality label + alternatives.
- tableVII_long.csv: every cell, tidy long format (province, year, doc, page, section, row, group, subkey, value).
- tables/: one readable CSV per place-year.
Quality labels: verified = passes all internal arithmetic checks; minor-issues = a few cells off
(often source typos); needs-review = larger discrepancies; see the project provenance for lineage.
{len(catalog)} place-years · {sum(c['n_cells'] for c in catalog):,} cells.
""")
    with open(os.path.join(OUT, "coverage_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["province", "year", "status", "doc", "quality"])
        w.writeheader(); w.writerows(sorted(cov_rows, key=lambda r: (r["province"].lower(), r["year"])))
    # browsable index: peruse the best table per place-year WITHOUT downloading the zip
    import html as _h
    QB={"verified":("✓ verified","#0a5d1e"),"minor-issues":("~ minor","#b8860b"),
        "needs-review":("⚠ review","#9c0006"),"incomplete":("… incomplete","#777"),"unlabeled":("·","#999")}
    byprov={}
    for c in catalog:
        byprov.setdefault(c["province"],[]).append(c)
    out=["<!doctype html><meta charset=utf-8><title>Best Table VII per place-year</title>",
         "<style>body{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}",
         "header{background:#1f3864;color:#fff;padding:12px 20px}header a{color:#ffd966}",
         ".sec{margin:14px 20px 4px;font-weight:700;color:#1f3864;border-bottom:2px solid #e3e9f5;font-size:16px}",
         ".grid{display:flex;flex-wrap:wrap;gap:8px;margin:8px 20px}",
         ".c{background:#fff;border:1px solid #e2e6ee;border-left:4px solid #1f3864;border-radius:7px;padding:8px 12px;",
         "text-decoration:none;color:inherit;width:250px}.c:hover{box-shadow:0 2px 8px rgba(0,0,0,.1)}",
         ".b{font-size:11px;padding:0 6px;border-radius:8px;color:#fff}</style>",
         f"<header><a href='index.html'>← home</a> <b>Best Table VII — every place &amp; census year</b> · ",
         f"{len(catalog)} place-years · <a href='data/tableVII_best.zip'>download all (zip)</a> · ",
         "<a href='data/coverage_matrix.png'>coverage map</a></header>"]
    for prov in sorted(byprov):
        out.append(f"<div class=sec>{_h.escape(prov)}</div><div class=grid>")
        for c in sorted(byprov[prov], key=lambda x: x["year"]):
            ql,qc=QB.get(c["quality"],("·","#999"))
            out.append(f"<a class=c href='{c['site_page']}'><b>{c['year']}</b> "
                       f"<span class=b style='background:{qc}'>{ql}</span> "
                       f"<span class=b style='background:#456'>{c['source_kind']}</span><br>"
                       f"<span style='color:#789;font-size:12px'>{c['n_cells']:,} cells · doc {c['doc'][:10]} pp{c['page_range']}</span></a>")
        out.append("</div>")
    open(os.path.join(ROOT,"census_site/best.html"),"w").write("\n".join(out))
    print(f"browsable index -> census_site/best.html ({len(catalog)} entries)")
    from collections import Counter
    kc = Counter(c["source_kind"] for c in catalog)
    qc = Counter(c["quality"] for c in catalog)
    print(f"best dataset: {len(catalog)} place-years -> {zpath} ({os.path.getsize(zpath)//1024//1024}MB)")
    print(f"  source kinds: {dict(kc)} · quality: {dict(qc)}")


if __name__ == "__main__":
    main()
