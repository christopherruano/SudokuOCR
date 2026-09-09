#!/usr/bin/env python3
"""SINGLE-AGE (year-by-year) age returns — hunted across the ENTIRE corpus. A table qualifies if any
section has >=15 consecutive single-year age rows (labels like '0-1','1-2' or bare '0','1','2'...).
These are the demographic gold (age-heaping, smoothing, single-age analysis). Duplication with the
best-per-place-year bundle is fine and intended. Emits:
  census_site/data/tableVII_single_age.zip   single_age_long.csv + catalog.csv + README
  census_site/singleage.html                 browsable index of every single-age table found
  python3 build_single_age.py
"""
import os, re, csv, io, json, glob, zipfile, datetime, html
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
SINGLE = re.compile(r"^\s*(\d{1,2})\s*(?:[-–]\s*(\d{1,2}))?\s*\.?\s*$")


def age_of(lbl):
    m = SINGLE.match(str(lbl or ""))
    if not m:
        return None
    a = int(m.group(1)); b = m.group(2)
    if b is not None and int(b) not in (a, a + 1):
        return None
    return a if a <= 100 else None


def main():
    loc = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        loc[r["imghash"]] = (r.get("dochash", ""), r.get("pdf_page", ""), r.get("assembly_run", ""))
    # doc-level year/region/title fallback (table metadata is often missing them)
    from canonical_places import canonicalize
    docmeta = {}
    titles_by_saoa = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/document_titles.csv"))):
        titles_by_saoa[r["saoa"]] = r
    for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv"))):
        if r.get("dochash"):
            t = titles_by_saoa.get(r["saoa"], {})
            official = (t.get("official_title") or r.get("title") or "").strip()
            canon, kind = canonicalize(r.get("region") or official)
            region = canon if kind != "junk" and canon else (r.get("region", "") or "")
            docmeta[r["dochash"]] = (t.get("census_year") or r.get("year", ""), region, official)
    # dochash not in coverage -> derive year (from the source-PDF path's /18xx|19xx/ folder) + saoa title
    pm_src = {}
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r.get("dochash") and r.get("source_pdf") and r["dochash"] not in pm_src:
            pm_src[r["dochash"]] = r["source_pdf"]
    for dh, sp in pm_src.items():
        if dh in docmeta and docmeta[dh][0]:
            continue
        ym = re.search(r"/(1[89]\d{2})/", sp)
        sm = re.search(r"(\d{8})", os.path.basename(sp))
        t = titles_by_saoa.get(sm.group(1), {}) if sm else {}
        yr = t.get("census_year") or (ym.group(1) if ym else "")
        official = (t.get("official_title") or "").strip()
        canon, kind = canonicalize(t.get("region") or official)
        old = docmeta.get(dh, ("", "", ""))
        docmeta[dh] = (old[0] or yr, old[1] or (canon if kind != "junk" and canon else ""), old[2] or official)
    # fallback attributions: dochash from the corpus DIR name; pdf_page from assembly-run members
    import json as _j
    dir_dh = re.compile(r"^([0-9a-f]{16})[_.]")
    page_by_tid = {}
    for run in _j.load(open(os.path.join(ROOT, "export/tableVII_runs.json"))):
        for m in run.get("members", []):
            page_by_tid[m.get("table_id", "")] = (str(m.get("page", "")), run.get("dochash", ""))
    seen = set()
    long_rows = []
    catalog = []
    files = glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json"))
    for f in files:
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        ih = m.group(1) if m else None
        if not ih or ih in seen:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = d.get("metadata", {}) or {}
        dh0, _pg0, _a0 = loc.get(ih, ("", "", ""))
        if not dh0:                                   # recover dochash from the corpus dir name
            m2 = dir_dh.match(os.path.basename(os.path.dirname(f)))
            if m2: dh0 = m2.group(1)
        if not dh0 or not _pg0:                       # recover page via assembly membership
            tid_guess = "digitized_tables/" + os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f).replace(".json", "")
            pgm, dhm = page_by_tid.get(tid_guess, ("", ""))
            _pg0 = _pg0 or pgm
            dh0 = dh0 or dhm
        dyr, dreg, dtitle = docmeta.get(dh0, ("", "", ""))
        yr_eff = str(meta.get("year") or "").strip()
        if yr_eff in ("", "0", "None"):
            yr_eff = dyr
        reg_eff = (meta.get("region") or "").strip() or dreg
        title_eff = (meta.get("title") or "").strip() or (dtitle + " — single-age returns" if dtitle else "")
        hits = 0
        for s in d.get("data", []):
            rows = s.get("rows", [])
            if len(rows) < 15:
                continue
            rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
            ages = [age_of(r.get(rh)) for r in rows] if rh else []
            # longest consecutive run of single ages
            best = run = 0
            prev = None
            for a in ages:
                if a is not None and (prev is None or a == prev + 1 or a == prev):
                    run += 1; best = max(best, run)
                else:
                    run = 1 if a is not None else 0
                prev = a
            if best < 15:
                continue
            hits += 1
            dh, pg, arun = loc.get(ih, ("", "", ""))
            dh = dh or dh0; pg = pg or _pg0
            for r_ in rows:
                a = age_of(r_.get(rh))
                if a is None:
                    continue
                for g, v in r_.items():
                    if isinstance(v, dict):
                        for k2, val in v.items():
                            long_rows.append([reg_eff, yr_eff, dh, pg, ih[:12],
                                              s.get("name", "")[:40], a, g, k2, val])
        if hits:
            seen.add(ih)
            dh, pg, arun = loc.get(ih, ("", "", ""))
            dh = dh or dh0; pg = pg or _pg0
            catalog.append(dict(region=reg_eff, year=yr_eff, title=title_eff[:70], dochash=dh,
                                pdf_page=pg, imghash=ih, n_sections=hits, assembly_run=arun))
    # group per-page hits into TABLE entries: consecutive pages of the same doc = one multi-page table
    catalog.sort(key=lambda c: (c["dochash"], int(c["pdf_page"] or 0)))
    grouped = []
    unattributed = []
    for c in catalog:
        if not (c["region"] and c["title"] and c["year"]):
            unattributed.append(c); continue
        pg = int(c["pdf_page"] or 0)
        g = grouped[-1] if grouped else None
        if g and g["dochash"] == c["dochash"] and pg - g["_last"] <= 1 and g["title"] == c["title"]:
            g["_last"] = pg; g["n_pages"] += 1; g["page_span"] = f"{g['_first']}-{pg}"
        else:
            grouped.append(dict(region=c["region"], year=c["year"], title=c["title"], dochash=c["dochash"],
                                page_span=str(pg), n_pages=1, _first=pg, _last=pg))
    for g in grouped:
        g.pop("_first", None); g.pop("_last", None)
    print(f"grouped into {len(grouped)} table entries · unattributed pages held back: {len(unattributed)}")
    if unattributed:
        import csv as _c
        with open(os.path.join(ROOT, "export/single_age_unattributed.csv"), "w", newline="") as f:
            w = _c.DictWriter(f, fieldnames=list(unattributed[0].keys())); w.writeheader(); w.writerows(unattributed)
    catalog_pages = catalog
    catalog = grouped
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    zpath = os.path.join(ROOT, "census_site/data/tableVII_single_age.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        cb = io.StringIO(); w = csv.DictWriter(cb, fieldnames=list(catalog[0].keys())); w.writeheader(); w.writerows(catalog)
        z.writestr("catalog.csv", cb.getvalue())
        lb = io.StringIO(); lw = csv.writer(lb)
        lw.writerow(["region", "year", "dochash", "pdf_page", "imghash", "section", "age", "group", "subkey", "value"])
        lw.writerows(long_rows)
        z.writestr("single_age_long.csv", lb.getvalue())
        z.writestr("README.md", f"# Year-by-year (single-age) census age returns ({stamp})\n"
                   f"Every table in the corpus with >=15 consecutive SINGLE-YEAR age rows — unbinned.\n"
                   f"{len(catalog)} tables · {len(long_rows):,} cells. Overlaps with the best-per-place-year\n"
                   f"bundle by design: this is the ANALYSIS layer (age heaping, smoothing, single-age demography).\n")
    # browsable index
    byry = defaultdict(list)
    for c in catalog:
        byry[(c["region"] or "(unknown)", str(c["year"]))].append(c)
    out = ["<!doctype html><meta charset=utf-8><title>Single-age returns</title>",
           "<style>body{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}",
           "header{background:#1f3864;color:#fff;padding:12px 20px}header a{color:#ffd966}",
           "table{border-collapse:collapse;margin:12px 20px;font-size:13px}td,th{border:1px solid #dde;padding:3px 9px}",
           "th{background:#eef3f7}</style>",
           f"<header><a href='index.html'>← home</a> <b>★ Year-by-year age returns</b> · {len(catalog)} tables · ",
           f"<a href='data/tableVII_single_age.zip' style='font-weight:700'>download all (zip)</a></header>",
           "<table><tr><th>region</th><th>year</th><th>title</th><th>doc</th><th>pages</th><th># pages</th></tr>"]
    for (reg, yr), lst in sorted(byry.items()):
        for c in lst:
            out.append(f"<tr><td>{html.escape(reg)}</td><td>{yr}</td><td>{html.escape(c['title'])}</td>"
                       f"<td>{c['dochash'][:10]}</td><td>{c['page_span']}</td><td>{c['n_pages']}</td></tr>")
    out.append("</table>")
    open(os.path.join(ROOT, "census_site/singleage.html"), "w").write("\n".join(out))
    print(f"single-age hunt: {len(catalog)} tables · {len(long_rows):,} cells -> {zpath} "
          f"({os.path.getsize(zpath)//1024}KB) + census_site/singleage.html")
    from collections import Counter
    print("  by year:", dict(sorted(Counter(str(c['year']) for c in catalog).items())))


if __name__ == "__main__":
    main()
