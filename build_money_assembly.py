#!/usr/bin/env python3
"""Focused assembly of the core-money (civil-condition / age-sex) tables from the
flat (saoa,page) OCR results — WITHOUT touching the main corpus/reingest.

Groups OCR'd pages into logical tables (by volume + manifest block = first..last page),
orders by PDF page, carries each page's constraint status, and tags the Age/Sex/Civil
Condition topic. Output: money_assembled.json (for the Neil static site) + a summary.
"""
import json, csv, re, os, glob
from collections import defaultdict

V1 = "export_final/sendout_final/sendout_v1"
RESULT_DIRS = ["export/money_renders_ocr", "export/money_gapfill_ocr"]
CORR = "export/money_gapfill_ocr/corrected"   # human-applied fixes win
VII = re.compile(r"civil\s*condition|age.{0,4}sex|marital|conjugal|age\s*and\s*sex|sex\s*and\s*age", re.I)


def load_provenance():
    prov = {}           # (saoa,page) -> {title, province, year, block, jstor, cluster_path}
    p = os.path.join(V1, "tableVII_inventory_cc.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            sa, pg = r.get("saoa_id", ""), r.get("pdf_page", "")
            if sa and str(pg).isdigit():
                prov[(sa, int(pg))] = {
                    "title": r.get("title", ""), "province": r.get("province_or_volume", ""),
                    "year": r.get("year", ""), "jstor": r.get("jstor_url", ""),
                    "block": (r.get("first_page", ""), r.get("last_page", "")),
                    "cluster_path": r.get("cluster_path", "")}
    return prov


def best_result_path(saoa, page):
    """prefer a human-corrected copy, else the OCR result."""
    name = f"{saoa}_p{page}_oneshot.json"
    c = os.path.join(CORR, name)
    if os.path.exists(c):
        return c
    for d in RESULT_DIRS:
        hits = glob.glob(f"{d}/**/{saoa}_p{page}_*oneshot.json", recursive=True)
        if hits:
            return hits[0]
    return None


def page_status(d):
    c = d.get("constraints", {}) or {}
    tot, fail = c.get("total_checks", 0) or 0, c.get("failed", 0) or 0
    nrows = sum(len(s.get("rows", [])) for s in d.get("data", []))
    if nrows == 0:
        return "empty", 0, 0, 0
    if tot and fail == 0:
        return "clean", tot, fail, nrows
    if tot and fail <= 2:
        return "minor", tot, fail, nrows
    if tot and fail / tot < 0.15:
        return "few_off", tot, fail, nrows
    return "many_off", tot, fail, nrows


def main():
    prov = load_provenance()
    # enumerate every money OCR result -> (saoa, page)
    pages = {}
    for d in RESULT_DIRS:
        for f in glob.glob(f"{d}/**/*_oneshot.json", recursive=True):
            m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
            if m:
                pages.setdefault((m.group(1), int(m.group(2))), f)
    # group into tables by (saoa, manifest block); fall back to (saoa, page) singleton
    tables = defaultdict(list)
    for (sa, pg) in pages:
        pr = prov.get((sa, pg), {})
        blk = pr.get("block", ("", ""))
        key = (sa, blk if blk != ("", "") else (pg, pg))
        tables[key].append((sa, pg))

    out = []
    for (sa, blk), members in tables.items():
        members = sorted(members, key=lambda x: x[1])
        pr0 = prov.get(members[0], {})
        pageinfo = []; is_vii = False
        title = province = year = ""
        for (s, pg) in members:
            path = best_result_path(s, pg)
            if not path:
                continue
            try:
                d = json.load(open(path))
            except Exception:
                continue
            md = d.get("metadata", {}) or {}
            secs = [x.get("name", "") for x in d.get("data", []) if x.get("name")]
            # TITLE/PROVENANCE come from the OCR'd content (printed title/region/year),
            # falling back to the (often empty) manifest provenance.
            ptitle = (md.get("title") or (secs[0] if secs else "")
                      or prov.get((s, pg), {}).get("title", "") or "").strip()
            if not title and ptitle:
                title = ptitle
            if not province and md.get("region"):
                province = str(md.get("region"))
            if not year and md.get("year"):
                year = str(md.get("year"))
            st, tot, fail, nrows = page_status(d)
            if VII.search(ptitle) or any(VII.search(x) for x in secs):
                is_vii = True
            pageinfo.append({"saoa": s, "page": pg, "status": st, "checks": tot, "failed": fail,
                             "rows": nrows, "title": ptitle[:90], "result": path,
                             "jstor": prov.get((s, pg), {}).get("jstor", "")})
        if not pageinfo:
            continue
        province = province or pr0.get("province", "")
        year = year or pr0.get("year", "")
        title = title or pr0.get("title", "")
        # table-level status (worst-of, but ignore empties for "reasonably good" floor)
        real = [p for p in pageinfo if p["status"] != "empty"]
        if not real:
            tstatus = "empty"
        elif all(p["status"] == "clean" for p in real):
            tstatus = "clean"
        elif any(p["status"] == "many_off" for p in real):
            tstatus = "needs_review"
        elif any(p["status"] in ("minor", "few_off") for p in real):
            tstatus = "minor_issues"
        else:
            tstatus = "ok"
        out.append({"saoa": sa, "block": list(blk), "year": year, "province": province,
                    "title": title[:120], "is_tableVII": is_vii, "n_pages": len(pageinfo),
                    "n_real": len(real), "status": tstatus, "pages": pageinfo})

    out.sort(key=lambda t: (not t["is_tableVII"], t["status"] != "clean", -t["n_real"]))
    json.dump(out, open(f"{V1}/money_assembled.json", "w"))
    from collections import Counter
    print(f"assembled tables: {len(out)}  (from {len(pages)} OCR'd pages)")
    print("  by status:", dict(Counter(t["status"] for t in out)))
    vii = [t for t in out if t["is_tableVII"]]
    print(f"  Age/Sex/Civil-Condition (Table VII topic) tables: {len(vii)}  "
          f"(clean={sum(1 for t in vii if t['status']=='clean')}, "
          f"minor={sum(1 for t in vii if t['status']=='minor_issues')}, "
          f"review={sum(1 for t in vii if t['status']=='needs_review')})")
    keep = [t for t in out if t["status"] != "empty"]
    print(f"  reasonably-good-and-up (for Neil, drop empties): {len(keep)}")
    print(f"wrote {V1}/money_assembled.json")


if __name__ == "__main__":
    main()
