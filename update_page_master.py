#!/usr/bin/env python3
"""THE single source of truth: one row per corpus page, tracking WHERE it is, its SOURCE, and which
ASSEMBLY / sub-assembly it belongs to. Writes page_number_status.csv.

Per page it records:
  imghash, table_dir, dochash, source_pdf   - identity + the original source-of-truth PDF
  pdf_page, page_status                      - page number + how it was obtained (authoritative/recovered)
  is_tableVII, title, title_role             - role from the cont-marker (start/continued/concluded)
  assembly_run, run_range, run_position      - which multi-page Table VII run, and where in it
  sub_group                                  - sub-table index within the run (segmented by start markers)
  review_verdict                             - the human verdict on that run, if reviewed

Page numbers are recovered deterministically: imghash = sha1(cluster_path:page) -> brute-force the page.
Assembly membership is joined from build_tableVII_runs.py (re-run here) + the review verdicts.
  python3 update_page_master.py
"""
import csv, re, os, hashlib, glob, json, subprocess, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CROP = os.path.join(V1, "crop_pdf_pages.csv")
MASTER = os.path.join(V1, "page_number_status.csv")
DIG = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERDICTS = os.path.join(ROOT, "export/review_verdicts/assembly.json")
# PRECISE title-based relevance test (the user's "fair test"): a page is Table VII content if its
# title is age+sex/civil-condition, civil-condition, or ages-by-religion — and NOT a clearly different
# topic (education/language/birthplace/occupation/etc.). Fixes the loose 'vii' (matched 'VIII') + bare
# 'age' (matched 'Education by Age'). Same classifier the run detector uses for boundaries.
_AGE = re.compile(r"\bages?\b", re.I)
_SEXCIV = re.compile(r"\bsex\b|civil\s*condition|marital|conjugal", re.I)
_CIVIL = re.compile(r"civil\s*condition", re.I)
_BYREL = re.compile(r"ages?\s+by\s+relig", re.I)
_OTHER = re.compile(r"education|literacy|language|birth[\s-]*place|occupation|means of livelihood|"
                    r"infirmit|\bcaste\b|\btribe\b|\brace\b|housing|\btown|\bvillage|variation|migration|"
                    r"density|christian sect", re.I)


def is_vii_title(title):
    t = title or ""
    if not t.strip():
        return False
    if _OTHER.search(t) and not _CIVIL.search(t):
        return False
    if _BYREL.search(t) or _CIVIL.search(t):
        return True
    if _AGE.search(t) and _SEXCIV.search(t):
        return True
    return False
CONTD = re.compile(r"contd|continued|\(cont", re.I)
CONCL = re.compile(r"concl", re.I)
# AI-confirmed Table VII pages (boundary fan-out): force is_tableVII=1 even if the title regex wouldn't
# (e.g. supplementary 'Table E' continuations the agents judged part of the run).
CONFIRMED_IH = set()
if os.path.exists(os.path.join(ROOT, "export/confirmed_vii_pages.json")):
    for _c in json.load(open(os.path.join(ROOT, "export/confirmed_vii_pages.json"))):
        CONFIRMED_IH.add(_c["imghash"])
COLS = ["imghash", "table_dir", "dochash", "source_pdf", "pdf_page", "page_status",
        "is_tableVII", "title", "title_role", "assembly_run", "run_range",
        "run_position", "sub_group", "review_verdict"]


def isnum(x):
    return str(x).lstrip("-").isdigit()


def wide_vol_map():
    """dochash -> cluster_path from every source that names a census PDF path."""
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


def title_role(title, isvii):
    if not isvii:
        return ""
    if CONCL.search(title):
        return "concluded"
    if CONTD.search(title):
        return "continued"
    return "start"


def main():
    crop = {}
    for r in csv.DictReader(open(CROP)):
        if isnum(r.get("pdf_page")):
            crop[r["imghash"]] = (r["pdf_page"], r.get("source", "cluster"))
    vol = wide_vol_map()

    rows = []
    by_ih = {}
    recovered = 0
    for f in glob.glob(os.path.join(DIG, "*/*_oneshot.json")):
        ihm = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if not ihm:
            continue
        ih = ihm.group(1)
        tdir = os.path.basename(os.path.dirname(f))
        dm = re.match(r"([0-9a-f]{16})_t", tdir)
        dh = dm.group(1) if dm else ""
        try:
            d = json.load(open(f))
        except Exception:
            d = {}
        title = ((d.get("metadata", {}) or {}).get("title", "") or "")
        si = d.get("source_image", "") or ""
        isvii = 1 if is_vii_title(title) else 0

        if ih in crop:
            page, status = crop[ih][0], ("pdf_authoritative" if crop[ih][1] != "hash_recovered" else "pdf_hash_recovered")
        else:
            page, status = "", "unknown"
            mm = re.search(r"_p(\d+)(?:_|\.|$)", os.path.basename(si))
            if mm:
                page, status = mm.group(1), "pdf_from_render_name"
            elif dh and dh in vol:
                cp = vol[dh]
                for pg in range(1, 800):
                    if hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest() == ih:
                        page, status = str(pg), "pdf_hash_recovered"; recovered += 1
                        crop[ih] = (str(pg), "hash_recovered")
                        break
        row = {"imghash": ih, "table_dir": tdir, "dochash": dh, "source_pdf": vol.get(dh, ""),
               "pdf_page": page, "page_status": status, "is_tableVII": isvii,
               "title": title[:80], "title_role": title_role(title, isvii),
               "assembly_run": "", "run_range": "", "run_position": "", "sub_group": "", "review_verdict": ""}
        rows.append(row); by_ih[ih] = row

    # persist the recovered page map
    with open(CROP, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["imghash", "pdf_page", "source"])
        for ih, (pg, srcv) in sorted(crop.items()):
            w.writerow([ih, pg, srcv])
    # write the base master first (build_tableVII_runs reads pdf_page/is_tableVII/title from it)
    with open(MASTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)

    # ---- assembly membership: (re)detect runs, then annotate each page ----
    subprocess.run(["python3", os.path.join(ROOT, "build_tableVII_runs.py")],
                   cwd=ROOT, capture_output=True)
    n_runs = 0
    if os.path.exists(RUNS):
        runs = json.load(open(RUNS))
        n_runs = len(runs)
        verds = json.load(open(VERDICTS)) if os.path.exists(VERDICTS) else {}
        for run in runs:
            key = f"{run['saoa']}:{run['page_range']}"
            vd = verds.get(key, {})
            vv = (vd.get("v") if isinstance(vd, dict) else vd) or ""
            members = run["members"]
            sub = 0
            for i, m in enumerate(members):
                im = re.search(r"([0-9a-f]{40})_oneshot", m["table_id"])
                if not im:
                    continue
                row = by_ih.get(im.group(1))
                # sub-group: bump at each fresh 'start' page within the run (after the first)
                if row and i > 0 and row.get("title_role") == "start":
                    sub += 1
                if not row:
                    continue
                pos = ("sole" if len(members) == 1 else
                       "first" if i == 0 else "last" if i == len(members) - 1 else "interior")
                row.update({"assembly_run": key,
                            "run_range": f"{run['page_range'][0]}-{run['page_range'][1]}",
                            "run_position": pos, "sub_group": sub, "review_verdict": vv})
        # rewrite master with assembly columns filled
        with open(MASTER, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)

    # ---- report ----
    st = collections.Counter(r["page_status"] for r in rows)
    in_asm = sum(1 for r in rows if r["assembly_run"])
    reviewed = sum(1 for r in rows if r["review_verdict"])
    nvii = sum(1 for r in rows if r["is_tableVII"])
    have = len(rows) - st.get("unknown", 0)
    print(f"MASTER -> {MASTER}  ({len(rows)} pages, {len(COLS)} columns)")
    print(f"  page numbers: {have}/{len(rows)} have a PDF page ({100*have//len(rows)}%); newly recovered {recovered}")
    print(f"  Table VII pages: {nvii}")
    print(f"  ASSEMBLY: {in_asm} pages belong to one of {n_runs} multi-page runs | {reviewed} have a review verdict")
    print(f"  tracks: where (pdf_page+source_pdf) · role (start/continued/concluded) · run · sub_group · verdict")


if __name__ == "__main__":
    main()
