#!/usr/bin/env python3
"""GAP LIST — diff SHOULD (content-detected Table VII pages) against HAVE (pages we
already digitized) to produce the minimal render list, plus a recall audit against
our validated-perfect pages.

Inputs (in export_final/sendout_final/sendout_v1/, or pass a dir as argv[1]):
  tableVII_inventory.csv   SHOULD  : saoa_id, pdf_page, score, block_id, header_snippet
  have_pdf_pages.csv       HAVE    : imghash, saoa_id, pdf_page
  master_tracking.csv      (local) : imghash -> validation / is_tableVII / status / table_id
  documents.csv            (local) : dochash -> saoa_id, year, province  (for labels)

Outputs:
  render_gap_list.csv          SHOULD - HAVE  = pages with NO crop -> render these
  reocr_present_imperfect.csv  in SHOULD, we HAVE a crop but it's not perfect -> re-OCR
  recall_misses.csv            validated-PERFECT pages that SHOULD did NOT flag (detector misses)
  gap_by_volume.csv            per volume: n_should / n_have / n_gap / perfect-recall
and a printed summary (overall recall %, total pages to render).
"""
import os, re, csv, sys
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def p(name):
    return os.path.join(DIR, name)


def need(name):
    if not os.path.exists(p(name)):
        print(f"  …waiting for {name} (not here yet)")
        return False
    return True


def saoa_of(s):
    m = re.search(r"(\d{6,})", s or "")
    return m.group(1) if m else ""


def main():
    print(f"gap list — reading from {DIR}")
    have_ok = need("have_pdf_pages.csv")
    should_ok = need("tableVII_inventory.csv")
    if not (have_ok and should_ok):
        print("Need both inventories from the flash-drive CC before I can diff. "
              "Place them in the folder above and re-run.")
        return

    # documents.csv: saoa <-> dochash, year, province (for labels)
    saoa_meta = {}
    if os.path.exists(p("documents.csv")):
        for r in csv.DictReader(open(p("documents.csv"))):
            sa = saoa_of(r.get("source_pdf", ""))
            if sa:
                saoa_meta[sa] = {"year": r.get("year", ""), "province": r.get("region", "")}

    # master_tracking.csv: imghash -> validation/is_tableVII/status/table_id
    val = {}
    if os.path.exists(p("master_tracking.csv")):
        for r in csv.DictReader(open(p("master_tracking.csv"))):
            ih = r.get("imghash", "")
            if ih:
                val[ih] = {"validation": r.get("validation", ""), "vii": r.get("is_tableVII", ""),
                           "status": r.get("status", ""), "table_id": r.get("table_id", "")}

    # HAVE: imghash -> (saoa, pdf_page)
    have_pages = set()                       # (saoa, pdf_page)
    have_perfect = set()                     # (saoa, pdf_page) that are validated-perfect VII
    have_imperfect = {}                      # (saoa, pdf_page) -> (imghash, validation, status)
    for r in csv.DictReader(open(p("have_pdf_pages.csv"))):
        sa = r.get("saoa_id", "") or saoa_of(r.get("source_pdf", ""))
        pg = r.get("pdf_page", "")
        if not (sa and str(pg).lstrip("-").isdigit()):
            continue
        key = (sa, int(pg)); ih = r.get("imghash", "")
        have_pages.add(key)
        v = val.get(ih, {})
        if v.get("validation") == "perfect":
            have_perfect.add(key)
        elif v.get("vii") == "Y" and v.get("validation") not in ("perfect", ""):
            have_imperfect[key] = (ih, v.get("validation", ""), v.get("status", ""))

    # SHOULD: (saoa, pdf_page) -> {score, header}
    should = {}
    for r in csv.DictReader(open(p("tableVII_inventory.csv"))):
        sa = r.get("saoa_id", ""); pg = r.get("pdf_page", "")
        if sa and str(pg).lstrip("-").isdigit():
            should[(sa, int(pg))] = {"score": r.get("score", ""), "header": r.get("header_snippet", "")}
    should_keys = set(should)

    # ---- diffs ----
    gaps = sorted(should_keys - have_pages)                       # render these
    recall_misses = sorted(have_perfect - should_keys)            # detector missed a KNOWN-perfect page
    reocr = sorted(k for k in (should_keys & set(have_imperfect)))  # have crop, not perfect, in SHOULD

    def lbl(sa):
        m = saoa_meta.get(sa, {})
        return m.get("year", ""), m.get("province", "")

    with open(p("render_gap_list.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["saoa_id", "pdf_page", "year", "province", "score", "header_snippet"])
        for sa, pg in gaps:
            y, pr = lbl(sa); s = should[(sa, pg)]
            w.writerow([sa, pg, y, pr, s["score"], s["header"]])

    with open(p("reocr_present_imperfect.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["saoa_id", "pdf_page", "imghash", "validation", "status"])
        for sa, pg in reocr:
            ih, v, st = have_imperfect[(sa, pg)]
            w.writerow([sa, pg, ih, v, st])

    with open(p("recall_misses.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["saoa_id", "pdf_page", "year", "province",
                                       "note"])
        for sa, pg in recall_misses:
            y, pr = lbl(sa)
            w.writerow([sa, pg, y, pr, "validated-perfect Table VII that content-detection did NOT flag"])

    # per-volume rollup
    vol = defaultdict(lambda: {"should": 0, "have": 0, "gap": 0, "perf": 0, "perf_hit": 0})
    for (sa, pg) in should_keys:
        vol[sa]["should"] += 1
    for (sa, pg) in have_pages:
        vol[sa]["have"] += 1
    for (sa, pg) in gaps:
        vol[sa]["gap"] += 1
    for (sa, pg) in have_perfect:
        vol[sa]["perf"] += 1
        if (sa, pg) in should_keys:
            vol[sa]["perf_hit"] += 1
    with open(p("gap_by_volume.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["saoa_id", "year", "province", "n_should", "n_have",
                                       "n_gap", "perfect_pages", "perfect_recall"])
        for sa in sorted(vol, key=lambda s: -vol[s]["gap"]):
            v = vol[sa]; y, pr = lbl(sa)
            rec = f"{100*v['perf_hit']//v['perf']}%" if v["perf"] else ""
            w.writerow([sa, y, pr, v["should"], v["have"], v["gap"], v["perf"], rec])

    # ---- summary ----
    tot_perf = len(have_perfect); hit = len(have_perfect & should_keys)
    recall = (100 * hit // tot_perf) if tot_perf else 0
    print(f"\nSHOULD pages (content-detected Table VII): {len(should_keys)}")
    print(f"HAVE pages (already digitized):            {len(have_pages)}")
    print(f"--> RENDER gap list (SHOULD - HAVE):       {len(gaps)}  -> render_gap_list.csv")
    print(f"--> re-OCR (have crop, imperfect, in SHOULD): {len(reocr)}  -> reocr_present_imperfect.csv")
    print(f"\nRECALL AUDIT (ground truth = our validated-perfect pages):")
    print(f"  perfect pages: {tot_perf} | flagged by detection: {hit} | RECALL = {recall}%")
    print(f"  detector MISSES: {len(recall_misses)}  -> recall_misses.csv "
          f"({'investigate — these are real Table VII we already proved' if recall_misses else 'none — detection covered every known-perfect page'})")
    print(f"\nper-volume breakdown -> gap_by_volume.csv")
    if recall < 95 and tot_perf:
        print("\n⚠ recall < 95% on KNOWN pages — tighten the detector (or widen render margins) "
              "before trusting the gap list as complete.")


if __name__ == "__main__":
    main()
