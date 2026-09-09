#!/usr/bin/env python3
"""Make the PDF page number IRONCLAD: one authoritative imghash -> pdf_page map covering the WHOLE corpus.
Sources, in priority:
  1. crop_pdf_pages.csv          (cluster-extracted; authoritative for cluster crops)
  2. each table's source_image   ({saoa}_p{pdfpage}.png render filename = the page we rendered)
  3. render->imghash recompute    (imghash = sha1(f"{cluster_path}:{page}") for local renders) — covers
     tables whose source_image is the {imghash}.png copy (no page in the name)
Writes export_final/.../crop_pdf_pages.csv (extended, deduped, cluster value wins) + reports coverage.
  python3 build_authoritative_pages.py
"""
import csv, glob, os, re, json, hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
CROP = os.path.join(V1, "crop_pdf_pages.csv")
DIG = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def main():
    # 1) cluster-authoritative
    auth = {}
    src = {}
    for r in csv.DictReader(open(CROP)):
        if (r.get("pdf_page") or "").strip().isdigit():
            auth[r["imghash"]] = int(r["pdf_page"])
            src[r["imghash"]] = "cluster"
    n_cluster = len(auth)

    # cluster_path per saoa -> for the render->imghash recompute
    cp_of = {}
    cc = os.path.join(V1, "cc_volumes.csv")
    if os.path.exists(cc):
        for r in csv.DictReader(open(cc)):
            if r.get("cluster_path"):
                cp_of[r["saoa_id"]] = r["cluster_path"]

    # 3) precompute render->imghash for every local render ({saoa}_p{page}.png), all render dirs
    render_hash = {}   # imghash -> page
    render_globs = [
        os.path.join(ROOT, "export/flashdrive_renders", "*.png"),
        os.path.expanduser("~/Downloads/core_money_renders/*/*/*_p*.png"),
        os.path.expanduser("~/Downloads/final_tableVII/*_p*.png"),
        os.path.join(ROOT, "export/money_renders_ocr", "**", "*_p*.png"),
    ]
    for g in render_globs:
        for png in glob.glob(g, recursive=True):
            m = re.search(r"(\d+)_p(\d+)\.png$", os.path.basename(png))
            if not m:
                continue
            sa, pg = m.group(1), int(m.group(2))
            cp = cp_of.get(sa)
            if cp:
                ih = hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest()
                render_hash.setdefault(ih, pg)

    # 2)+3) walk the corpus, fill any imghash not already cluster-authoritative
    n_src = n_recompute = n_unknown = total = 0
    for f in glob.glob(os.path.join(DIG, "*/*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if not m:
            continue
        ih = m.group(1); total += 1
        if ih in auth:
            continue
        # (2) from source_image render filename
        try:
            si = (json.load(open(f)).get("source_image") or "")
        except Exception:
            si = ""
        mm = re.search(r"_p(\d+)(?:_|\.|$)", os.path.basename(si))
        if mm:
            auth[ih] = int(mm.group(1)); src[ih] = "render_name"; n_src += 1
        elif ih in render_hash:                       # (3) recomputed
            auth[ih] = render_hash[ih]; src[ih] = "render_hash"; n_recompute += 1
        else:
            n_unknown += 1

    # write extended authoritative map
    with open(CROP, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["imghash", "pdf_page", "source"])
        for ih in sorted(auth):
            w.writerow([ih, auth[ih], src.get(ih, "")])
    print(f"AUTHORITATIVE pdf-page map: {len(auth)} imghashes")
    print(f"  from cluster: {n_cluster} | from render filename: {n_src} | recomputed render->imghash: {n_recompute}")
    print(f"  corpus tables walked: {total} | still UNKNOWN pdf_page: {n_unknown}")
    print(f"-> wrote {CROP}")


if __name__ == "__main__":
    main()
