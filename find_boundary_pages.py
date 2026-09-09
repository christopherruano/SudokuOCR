#!/usr/bin/env python3
"""Turn the assembly-boundary feedback ("runs extend before/after") into PAGES TO READ.
For each detected Table VII run, probe the pages just before the start and just after the end; any that
have NO read yet (gap) get RENDERED from the drive so they can be OCR'd. Repeated rounds let runs grow
to their true title-page start and district-level extent. Pulls from /Volumes/Expansion (the drive).
  python3 find_boundary_pages.py [--reach N]   (default reach=4 pages each side)
Writes export/boundary_grow_renders/*.png + export/boundary_grow_list.txt
"""
import json, os, re, sys, csv, hashlib, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
OUT = os.path.join(ROOT, "export/boundary_grow_renders")
REACH = int(sys.argv[sys.argv.index("--reach") + 1]) if "--reach" in sys.argv else 4

try:
    import fitz
    from PIL import Image
except ImportError:
    sys.exit("need pymupdf + pillow")


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
    vol = vol_map()
    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    # which (saoa,page) already have a read (corpus imghash exists)?
    have = set()
    for f in glob.glob(os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables/*/*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if m:
            have.add(m.group(1))
    # also pages already rendered this campaign
    for d in ("export/boundary_grow_renders", "export/boundary_gap_renders", "export/empty_vii_renders"):
        pass

    os.makedirs(OUT, exist_ok=True)
    targets = []   # (saoa, page, cluster_path)
    seen = set()
    for run in runs:
        dh = run["dochash"]
        cp = vol.get(dh)
        if not cp:
            continue
        sa = run["saoa"]
        s, e = run["page_range"]
        probe = list(range(max(1, s - REACH), s)) + list(range(e + 1, e + REACH + 1))
        for pg in probe:
            key = (sa, pg)
            if key in seen:
                continue
            seen.add(key)
            ih = hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest()
            if ih in have:          # already read -> the detector will just include it; skip
                continue
            targets.append((sa, pg, cp))

    # render the gap probes from the drive
    lst = open(os.path.join(ROOT, "export/boundary_grow_list.txt"), "w")
    rendered = skip = nodrv = 0
    by_pdf = {}
    for sa, pg, cp in targets:
        by_pdf.setdefault(cp, []).append((sa, pg))
    for cp, items in by_pdf.items():
        ym = re.search(r"/(\d{4})/", cp)
        yr = ym.group(1) if ym else ""
        path = os.path.join(DRIVE, yr, os.path.basename(cp))
        if not os.path.exists(path):
            hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(cp)))
            path = hits[0] if hits else ""
        if not path:
            nodrv += len(items); continue
        try:
            doc = fitz.open(path)
        except Exception:
            nodrv += len(items); continue
        for sa, pg in items:
            out = os.path.join(OUT, f"{sa}_p{pg}.jpg")
            if os.path.exists(out):
                skip += 1
            elif 1 <= pg <= doc.page_count:
                try:
                    # FULL 300-DPI GRAYSCALE JPEG: keeps full resolution (helps OCR accuracy on dense
                    # tables) but ~1/3 the size of the 2.6MB RGB PNG (census tables are B&W), so it still
                    # uploads to Gemini. We do a FULL read of each boundary page (not just the title), so
                    # pages that join a run already have good cell data and don't need re-OCR.
                    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72),
                                                           colorspace=fitz.csGRAY)
                    Image.frombytes("L", (pix.width, pix.height), pix.samples).save(
                        out, "JPEG", quality=88, optimize=True)
                    pix = None; rendered += 1
                except Exception:
                    continue
            else:
                continue
            if os.path.exists(out):
                lst.write(os.path.abspath(out) + "\n")
        doc.close()
    lst.close()
    n = sum(1 for _ in open(os.path.join(ROOT, "export/boundary_grow_list.txt")))
    print(f"boundary-grow probes: {len(targets)} unread pages near run edges | rendered {rendered} | had {skip} | no-drive {nodrv}")
    print(f"-> export/boundary_grow_list.txt ({n} pages to OCR)")
    return n


if __name__ == "__main__":
    main()
