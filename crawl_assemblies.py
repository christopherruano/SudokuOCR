#!/usr/bin/env python3
"""DIRECTIONAL assembly crawler — grow each Table VII run only in the direction(s) it actually needs,
walking outward page-by-page until a boundary (blank / different table / title / TOC page), then stop.

Which directions are open, per run:
  * If you REVIEWED it (export/review_verdicts/assembly.json):
      verdict 'correct'            -> no crawl (you confirmed it's right)
      note mentions before/after/both, 'end good', 'start good', 'earlier' -> follow exactly
      verdict wrong/boundary, unclear note -> both directions
  * Else (un-reviewed) use the detector's own signal:
      start_type continued_origin_missing / flag 'start_may_be_earlier' -> crawl BEFORE
      end_type not 'concluded'                                          -> crawl AFTER

The walk (per open direction) skips pages already read AND Table VII (they're part of the run, keep
going), STOPS at a page already read that is NOT Table VII (that's the boundary — direction done), and
when it reaches the first UNREAD page it renders that one page from the drive and queues it for OCR.
Across rounds the runs absorb new Table VII pages and the crawl advances; directions whose next page
reads non-VII naturally stop. Convergence = a round that queues nothing.

  python3 crawl_assemblies.py [--reach N] [--dry-run]
Writes export/boundary_grow_renders/*.jpg + export/boundary_grow_list.txt (same outputs the grow loop
already OCRs), plus export/crawl_directions.csv (a transparency report of what it decided per run).
"""
import json, os, re, sys, csv, hashlib, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
OUT = os.path.join(ROOT, "export/boundary_grow_renders")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERD = os.path.join(ROOT, "export/review_verdicts/assembly.json")
MASTER = os.path.join(V1, "page_number_status.csv")
REACH = int(sys.argv[sys.argv.index("--reach") + 1]) if "--reach" in sys.argv else 6
DRY = "--dry-run" in sys.argv

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


def read_index():
    """(dochash, pdf_page) -> True if that page is read AND Table VII, False if read but not VII.
    Missing key = unread. Built from the master, which carries is_tableVII per crop; a page counts as
    VII if ANY crop on it is VII."""
    page = {}
    if not os.path.exists(MASTER):
        return page
    for r in csv.DictReader(open(MASTER)):
        dh, pg = r.get("dochash", ""), r.get("pdf_page", "")
        if not (dh and pg.lstrip("-").isdigit()):
            continue
        k = (dh, int(pg))
        page[k] = page.get(k, False) or (r.get("is_tableVII") == "1")
    return page


def probed_pages():
    """(saoa, pdf_page) the crawl already OCR'd (export/boundary_grow_ocr). The corpus adapter DROPS
    0-row reads (blank pages / prose / pure title pages), so they never reach the master — but a page we
    already probed that did NOT come back as a corpus Table VII read is a confirmed boundary. Without
    this set the crawl would re-render & re-OCR those same blank boundary pages every single round."""
    s = set()
    for f in glob.glob(os.path.join(ROOT, "export/boundary_grow_ocr/**/*_oneshot.json"), recursive=True):
        m = re.search(r"(\d+|[0-9a-f]{16})_p(\d+)_oneshot", os.path.basename(f))
        if m:
            s.add((m.group(1), int(m.group(2))))
    return s


def open_directions(run, verd):
    """Return subset of {'before','after'} to crawl for this run."""
    v = (verd.get("v") if isinstance(verd, dict) else verd) or ""
    note = ((verd.get("note") if isinstance(verd, dict) else "") or "").lower()
    if v == "correct":
        return set(), "reviewed:correct->skip"
    if v in ("wrong", "boundary"):
        both = "both" in note or "before and after" in note or "both ends" in note or "both direction" in note
        before = both or "before" in note or "earlier" in note
        after = both or ("after" in note and "before" not in note.split("after")[0][-12:]) or "after" in note
        # 'end good' / 'start good' explicitly close a side
        if re.search(r"end (is )?good|end good", note):
            after = False
        if re.search(r"start (is )?good|before good", note):
            before = False
        dirs = set()
        if before:
            dirs.add("before")
        if after:
            dirs.add("after")
        if not dirs:                      # flagged wrong but unclear -> crawl both
            dirs = {"before", "after"}
        return dirs, f"reviewed:{v}:{note[:40]}"
    # ---- un-reviewed: use the detector's structural signal ----
    dirs = set()
    flags = run.get("flags") or []
    if run.get("start_type") == "continued_origin_missing" or any("earlier" in f for f in flags):
        dirs.add("before")
    if run.get("end_type") != "concluded":
        dirs.add("after")
    return dirs, "detector:" + ("+".join(sorted(dirs)) or "none")


def main():
    vol = vol_map()
    runs = json.load(open(RUNS))
    verds = json.load(open(VERD)) if os.path.exists(VERD) else {}
    read = read_index()
    probed = probed_pages()
    os.makedirs(OUT, exist_ok=True)

    # decide directions + walk to the first unread page per open direction
    targets = []       # (saoa, pdf_page, cluster_path)
    seen = set()
    report = []
    open_b = open_a = closed = 0
    for run in runs:
        dh = run["dochash"]
        cp = vol.get(dh)
        key = f"{run['saoa']}:{run['page_range']}"
        dirs, why = open_directions(run, verds.get(key, ""))
        s, e = run["page_range"]
        decided = {"before": "", "after": ""}
        for d in dirs:
            step = -1 if d == "before" else 1
            pg = (s - 1) if d == "before" else (e + 1)
            reach = 0
            outcome = "edge"
            while reach < REACH and pg >= 1:
                st = read.get((dh, pg))
                if st is True:                     # read & VII -> part of the run, keep walking out
                    pg += step; reach += 1; continue
                if st is False:                    # read & NOT VII -> boundary, this direction is done
                    outcome = f"boundary@p{pg}"; break
                if (run["saoa"], pg) in probed:    # already OCR'd by the crawl, came back 0-row/blank
                    outcome = f"boundary@p{pg}(probed)"; break   # -> boundary; stop, don't re-probe
                # unread -> this is the page to render & OCR
                if cp and (run["saoa"], pg) not in seen:
                    seen.add((run["saoa"], pg)); targets.append((run["saoa"], pg, cp))
                    outcome = f"probe@p{pg}"
                else:
                    outcome = f"probe@p{pg}(dup)"
                break
            decided[d] = outcome
            if d == "before":
                open_b += 1
            else:
                open_a += 1
        if not dirs:
            closed += 1
        report.append({"run": key, "region": run.get("region", ""), "year": run.get("year", ""),
                       "directions": "+".join(sorted(dirs)) or "none", "why": why,
                       "before": decided["before"], "after": decided["after"]})

    # transparency report
    with open(os.path.join(ROOT, "export/crawl_directions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "region", "year", "directions", "why", "before", "after"])
        w.writeheader(); w.writerows(report)

    probes = len(targets)
    print(f"CRAWL: {len(runs)} runs | before-open {open_b} | after-open {open_a} | no-crawl {closed}")
    print(f"  next unread pages to OCR this round: {probes}  (-> export/crawl_directions.csv for per-run plan)")
    if DRY:
        print("  --dry-run: not rendering.")
        return probes

    # render the probe pages from the drive (full 300-DPI grayscale JPEG; small enough to upload)
    lst = open(os.path.join(ROOT, "export/boundary_grow_list.txt"), "w")
    by_pdf = {}
    for sa, pg, cp in targets:
        by_pdf.setdefault(cp, []).append((sa, pg))
    rendered = skip = nodrv = 0
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
    print(f"  rendered {rendered} | had {skip} | no-drive {nodrv} -> export/boundary_grow_list.txt ({n} pages)")
    return n


if __name__ == "__main__":
    main()
