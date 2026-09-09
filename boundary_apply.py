#!/usr/bin/env python3
"""Set a Table VII run's exact page-range (from the review app's bounds editor) and rebuild its members
deterministically from page_number_status.csv — fast, no full-corpus scan. Pages in the new range that
have no OCR'd read yet are reported as 'missing' and queued for the grow-loop OCR. The runs file is
backed up once per process to tableVII_runs.json.prebounds.

The page-range is the source of truth; members are derived (table_id = digitized_tables/<table_dir>/
<imghash>_oneshot, page = pdf_page) exactly matching what the app's load_table() expects.
"""
import json, os, csv, shutil, hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
PNS = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/page_number_status.csv")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced")
GROWQ = os.path.join(ROOT, "export/boundary_grow_queue.txt")


def synthetic(dochash, pg):
    """Deterministic (imghash, table_dir, table_id) for a drive page that was never indexed as a table
    crop — lets us pull ANY physical PDF page into an assembly (render+OCR to a stable path)."""
    ih = hashlib.sha1(f"grow:{dochash}:{pg}".encode()).hexdigest()
    td = f"{dochash}_grow"
    return ih, td, f"digitized_tables/{td}/{ih}_oneshot"


_SRC = None
def source_pdf_for(dochash):
    """The source PDF path for a document (same for all its pages) — for rendering un-indexed pages."""
    global _SRC
    if _SRC is None:
        _SRC = {}
        for r in csv.DictReader(open(PNS)):
            if r.get("source_pdf") and r["dochash"] not in _SRC:
                _SRC[r["dochash"]] = r["source_pdf"]
    return _SRC.get(dochash)


def cache_page_image(dochash, pg, imghash, dpi=200):
    """Render a page from the drive and cache it to census_site/img/<imghash>.jpg so /scan serves it even
    when the drive is LATER disconnected. Best-effort — no-op if the drive isn't currently mounted."""
    import glob
    out = os.path.join(ROOT, "census_site/img", imghash + ".jpg")
    if os.path.exists(out):
        return True
    src = source_pdf_for(dochash)
    if not src:
        return False
    hits = glob.glob(os.path.join("/Volumes/Expansion/census_pdfs", "*", os.path.basename(src)))
    if not hits:
        return False
    try:
        import fitz
        from PIL import Image
        doc = fitz.open(hits[0])
        if not (1 <= pg <= doc.page_count):
            doc.close(); return False
        pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csGRAY)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(out, "JPEG", quality=88)
        doc.close()
        return True
    except Exception:
        return False

_IDX = None


def page_index():
    """(dochash, pdf_page) -> list of (imghash, table_dir, is_tableVII). A page can have several crops;
    we keep them all so member_for can pick the one that's actually OCR'd. Cached for the process."""
    global _IDX
    if _IDX is None:
        import collections
        idx = collections.defaultdict(list)
        with open(PNS) as f:
            for r in csv.DictReader(f):
                try:
                    pg = int(r["pdf_page"])
                except Exception:
                    continue
                if r["imghash"] and r["table_dir"]:
                    idx[(r["dochash"], pg)].append((r["imghash"], r["table_dir"], r["is_tableVII"] == "1"))
        _IDX = idx
    return _IDX


def member_for(dochash, pg):
    """Resolve a page to its corpus member. Among the page's crops, PREFER one that is actually OCR'd
    (and is Table VII); only if none are OCR'd do we return a candidate flagged as missing."""
    cands = page_index().get((dochash, pg)) or []
    ocrd = []
    for ih, td, vii in cands:
        tid = f"digitized_tables/{td}/{ih}_oneshot"
        if os.path.exists(os.path.join(CORPUS, tid + ".json")):
            ocrd.append((vii, tid))
    if ocrd:
        ocrd.sort(key=lambda x: not x[0])          # Table VII crops first
        return {"table_id": ocrd[0][1], "page": pg}, True
    # synthetic fallback: a drive page with no OCR'd indexed crop (covers un-indexed pages too)
    _ih, _td, syn_tid = synthetic(dochash, pg)
    if os.path.exists(os.path.join(CORPUS, syn_tid + ".json")):
        return {"table_id": syn_tid, "page": pg}, True
    if cands:
        ih, td, _ = sorted(cands, key=lambda c: not c[2])[0]
        return {"table_id": f"digitized_tables/{td}/{ih}_oneshot", "page": pg}, False
    return {"table_id": syn_tid, "page": pg}, False    # un-indexed page -> grow_ocr renders it from the drive


def mark_title_page(dochash, pg, title=""):
    """Record a page as a TITLE/caption page: write an empty placeholder read so it counts as a member
    (in the assembly for provenance) but has no data — so it's never OCR'd, never queued as missing, and
    contributes no rows/constraints. Refuses to clobber a page that already has real data rows."""
    m, has = member_for(dochash, pg)
    if not m:
        return {"ok": False, "err": "page not resolvable"}
    path = os.path.join(CORPUS, m["table_id"] + ".json")
    if has and os.path.exists(path):
        try:
            d = json.load(open(path))
            nr = sum(len(s.get("rows", [])) for s in d.get("data", []))
        except Exception:
            nr = 0
        if nr > 0:
            return {"ok": False, "already_data": True, "rows": nr}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"metadata": {"title_page": True, "title": title or "Title / caption page (no data)",
                            "dochash": dochash, "pdf_page": pg},
               "data": []}, open(path, "w"))
    ih = m["table_id"].rsplit("/", 1)[1].replace("_oneshot", "")
    cached = cache_page_image(dochash, pg, ih)        # persist the scan for offline (drive may disconnect)
    return {"ok": True, "table_id": m["table_id"], "image_cached": cached}


def find_run(runs, saoa, orig_range):
    saoa = str(saoa)
    orig = [int(orig_range[0]), int(orig_range[1])]
    for r in runs:
        if str(r["saoa"]) == saoa and r["page_range"] == orig:
            return r
    cand = [r for r in runs if str(r["saoa"]) == saoa
            and not (r["page_range"][1] < orig[0] or r["page_range"][0] > orig[1])]
    return cand[0] if len(cand) == 1 else None


def set_bounds(saoa, orig_range, start, end):
    start, end = int(start), int(end)
    if end < start:
        return {"ok": False, "err": "end < start"}
    runs = json.load(open(RUNS))
    run = find_run(runs, saoa, orig_range)
    if not run:
        return {"ok": False, "err": f"run not found for {saoa}:{orig_range}"}
    dh = run["dochash"]
    members, missing = [], []
    for pg in range(start, end + 1):
        m, has = member_for(dh, pg)
        if m and has:
            members.append(m)
        else:
            missing.append(pg)
    if not members:
        return {"ok": False, "err": "no OCR'd pages in requested range"}
    bak = RUNS + ".prebounds"
    if not os.path.exists(bak):
        shutil.copy(RUNS, bak)
    old = list(run["page_range"])
    # page-range = the requested extent (true table); members cover the pages we have reads for.
    run["page_range"] = [start, end]
    run["n_pages"] = end - start + 1
    run["members"] = members
    run["start_type"] = "human_set"
    run["end_type"] = "human_set"
    run["flags"] = [f for f in run.get("flags", []) if f]  # keep, but human-set supersedes
    if missing:
        run["missing_pages"] = missing
    else:
        run.pop("missing_pages", None)
    json.dump(runs, open(RUNS, "w"), indent=0)
    # queue not-yet-OCR'd pages for the grow loop
    if missing:
        with open(GROWQ, "a") as f:
            for pg in missing:
                f.write(f"{saoa}\t{dh}\t{pg}\n")
    return {"ok": True, "old": old, "new": [start, end], "n_members": len(members),
            "missing": missing, "dochash": dh, "saoa": str(saoa)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        saoa, a, b, *rest = sys.argv[1:]
        print(json.dumps(set_bounds(sys.argv[1], [int(sys.argv[2]), int(sys.argv[3])],
                                    int(sys.argv[3]), int(sys.argv[4])), indent=1))
    else:
        print("usage: boundary_apply.py <saoa> <orig_start> <orig_end> <new_end>  (test)")
