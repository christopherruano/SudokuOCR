#!/usr/bin/env python3
"""On-demand OCR for pages pulled into an assembly via the review app's 'include page' button. Renders
each not-yet-OCR'd page of a run from the drive, runs the full Gemini oneshot protocol, writes straight
into the canonical corpus path (digitized_tables/<table_dir>/<imghash>_oneshot.json) so it becomes an
assembly member, then rebuilds the run's members so the pages show up.
  python3 grow_ocr.py <saoa>                  # OCR every missing page in that run (preferred)
  python3 grow_ocr.py <dochash> <page> <saoa> # legacy single-page
"""
import sys, os, re, csv, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DRIVE = "/Volumes/Expansion/census_pdfs"
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
RENDERS = os.path.join(ROOT, "export/grow_renders")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
PROG = os.path.join(ROOT, "export/grow_progress.json")


def save_prog(prog):
    import json
    try:
        json.dump(prog, open(PROG, "w"))
    except Exception:
        pass


def page_rec(dochash, pg):
    # target EXACTLY the crop boundary_apply.member_for resolves; source_pdf from any row for the page.
    import boundary_apply
    m, _ = boundary_apply.member_for(dochash, pg)
    if not m:
        return None, None, None
    mm = re.match(r"digitized_tables/(.+)/([0-9a-f]{40})_oneshot", m["table_id"])
    if not mm:
        return None, None, None
    td, ih = mm.group(1), mm.group(2)
    src = None
    for r in csv.DictReader(open(os.path.join(V1, "page_number_status.csv"))):
        if r["dochash"] == dochash and r["pdf_page"] == str(pg):
            src = r["source_pdf"]; break
    if not src:                                  # un-indexed page -> source PDF is doc-level
        src = boundary_apply.source_pdf_for(dochash)
    return ih, td, src


def render(source_pdf, pg, out):
    import fitz
    from PIL import Image
    hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(source_pdf)))
    if not hits:
        print(f"grow_ocr: PDF not on drive ({os.path.basename(source_pdf)}) — drive mounted?"); return False
    doc = fitz.open(hits[0])
    if not (1 <= pg <= doc.page_count):
        doc.close(); print(f"grow_ocr: page {pg} out of range"); return False
    pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csGRAY)
    Image.frombytes("L", (pix.width, pix.height), pix.samples).save(out, "JPEG", quality=90)
    doc.close(); return True


def ocr_page(dochash, pg):
    ih, td, src = page_rec(dochash, pg)
    if not ih:
        print(f"grow_ocr: no page record for {dochash} p{pg}"); return False
    canon_dir = os.path.join(CORPUS, td)
    canon = os.path.join(canon_dir, f"{ih}_oneshot.json")
    if os.path.exists(canon):
        return True
    os.makedirs(RENDERS, exist_ok=True)
    rpath = os.path.join(RENDERS, f"{ih}.jpg")
    if not (os.path.exists(rpath) or render(src, pg, rpath)):
        return False
    print(f"grow_ocr: OCRing p{pg} ({ih[:10]}) with Gemini protocol …")
    sys.argv = ["x"]
    from oneshot import extract_and_verify
    extract_and_verify(rpath, output_dir=canon_dir)        # writes <ih>_oneshot.json into the canonical dir
    if not os.path.exists(canon):
        for f in glob.glob(os.path.join(canon_dir, f"{ih}*_oneshot.json")):
            os.replace(f, canon); break
    ok = os.path.exists(canon)
    print(f"grow_ocr: p{pg} -> {'wrote ' + os.path.basename(canon) if ok else 'FAILED'}")
    return ok


def rebuild(saoa):
    import json, boundary_apply
    runs = [r for r in json.load(open(RUNS)) if str(r["saoa"]) == str(saoa)]   # ALL runs for this saoa
    if not runs:
        print("grow_ocr: run not found for rebuild"); return
    for run in runs:
        s, e = run["page_range"]
        res = boundary_apply.set_bounds(saoa, [s, e], s, e)
        print(f"grow_ocr: rebuilt {saoa} {[s, e]} -> {res.get('n_members')} members, missing={res.get('missing')}")


LOCKDIR = os.path.join(ROOT, "export/grow_ocr.lock")
PENDING = os.path.join(ROOT, "export/grow_ocr_pending.txt")


def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def acquire_lock():
    """Single grow worker at a time. Returns True if we hold the lock; steals a stale lock (dead PID)."""
    for _ in range(2):
        try:
            os.mkdir(LOCKDIR)
            open(os.path.join(LOCKDIR, "pid"), "w").write(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                pid = int(open(os.path.join(LOCKDIR, "pid")).read().strip())
            except Exception:
                pid = None
            if pid and not _alive(pid):
                try:
                    os.remove(os.path.join(LOCKDIR, "pid")); os.rmdir(LOCKDIR)
                except Exception:
                    pass
                continue
            return False
    return False


def release_lock():
    for p in (os.path.join(LOCKDIR, "pid"), LOCKDIR):
        try:
            (os.remove if os.path.isfile(p) else os.rmdir)(p)
        except Exception:
            pass


def process_saoa(saoa):
    import json, boundary_apply
    run = next((r for r in json.load(open(RUNS)) if str(r["saoa"]) == str(saoa)), None)
    if not run:
        print(f"grow_ocr: no run for saoa {saoa}"); return
    dh = run["dochash"]; s, e = run["page_range"]
    missing = [pg for pg in range(s, e + 1) if not boundary_apply.member_for(dh, pg)[1]]
    print(f"grow_ocr: {saoa} {[s, e]} — {len(missing)} missing page(s): {missing}")
    prog = {"saoa": saoa, "region": run.get("region", ""), "range": [s, e], "state": "running",
            "pages": [{"page": p, "status": "pending"} for p in missing]}
    save_prog(prog)
    for i, pg in enumerate(missing):
        prog["pages"][i]["status"] = "ocring"; save_prog(prog)
        ok = ocr_page(dh, pg)
        prog["pages"][i]["status"] = "done" if ok else "failed"; save_prog(prog)
    prog["state"] = "done"; save_prog(prog)
    rebuild(saoa)


def main():
    if len(sys.argv) == 2:                                  # <saoa>: serialized via lock + pending queue
        saoa = sys.argv[1]
        if not acquire_lock():                             # another worker is running -> queue & exit
            with open(PENDING, "a") as f:
                f.write(saoa + "\n")
            print(f"grow_ocr: worker busy — queued {saoa} (it'll be drained in turn)")
            return
        try:
            todo, done = [saoa], set()
            while todo:
                sa = todo.pop(0)
                if sa in done:
                    continue
                done.add(sa)
                process_saoa(sa)
                if os.path.exists(PENDING):                 # pick up anything queued while we worked
                    nxt = [x for x in open(PENDING).read().split() if x.strip()]
                    open(PENDING, "w").write("")
                    todo += [x for x in nxt if x not in done]
        finally:
            release_lock()
    elif len(sys.argv) == 4:                                # legacy: <dochash> <page> <saoa>
        ocr_page(sys.argv[1], int(sys.argv[2])); rebuild(sys.argv[3])
    else:
        print("usage: grow_ocr.py <saoa>  |  grow_ocr.py <dochash> <page> <saoa>")


if __name__ == "__main__":
    main()
