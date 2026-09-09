#!/usr/bin/env python3
"""
build_documents_ocr.py — drainer for the /documents portal's OCR queue.

When the human confirms a document's Table VII page range in /documents ("Confirm range &
queue OCR"), the range is appended to export/documents_ocr_queue.jsonl. This drainer renders
those pages from the drive and OCRs them with Gemini (the proven memory-safe flash-drive path),
then incorporates the results into the corpus so they appear in /documents as OCR'd sub-areas.

Single-instance (lockfile) so rapid confirms just append and get swept by the running batch.
Memory-safe: NKEYS/WORKERS overridable, defaults low.

  python3 build_documents_ocr.py            # drain the whole queue once
"""
import os, sys, json, glob, subprocess, hashlib, csv

ROOT = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/Volumes/Expansion/census_pdfs"
QUEUE = os.path.join(ROOT, "export/documents_ocr_queue.jsonl")
DONE = os.path.join(ROOT, "export/documents_ocr_queue.done.jsonl")
RENDERS = os.path.join(ROOT, "export/documents_ocr_renders")
RESULTS = os.path.join(ROOT, "export/documents_ocr_results")
LIST = os.path.join(ROOT, "export/documents_ocr_list.txt")
LOCK = os.path.join(ROOT, "export/documents_ocr.lock")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def cluster_path_of():
    """saoa -> cluster_path (for the corpus dochash), from cc_volumes."""
    cp = {}
    p = os.path.join(V1, "cc_volumes.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r.get("cluster_path"):
                cp[r["saoa_id"]] = r["cluster_path"]
    return cp


def already_ocrd(saoa, pg, cp):
    """True if this page already has a corpus read (imghash = sha1(cluster_path:page))."""
    c = cp.get(saoa)
    if not c:
        return False
    ih = hashlib.sha1(f"{c}:{pg}".encode()).hexdigest()
    return bool(glob.glob(os.path.join(CORPUS, "*", ih + "_oneshot.json")))


def render(path, pg, out):
    import fitz
    from PIL import Image
    if not os.path.exists(path):
        hits = glob.glob(os.path.join(DRIVE, "*", os.path.basename(path)))
        path = hits[0] if hits else None
    if not path:
        return False
    doc = fitz.open(path)
    ok = False
    if 1 <= pg <= doc.page_count:
        pix = doc.load_page(pg - 1).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csGRAY)
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(out, "JPEG", quality=90)
        ok = True
    doc.close()
    return ok


def main():
    if not os.path.exists(QUEUE):
        print("no queue"); return
    if not os.path.isdir(DRIVE):
        # drive unmounted → do NOT drain (would archive un-OCR'd ranges). Leave queue for retry.
        print("documents_ocr: DRIVE not mounted — leaving queue intact; remount and it will resume."); return
    if os.path.exists(LOCK):
        print("drainer already running (lock present) — new pages will be swept by it"); return
    open(LOCK, "w").write(str(os.getpid()))
    try:
        cp = cluster_path_of()
        entries = [json.loads(l) for l in open(QUEUE) if l.strip()]
        # aggregate pages per (saoa, path), dedup, skip already-OCR'd
        want = {}
        for e in entries:
            key = (e["saoa"], e.get("path", ""))
            want.setdefault(key, set()).update(e.get("pages") or range(e["start"], e["end"] + 1))
        os.makedirs(RENDERS, exist_ok=True)
        paths = []
        for (saoa, path), pages in want.items():
            for pg in sorted(pages):
                if already_ocrd(saoa, pg, cp):
                    continue
                out = os.path.join(RENDERS, f"{saoa}_p{pg}.jpg")
                if os.path.exists(out) or render(path, pg, out):
                    paths.append(out)
        if not paths:
            print("nothing to render (all already OCR'd?)")
            _archive(entries)
            return
        open(LIST, "w").write("\n".join(paths) + "\n")
        print(f"documents_ocr: rendered {len(paths)} page(s) -> OCR (Gemini, memory-safe)")
        env = dict(os.environ, LIST=LIST, RESULTS=RESULTS,
                   NKEYS=os.environ.get("NKEYS", "6"), WORKERS=os.environ.get("WORKERS", "2"),
                   MAXCALLS=os.environ.get("MAXCALLS", "12"))
        subprocess.run(["bash", os.path.join(ROOT, "run_flashdrive_ocr.sh")], env=env, cwd=ROOT)
        # incorporate into the corpus (adapter reads export/documents_ocr_results)
        subprocess.run(["python3", os.path.join(ROOT, "build_money_corpus_adapter.py")], cwd=ROOT)
        placed = sum(1 for p in paths if glob.glob(os.path.join(
            RESULTS, "**", os.path.basename(p).replace(".jpg", "_oneshot.json")), recursive=True))
        print(f"documents_ocr: OCR complete, {placed}/{len(paths)} results; incorporated into corpus.")
        _archive(entries)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


def _archive(entries):
    """Move drained entries to the .done ledger and clear the live queue."""
    with open(DONE, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    if os.path.exists(QUEUE):
        os.remove(QUEUE)


if __name__ == "__main__":
    main()
