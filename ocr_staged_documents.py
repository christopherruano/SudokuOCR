#!/usr/bin/env python3
"""OCR the not-yet-OCR'd pages inside the USER-CONFIRMED /documents ranges, using renders already staged
in export/documents_ocr_renders/ (<saoa>_p<page>.jpg) — works with the drive UNMOUNTED. Maps each page to
its canonical corpus identity via grow_ocr.page_rec (imghash + table_dir), runs one memory-safe 17-key
batch, places results, rebuilds affected runs. Pages with no staged render are listed for the drive pass.
  python3 ocr_staged_documents.py
"""
import sys, os, json, csv, glob, shutil, subprocess
sys.argv = ["x"]
import grow_ocr as G, boundary_apply as B

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(ROOT, "export/documents_ocr_renders")
RENDERS = os.path.join(ROOT, "export/grow_renders")
RESULTS = os.path.join(ROOT, "export/docs_staged_ocr_results")
LIST = os.path.join(ROOT, "export/docs_staged_list.txt")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def ranges_of(v):
    if "ranges" in v:
        return [tuple(int(x) for x in r.split("-")) for r in v["ranges"]]
    if "range" in v:
        return [tuple(v["range"])]
    return []


def main():
    if not G.acquire_lock():
        print("another grow/OCR batch is running — try again after it finishes"); return
    try:
        verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
        cov = {r["saoa"]: r.get("dochash", "") for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv")))}
        runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
        sa2dh = {str(r["saoa"]): r["dochash"] for r in runs}
        os.makedirs(RENDERS, exist_ok=True)
        targets = {}                                   # ih -> (saoa, dh, pg, td)
        need_drive = []
        for k, v in verd.items():
            sa = k.split(":")[1]
            dh = cov.get(sa) or sa2dh.get(sa)
            if not dh:
                continue
            for s, e in ranges_of(v):
                for pg in range(s, e + 1):
                    m, has = B.member_for(dh, pg)
                    if has:
                        continue
                    staged = next((p for p in (os.path.join(STAGED, f"{sa}_p{pg}.jpg"),
                                               os.path.join(STAGED, f"{sa}_{pg}.jpg")) if os.path.exists(p)), None)
                    if not staged:
                        need_drive.append(f"{sa}\t{dh}\t{pg}")
                        continue
                    ih, td, _src = G.page_rec(dh, pg)
                    if not ih:
                        need_drive.append(f"{sa}\t{dh}\t{pg}\t(no page_rec)")
                        continue
                    dst = os.path.join(RENDERS, ih + ".jpg")
                    if not os.path.exists(dst):
                        shutil.copy(staged, dst)
                    targets[ih] = (sa, dh, pg, td)
        open(os.path.join(ROOT, "export/docs_need_drive.txt"), "w").write("\n".join(need_drive) + "\n")
        print(f"OCR-able from staged renders: {len(targets)} · need drive: {len(need_drive)} (-> export/docs_need_drive.txt)")
        if not targets:
            return
        open(LIST, "w").write("\n".join(os.path.join(RENDERS, ih + ".jpg") for ih in targets) + "\n")
        env = dict(os.environ, LIST=LIST, RESULTS=RESULTS, NKEYS="17", WORKERS="2", MAXCALLS="12")
        subprocess.run(["bash", os.path.join(ROOT, "run_flashdrive_ocr.sh")], env=env, cwd=ROOT)
        placed = 0
        affected = set()
        for ih, (sa, dh, pg, td) in targets.items():
            res = glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)
            if not res:
                continue
            canon = os.path.join(CORPUS, td, ih + "_oneshot.json")
            os.makedirs(os.path.dirname(canon), exist_ok=True)
            shutil.copy(res[0], canon)
            placed += 1
            affected.add(sa)
        for sa in affected:
            try:
                G.rebuild(sa)
            except Exception as exc:
                print("  rebuild", sa, "failed:", exc)
        print(f"placed {placed}/{len(targets)} reads · rebuilt {len(affected)} doc(s)")
    finally:
        G.release_lock()


if __name__ == "__main__":
    main()
