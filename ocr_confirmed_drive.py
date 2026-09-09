#!/usr/bin/env python3
"""Render + OCR the remaining not-yet-OCR'd pages of the USER-CONFIRMED /documents ranges from the drive
(export/docs_need_drive.txt: saoa<TAB>dochash<TAB>page). ONLY confirmed ranges — no predictions. One
memory-safe 17-key batch; places into the canonical corpus; rebuilds affected docs' runs.
  python3 ocr_confirmed_drive.py
"""
import sys, os, json, glob, shutil, subprocess
sys.argv = ["x"]
import grow_ocr as G, boundary_apply as B

ROOT = os.path.dirname(os.path.abspath(__file__))
RENDERS = os.path.join(ROOT, "export/grow_renders")
RESULTS = os.path.join(ROOT, "export/docs_drive_ocr_results")
LIST = os.path.join(ROOT, "export/docs_drive_list.txt")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def main():
    if not G.acquire_lock():
        print("grow lock busy — run again when the current batch finishes"); return
    try:
        items = []
        for line in open(os.path.join(ROOT, "export/docs_need_drive.txt")):
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2].isdigit():
                items.append((parts[0], parts[1], int(parts[2])))
        os.makedirs(RENDERS, exist_ok=True)
        targets = {}
        skipped = norec = norender = 0
        for sa, dh, pg in items:
            m, has = B.member_for(dh, pg)
            if has:
                skipped += 1; continue                    # already landed via the staged batch
            ih, td, src = G.page_rec(dh, pg)
            if not ih or not src:
                norec += 1; continue
            out = os.path.join(RENDERS, ih + ".jpg")
            if not (os.path.exists(out) or G.render(src, pg, out)):
                norender += 1; continue
            targets[ih] = (sa, dh, pg, td)
        print(f"drive pass: {len(items)} listed · already-done {skipped} · no-record {norec} · "
              f"render-failed {norender} · to OCR {len(targets)}")
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
            placed += 1; affected.add(sa)
        for sa in affected:
            try:
                G.rebuild(sa)
            except Exception as exc:
                print("  rebuild", sa, "failed:", exc)
        print(f"placed {placed}/{len(targets)} · rebuilt {len(affected)} doc(s)")
    finally:
        G.release_lock()


if __name__ == "__main__":
    main()
