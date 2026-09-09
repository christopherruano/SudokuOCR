#!/usr/bin/env python3
"""Parallel grow-OCR. Renders ALL missing pages of the queued runs and OCRs them in ONE controlled batch
using all 17 keys (run_flashdrive_ocr.sh, NKEYS=17) — NOT a process-per-page (that was the runaway).
Places results into the canonical corpus, rebuilds members. Single lock: only one batch at a time; rapid
include clicks just append to export/grow_ocr_pending.txt and get swept into the running batch.
  python3 grow_batch.py [<saoa> ...]    # plus whatever is queued in export/grow_ocr_pending.txt
"""
import sys, os, json, glob, shutil, subprocess
_argv = sys.argv
sys.argv = ["x"]
import grow_ocr as G, boundary_apply as B, census_review_app as A  # reuse page_rec/render/rebuild/lock
sys.argv = _argv

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
RENDERS = os.path.join(ROOT, "export/grow_renders")
RESULTS = os.path.join(ROOT, "export/grow_ocr_results")
LIST = os.path.join(ROOT, "export/grow_render_list.txt")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
PROG = os.path.join(ROOT, "export/grow_progress.json")


def collect(saoas):
    runs = json.load(open(RUNS))
    targets, affected = {}, set()
    for sa in saoas:
        for run in [r for r in runs if str(r["saoa"]) == str(sa)]:   # ALL runs sharing this saoa
            dh = run["dochash"]; s, e = run["page_range"]; tp = set(run.get("title_pages", []))
            for pg in range(s, e + 1):
                if pg in tp:
                    continue
                m, has = B.member_for(dh, pg)
                if has:
                    continue                              # already OCR'd
                ih, td, src = G.page_rec(dh, pg)
                if ih and td and src:
                    targets[ih] = (dh, pg, td, src)
                    affected.add(str(sa))
    return targets, affected


def run_batch(saoas):
    targets, affected = collect(saoas)
    if not targets:
        return 0, affected
    os.makedirs(RENDERS, exist_ok=True)
    paths = []
    for ih, (dh, pg, td, src) in targets.items():
        out = os.path.join(RENDERS, ih + ".jpg")
        if os.path.exists(out) or G.render(src, pg, out):
            paths.append(out)
    open(LIST, "w").write("\n".join(paths) + "\n")
    json.dump({"state": "running", "total": len(paths), "placed": 0, "affected": len(affected)}, open(PROG, "w"))
    print(f"grow_batch: {len(paths)} page(s) rendered across {len(affected)} run(s) -> OCR with 17 keys")
    env = dict(os.environ, LIST=LIST, RESULTS=RESULTS, NKEYS="17", WORKERS="2", MAXCALLS="12")
    subprocess.run(["bash", os.path.join(ROOT, "run_flashdrive_ocr.sh")], env=env, cwd=ROOT)
    placed = 0
    for ih, (dh, pg, td, src) in targets.items():
        res = glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)
        if res:
            canon = os.path.join(CORPUS, td, ih + "_oneshot.json")
            os.makedirs(os.path.dirname(canon), exist_ok=True)
            shutil.copy(res[0], canon); placed += 1
    for sa in affected:
        G.rebuild(sa)
    json.dump({"state": "done", "total": len(paths), "placed": placed, "affected": len(affected)}, open(PROG, "w"))
    print(f"grow_batch: placed {placed}/{len(paths)} reads, rebuilt {len(affected)} run(s)")
    return placed, affected


def main():
    if not G.acquire_lock():
        for sa in _argv[1:]:
            open(G.PENDING, "a").write(sa + "\n")
        print("grow_batch: a batch is already running — queued (it'll be swept in)")
        return
    try:
        saoas = set(_argv[1:])
        if os.path.exists(G.PENDING):
            saoas.update(x for x in open(G.PENDING).read().split() if x.strip())
            open(G.PENDING, "w").write("")
        rounds = 0
        while saoas and rounds < 12:
            rounds += 1
            run_batch(list(saoas))
            saoas = set()                                 # pick up anything queued during the batch
            if os.path.exists(G.PENDING):
                saoas.update(x for x in open(G.PENDING).read().split() if x.strip())
                open(G.PENDING, "w").write("")
    finally:
        G.release_lock()


if __name__ == "__main__":
    main()
