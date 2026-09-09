#!/usr/bin/env python3
"""CONTINUOUS watcher: keeps every USER-CONFIRMED /documents range fully OCR'd, re-reading the verdicts
file each round so ranges confirmed WHILE it runs are picked up automatically. Per round:
  1. re-derive missing pages from export/review_verdicts/documents.json (the live file)
  2. resolve a render for each: staged documents_ocr_renders/<saoa>_p<pg>.jpg, else render from the drive
  3. one memory-safe 17-key batch; place into canonical corpus; rebuild affected docs
  4. sleep and repeat. Exits after MAX_ROUNDS or 3 consecutive all-clear rounds.
Safe across drive drops (pages that can't render just wait for the next round).
  python3 ocr_confirmed_watch.py
"""
import sys, os, json, csv, glob, shutil, subprocess, time
sys.argv = ["x"]
import grow_ocr as G, boundary_apply as B

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(ROOT, "export/documents_ocr_renders")
RENDERS = os.path.join(ROOT, "export/grow_renders")
RESULTS = os.path.join(ROOT, "export/docs_watch_ocr_results")
LIST = os.path.join(ROOT, "export/docs_watch_list.txt")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
MAX_ROUNDS = 60
SLEEP = 150


def ranges_of(v):
    if "ranges" in v:
        return [tuple(int(x) for x in r.split("-")) for r in v["ranges"]]
    if "range" in v:
        return [tuple(v["range"])]
    return []


def missing_pages():
    verd = json.load(open(os.path.join(ROOT, "export/review_verdicts/documents.json")))
    cov = {r["saoa"]: r.get("dochash", "") for r in csv.DictReader(open(os.path.join(ROOT, "export/document_coverage.csv")))}
    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    sa2dh = {str(r["saoa"]): r["dochash"] for r in runs}
    out = []
    for k, v in verd.items():
        sa = k.split(":")[1]
        dh = cov.get(sa) or sa2dh.get(sa)
        if not dh:
            continue
        for s, e in ranges_of(v):
            for pg in range(s, e + 1):
                if not B.member_for(dh, pg)[1]:
                    out.append((sa, dh, pg))
    return out, len(verd)


def resolve_render(sa, dh, pg, ih):
    dst = os.path.join(RENDERS, ih + ".jpg")
    if os.path.exists(dst):
        return dst
    for cand in (f"{sa}_p{pg}.jpg", f"{sa}_{pg}.jpg"):
        p = os.path.join(STAGED, cand)
        if os.path.exists(p):
            shutil.copy(p, dst); return dst
    ih2, td2, src = G.page_rec(dh, pg)
    if src and os.path.isdir("/Volumes/Expansion/census_pdfs") and G.render(src, pg, dst):
        return dst
    return None


def main():
    os.makedirs(RENDERS, exist_ok=True)
    allclear = 0
    for rnd in range(1, MAX_ROUNDS + 1):
        miss, nverd = missing_pages()
        stamp = time.strftime("%H:%M")
        if not miss:
            allclear += 1
            print(f"[{stamp}] round {rnd}: {nverd} confirmed docs — ALL PAGES OCR'D ✅ (all-clear {allclear}/3)")
            if allclear >= 3:
                print("stable — exiting"); return
            time.sleep(SLEEP); continue
        allclear = 0
        if not G.acquire_lock():
            print(f"[{stamp}] round {rnd}: another OCR batch holds the lock — waiting"); time.sleep(SLEEP); continue
        try:
            targets = {}
            unrenderable = 0
            for sa, dh, pg in miss:
                ih, td, _src = G.page_rec(dh, pg)
                if not ih:
                    unrenderable += 1; continue
                rp = resolve_render(sa, dh, pg, ih)
                if rp:
                    targets[ih] = (sa, dh, pg, td)
                else:
                    unrenderable += 1
            drive = "mounted" if os.path.isdir("/Volumes/Expansion/census_pdfs") else "NOT mounted"
            print(f"[{stamp}] round {rnd}: {nverd} docs · missing {len(miss)} · renderable {len(targets)} · "
                  f"waiting-on-drive/rec {unrenderable} · drive {drive}")
            if not targets:
                time.sleep(SLEEP); continue
            open(LIST, "w").write("\n".join(os.path.join(RENDERS, ih + ".jpg") for ih in targets) + "\n")
            env = dict(os.environ, LIST=LIST, RESULTS=RESULTS, NKEYS="17", WORKERS="2", MAXCALLS="12")
            subprocess.run(["bash", os.path.join(ROOT, "run_flashdrive_ocr.sh")], env=env, cwd=ROOT)
            placed = 0; affected = set()
            for ih, (sa, dh, pg, td) in targets.items():
                res = glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)
                if not res:
                    continue
                canon = os.path.join(CORPUS, td, ih + "_oneshot.json")
                os.makedirs(os.path.dirname(canon), exist_ok=True)
                shutil.copy(res[0], canon); placed += 1; affected.add(sa)
            for sa in affected:
                try:
                    G.rebuild(sa)
                except Exception as exc:
                    print("  rebuild", sa, "failed:", exc)
            print(f"[{time.strftime('%H:%M')}] round {rnd}: placed {placed}/{len(targets)} · rebuilt {len(affected)} doc(s)")
        finally:
            G.release_lock()
        time.sleep(30)
    print("MAX_ROUNDS reached — relaunch to continue")


if __name__ == "__main__":
    main()
