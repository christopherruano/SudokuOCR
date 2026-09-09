#!/usr/bin/env python3
"""Re-OCR the 'failed_read' empty pages found inside the incomplete Table VII runs (real VII pages that
came back with 0 rows). Renders each page fresh from the drive at 300 DPI, OCRs with all 17 Gemini keys
via run_flashdrive_ocr.sh, and overwrites the empty canonical read ONLY when the new read actually has
rows (the empty original is backed up to <ih>_oneshot.empty.bak). Then rebuilds affected runs.
  python3 reocr_empty.py            # uses export/incomplete_worklist.json (class==failed_read)
"""
import sys, os, json, glob, shutil, subprocess, re
_argv = sys.argv
sys.argv = ["x"]
import grow_ocr as G, boundary_apply as B
sys.argv = _argv

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "export/incomplete_worklist.json")
RENDERS = os.path.join(ROOT, "export/reocr_empty_renders")
RESULTS = os.path.join(ROOT, "export/reocr_empty_results")
LIST = os.path.join(ROOT, "export/reocr_empty_list.txt")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
PROG = os.path.join(ROOT, "export/reocr_empty_progress.json")


def nrows(d):
    return sum(len(s.get("rows", [])) for s in (d or {}).get("data", []))


def main():
    work = [w for w in json.load(open(WORK)) if w["class"] == "failed_read"]
    os.makedirs(RENDERS, exist_ok=True)
    # 1) render every page fresh from the drive
    rendered = []
    for w in work:
        ih = w["imghash"]
        src = w["source_pdf"] or B.source_pdf_for(w["dochash"])
        if not src:
            continue
        out = os.path.join(RENDERS, ih + ".jpg")
        if os.path.exists(out) or G.render(src, int(w["page"]), out):
            rendered.append((ih, w))
    open(LIST, "w").write("\n".join(os.path.join(RENDERS, ih + ".jpg") for ih, _ in rendered) + "\n")
    json.dump({"state": "ocr", "total": len(rendered), "recovered": 0}, open(PROG, "w"))
    print(f"reocr_empty: {len(rendered)}/{len(work)} pages rendered -> OCR with 17 keys")
    # 2) OCR them all in one memory-safe batch
    env = dict(os.environ, LIST=LIST, RESULTS=RESULTS, NKEYS="17", WORKERS="2", MAXCALLS="12")
    subprocess.run(["bash", os.path.join(ROOT, "run_flashdrive_ocr.sh")], env=env, cwd=ROOT)
    # 3) place ONLY improvements (new read has rows); back up the empty original
    recovered, still_empty, affected = 0, 0, set()
    for ih, w in rendered:
        res = glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)
        if not res:
            still_empty += 1; continue
        try:
            nd = json.load(open(res[0]))
        except Exception:
            still_empty += 1; continue
        if nrows(nd) <= 0:
            still_empty += 1; continue                 # came back empty again -> leave original, will go to Mistral
        canon = os.path.join(CORPUS, w["table_dir"], ih + "_oneshot.json")
        if os.path.exists(canon):
            shutil.copy(canon, canon.replace("_oneshot.json", "_oneshot.empty.bak"))
        os.makedirs(os.path.dirname(canon), exist_ok=True)
        shutil.copy(res[0], canon)
        recovered += 1; affected.add(w["saoa"])
    # 4) rebuild affected runs so the recovered rows show up
    for sa in affected:
        try:
            G.rebuild(sa)
        except Exception as e:
            print("  rebuild", sa, "failed:", e)
    json.dump({"state": "done", "total": len(rendered), "recovered": recovered,
               "still_empty": still_empty, "affected": len(affected)}, open(PROG, "w"))
    print(f"reocr_empty: recovered {recovered} pages (now have rows), {still_empty} still empty "
          f"(-> Mistral next), rebuilt {len(affected)} run(s)")
    open(os.path.join(ROOT, "export/reocr_empty_still.txt"), "w").write(
        "\n".join(ih for ih, w in rendered
                  if not glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)
                  or nrows(json.load(open(glob.glob(os.path.join(RESULTS, "**", ih + "_oneshot.json"), recursive=True)[0]))) <= 0))


if __name__ == "__main__":
    main()
