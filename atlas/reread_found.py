#!/usr/bin/env python3
"""Re-read, on the pages located by find_pages.py, the tables whose column check on those pages still
failed (out/found_pages_check.jsonl: columns invented / missing / mislabelled). Three independent readings
(Gemini 3.8 Flash -> out/reread/, Gemini 3.1 Pro -> out/reread_pro/, Gemini 3.7 Flash -> out/reread_third/;
credits key only) go through
apply_column_fixes.py's vote and gates like every other re-read. Earlier re-reads of these tables (made
on the wrong page) are moved to out/reread_superseded/.

  python3 atlas/reread_found.py             (structural re-reads)
  python3 atlas/reread_found.py --persons   (Persons-only reading of relinked tables, out/reread_persons_found/)
"""
import os, sys, json, glob, shutil, sqlite3, hashlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g
import reread_columns as R
from verify_found_pages import item, current_checks

STRUCT = {"columns_invented", "columns_missing", "columns_mislabelled", "columns_shifted"}
OUT = os.path.join(HERE, "out")
DENORM = ["year", "decade", "canonical_name", "entity_type", "parent_geo", "broad_region", "geo_level", "canonical_topic",
          "subject", "measure_class", "additive", "table_family", "measure_category", "source_image", "title"]


def main():
    from google import genai
    con = sqlite3.connect(f"file:{g.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    fp = {json.loads(l)["table_id"]: json.loads(l) for l in open(os.path.join(OUT, "found_pages.jsonl"))}
    persons = "--persons" in sys.argv       # targeted Persons-only reading of relinked tables lacking Persons
    chk = [d for d in current_checks().values() if d["verdict"] in ({"persons_only_missing"} if persons else STRUCT)]
    page_of = {}
    for f in glob.glob(os.path.join(HERE, "..", "census_site", "tables", "*.html")):
        page_of[os.path.basename(f)[:-5].rsplit("_", 1)[1]] = os.path.basename(f)[:-5]
    jobs = []
    for d in chk:
        tid = d["table_id"]; page = page_of.get(hashlib.md5(tid.encode()).hexdigest()[:7])
        if not page:
            continue
        it = item(con, fp[tid]); it["page"] = page
        old = [dict(r) for r in con.execute("select * from facts where table_id=?", (tid,))]
        it["row_labels"] = [f["row_label_std"] for f in old]
        t = dict(con.execute("select * from tables where table_id=?", (tid,)).fetchone())
        jobs.append((it, d, old, {k: t.get(k, "") for k in DENORM}, t["status"], t["measure_class"]))
    if persons:
        R.PERSONS_ONLY = R.SECOND = True; R.MODEL = "gemini-3.8-flash"; R.OUTD = os.path.join(OUT, "reread_persons_found")
        os.makedirs(R.OUTD, exist_ok=True)
        client = genai.Client(api_key=g.env_key()); c = collections.Counter()

        def pjob(j):
            it, d, old, dn, st, ms = j
            path = os.path.join(R.OUTD, it["page"] + ".json")
            if os.path.exists(path) and json.load(open(path)).get("found_pages", {}).get("pages") == fp[it["table_id"]]["found_pages"]:
                return json.load(open(path))
            res = R.process(client, it, d, None, old, dn, st, ms)
            res["found_pages"] = {"pdf": fp[it["table_id"]]["pdf"], "pages": fp[it["table_id"]]["found_pages"]}
            R._dump(res, path); return res
        with ThreadPoolExecutor(16) as ex:
            for fu in as_completed([ex.submit(pjob, j) for j in jobs]):
                c["read" if fu.result().get("read") else "error"] += 1
        print(len(jobs), "Persons-only readings:", dict(c), "\nREREAD_FOUND_DONE", flush=True)
        return
    sup = os.path.join(OUT, "reread_superseded"); os.makedirs(sup, exist_ok=True)
    for it, *_ in jobs:                      # earlier readings were made on the wrong page
        for sub in ("reread", "reread_pro", "reread_third", "reread_persons"):
            p = os.path.join(OUT, sub, it["page"] + ".json")
            q = json.load(open(p)) if os.path.exists(p) else None
            if q is not None and (q.get("found_pages") or {}).get("pages") != fp[it["table_id"]]["found_pages"]:
                os.makedirs(os.path.join(sup, sub), exist_ok=True); shutil.move(p, os.path.join(sup, sub, it["page"] + ".json"))
    print(f"{len(jobs)} tables to re-read on their found pages", flush=True)
    client = genai.Client(api_key=g.env_key())
    for model, outd, second in (("gemini-3.8-flash", "reread", False), ("gemini-3.1-pro-preview", "reread_pro", True),
                                 ("gemini-3.7-flash", "reread_third", True)):
        R.MODEL, R.SECOND, R.OUTD = model, second, os.path.join(OUT, outd)

        def job(j):
            it, d, old, dn, st, ms = j
            path = os.path.join(R.OUTD, it["page"] + ".json")
            if os.path.exists(path):
                old_res = json.load(open(path))
                if (old_res.get("found_pages") or {}).get("pages") == fp[it["table_id"]]["found_pages"]:
                    return old_res
            res = R.process(client, it, d, None, old, dn, st, ms)
            res["found_pages"] = {"pdf": fp[it["table_id"]]["pdf"], "pages": fp[it["table_id"]]["found_pages"]}
            R._dump(res, path)
            return res
        c = collections.Counter()
        with ThreadPoolExecutor(16) as ex:
            for fu in as_completed([ex.submit(job, j) for j in jobs]):
                c[fu.result().get("decision")] += 1
        print(model, dict(c), flush=True)
    print("REREAD_FOUND_DONE", flush=True)


if __name__ == "__main__":
    main()
