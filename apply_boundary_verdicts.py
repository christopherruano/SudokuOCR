#!/usr/bin/env python3
"""Apply AI boundary verdicts DIRECTLY to the runs (bypassing the flaky detector + duplicated corpus).

For each HIGH-confidence 'extend' verdict, extend that run's page-range to include the boundary page.
Members are then rebuilt as the best (most-rows) corpus read for each page in the range. Emits the NEXT
boundary pages (one beyond each freshly-extended end) so the agent loop can Gemini-OCR them and judge
again — the Gemini-OCR -> Opus-judge -> extend -> repeat loop.

  python3 apply_boundary_verdicts.py
Reads export/tableVII_runs.json + export/boundary_verdicts.json; writes the runs back in place and
export/next_boundaries.json (pages to OCR next).
"""
import json, os, re, csv, glob, hashlib, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
VERD = os.path.join(ROOT, "export/boundary_verdicts.json")
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
BG = os.path.join(ROOT, "export/boundary_grow_ocr")


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


def corpus_index():
    """imghash -> best (most-rows) corpus table_id (relative path, no .json)."""
    best = {}
    for f in glob.glob(os.path.join(CORPUS, "*", "*_oneshot.json")):
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(f))
        if not m:
            continue
        ih = m.group(1)
        try:
            nr = sum(len(s.get("rows", [])) for s in (json.load(open(f)).get("data", []) or []))
        except Exception:
            nr = 0
        tid = "digitized_tables/" + os.path.relpath(f, CORPUS)[:-5]   # strip .json
        if ih not in best or nr > best[ih][1]:
            best[ih] = (tid, nr)
    return {ih: v[0] for ih, v in best.items()}


def probe_reads():
    """(saoa,page) -> True if the crawl OCR'd it (so we know it's been seen)."""
    s = set()
    for f in glob.glob(os.path.join(BG, "**", "*_oneshot.json"), recursive=True):
        m = re.search(r"(\d+|[0-9a-f]{16})_p(\d+)_oneshot", os.path.basename(f))
        if m:
            s.add((m.group(1), int(m.group(2))))
    return s


def main():
    runs = json.load(open(RUNS))
    verds = json.load(open(VERD))
    vol = vol_map()
    cidx = corpus_index()
    seen = probe_reads()
    bykey = {f"{r['saoa']}:{r['page_range'][0]}-{r['page_range'][1]}": r for r in runs}

    # collect high-confidence extends per run
    ext = collections.defaultdict(lambda: {"before": [], "after": []})
    for v in verds:
        if v["verdict"] == "extend" and v["confidence"] == "high":
            ext[v["run"]][v["side"]].append(int(v["boundary_page"]))

    applied = 0
    next_boundaries = []
    for key, sides in ext.items():
        run = bykey.get(key)
        if not run:
            continue
        s, e = run["page_range"]
        ns = min([s] + sides["before"])
        ne = max([e] + sides["after"])
        if ns == s and ne == e:
            continue
        run["page_range"] = [ns, ne]
        applied += 1
        # remember to OCR the NEXT page beyond each freshly-extended edge
        sa = run["saoa"]
        if sides["before"]:
            nb = ns - 1
            if nb >= 1 and (sa, nb) not in seen:
                next_boundaries.append({"run": f"{sa}:{ns}-{ne}", "saoa": sa, "dochash": run["dochash"], "side": "before", "page": nb})
        if sides["after"]:
            nb = ne + 1
            if (sa, nb) not in seen:
                next_boundaries.append({"run": f"{sa}:{ns}-{ne}", "saoa": sa, "dochash": run["dochash"], "side": "after", "page": nb})

    # rebuild members = best corpus read for each page in every run's (possibly extended) range
    rebuilt = 0
    for run in runs:
        cp = vol.get(run["dochash"])
        if not cp:
            continue
        s, e = run["page_range"]
        members = []
        for pg in range(s, e + 1):
            ih = hashlib.sha1(f"{cp}:{pg}".encode()).hexdigest()
            tid = cidx.get(ih)
            if tid:
                members.append({"table_id": tid, "page": pg})
        if members:
            run["members"] = members
            run["n_pages"] = e - s + 1
            rebuilt += 1

    json.dump(runs, open(RUNS, "w"), indent=0)
    # dedup next_boundaries by (saoa,page)
    nb_seen = {}
    for nb in next_boundaries:
        nb_seen[(nb["saoa"], nb["page"])] = nb
    nb = list(nb_seen.values())
    json.dump(nb, open(os.path.join(ROOT, "export/next_boundaries.json"), "w"), indent=0)
    total = sum(r["n_pages"] for r in runs)
    print(f"applied AI extends to {applied} runs | rebuilt members for {rebuilt} runs")
    print(f"  total run-pages now: {total} | runs: {len(runs)}")
    print(f"  next boundaries to Gemini-OCR (loop round): {len(nb)} pages -> export/next_boundaries.json")


if __name__ == "__main__":
    main()
