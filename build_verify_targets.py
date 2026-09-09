#!/usr/bin/env python3
"""Build the spot-verification worklist: the boundaries that CHANGED in this remediation, each pointing
at a real cached scan (census_site/img/<imghash>.jpg) for an agent to read. Targets:
  - orphan_assembled runs: first AND last page (confirm the whole new run is one Age/Sex/Civil-Condition
    table, not a false-positive literacy/migration table)
  - split_mixed_family runs that are VII or VIII: first AND last page (confirm we cut at the right place)
  - backward_bridged runs: the new first page (confirm it really continues that table backward)
Writes export/verify_targets.json
"""
import json, csv, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
MASTER = os.path.join(V1, "page_number_status.csv")
IMG = os.path.join(ROOT, "census_site/img")
TAB = re.compile(r"TABLE\s+(?:No\.?\s*)?((?:[A-Z]-)?[IVXLC]+|[A-Z])\b", re.I)


def family(t):
    t = re.sub(r"[-.\s]*(cont(?:d|inued)?|concl(?:d|uded)?)\.?", " ", t or "", flags=re.I)
    m = TAB.search(t)
    return m.group(1).upper() if m else ""


def main():
    title, ih_of = {}, {}
    for r in csv.DictReader(open(MASTER)):
        dh, pg = r.get("dochash", ""), r.get("pdf_page", "")
        if dh and str(pg).isdigit():
            k = (dh, int(pg))
            t = r.get("title", "")
            if k not in title or len(t) > len(title[k]):
                title[k] = t
            if r.get("imghash") and k not in ih_of:
                ih_of[k] = r["imghash"]

    present = set(os.path.splitext(x)[0] for x in os.listdir(IMG)) if os.path.isdir(IMG) else set()
    runs = json.load(open(RUNS))
    targets = []

    def add(run, page, role):
        ih = ih_of.get((run["dochash"], page))
        if not ih or ih not in present:
            return
        targets.append({
            "run": f"{run['saoa']}:{run['page_range'][0]}-{run['page_range'][1]}",
            "origin": run.get("origin", "original"), "role": role, "page": page,
            "imghash": ih, "scan": f"census_site/img/{ih}.jpg",
            "master_title": title.get((run["dochash"], page), "")[:60],
            "claimed_family": run.get("tnum", "VII"),
        })

    for run in runs:
        origin = run.get("origin", "original")
        s, e = run["page_range"]
        if origin == "orphan_assembled":
            add(run, s, "orphan_first"); add(run, e, "orphan_last")
        elif origin == "split_mixed_family":
            fam = family(title.get((run["dochash"], s), ""))
            if fam in ("VII", "VIII"):
                add(run, s, "split_first"); add(run, e, "split_last")
        elif origin == "backward_bridged":
            add(run, s, "bridge_newstart")

    json.dump(targets, open(os.path.join(ROOT, "export/verify_targets.json"), "w"), indent=1)
    from collections import Counter
    print(f"wrote {len(targets)} verification targets -> export/verify_targets.json")
    print("  by role:", dict(Counter(t["role"] for t in targets)))
    print("  distinct runs:", len(set(t["run"] for t in targets)))


if __name__ == "__main__":
    main()
