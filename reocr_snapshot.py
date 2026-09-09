#!/usr/bin/env python3
"""Snapshot the flagged pages' tables before re-OCR, then diff them after, so the agent can show the
user exactly what the re-OCR changed (cell diffs + constraint-failure count before->after).

  python3 reocr_snapshot.py before                 # -> JSON snapshot of currently-flagged pages (stdout)
  python3 reocr_snapshot.py after  before.json     # -> markdown summary of what changed (stdout)
"""
import sys, os, json, glob, re

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.argv_backup = sys.argv
sys.argv = ["x"]
import constraint_guided_reocr as C   # page_fail_rows, row_label, check_constraints via census_review_app
import census_review_app as A
sys.argv = sys.argv_backup

FLAGS = os.path.join(ROOT, "export/reocr_page_flags.txt")
FLAGS_DONE = FLAGS + ".done"
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def find_json(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


def cells(d):
    out = {}
    for s in (d or {}).get("data", []):
        for row in s.get("rows", []):
            rh = next((k for k in row if not isinstance(row.get(k), dict)), None)
            lbl = str(row.get(rh, "")) if rh else "?"
            for g, sub in row.items():
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        out[f"{lbl}|{g}|{k}"] = v
    return out


def snap_one(ih):
    jf = find_json(ih)
    if not jf:
        return {"imghash": ih, "missing": True}
    d = json.load(open(jf))
    nf, nrows = C.page_fail_rows(d)
    return {"imghash": ih, "title": ((d.get("metadata", {}) or {}).get("title", "") or "")[:60],
            "fail_rows": nf, "n_rows": nrows, "cells": cells(d)}


def read_flags():
    for p in (FLAGS, FLAGS_DONE):
        if os.path.exists(p):
            ihs = [x for x in open(p).read().split() if re.fullmatch(r"[0-9a-f]{40}", x)]
            if ihs:
                return ihs
    return []


def do_before():
    ihs = read_flags()
    print(json.dumps({"pages": [snap_one(ih) for ih in ihs]}))


def do_after(beforefile):
    before = json.load(open(beforefile)).get("pages", [])
    lines = []
    improved = same = worse = 0
    for b in before:
        ih = b["imghash"]
        a = snap_one(ih)
        if a.get("missing"):
            lines.append(f"- `{ih[:12]}` — page not found after re-OCR")
            continue
        ob, nb = b.get("fail_rows", 0), a.get("fail_rows", 0)
        oc, nc = b.get("cells", {}), a.get("cells", {})
        changed = [(k, oc.get(k), nc.get(k)) for k in nc if k in oc and oc[k] != nc[k]]
        tag = "✅ improved" if nb < ob else ("— no change" if (nb == ob and not changed) else ("⚠ changed, fails same" if nb == ob else "❗ worse"))
        if nb < ob:
            improved += 1
        elif nb > ob:
            worse += 1
        else:
            same += 1
        lines.append(f"- `{ih[:12]}` **{a['title']}** — constraint-fails {ob}→{nb}  {tag}"
                     + (f"  ·  {len(changed)} cell(s) changed" if changed else ""))
        for k, ov, nv in changed[:6]:
            lbl, g, sk = k.split("|")
            lines.append(f"    - {lbl} · {g}/{sk}: {ov} → {nv}")
    head = f"### Re-OCR batch results — {len(before)} page(s):  {improved} improved · {same} unchanged · {worse} worse\n"
    print(head + "\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "before":
        do_before()
    elif len(sys.argv) >= 3 and sys.argv[1] == "after":
        do_after(sys.argv[2])
    else:
        print(__doc__)
