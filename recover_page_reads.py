#!/usr/bin/env python3
"""The Haiku page-number agents echoed a mangled table_id (Haiku can't reproduce a
130-char path). Recover the TRUE table_id for every read from the image the agent
actually opened (Read tool calls), which is 1:1 with a crop via crops.json.

For each agent transcript: parse its assigned crop index range, its ordered Read
img paths, and its ordered StructuredOutput reads. Pair read[k] <-> img[k] (the
ground truth the model looked at); fall back to positional crops[lo+k] when the
Read list and reads list don't line up. Write page_reads_fixed.json keyed by the
true crop table_id.
"""
import json, re, glob, os, sys
from collections import defaultdict

ROOT = "/Users/chrisruano/Research/SudokuOCR"
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
WF = ("/Users/chrisruano/.claude/projects/-Users-chrisruano-Research-SudokuOCR/"
      "57ead6ca-9002-4c0d-8099-80c8ad9e4ed7/subagents/workflows/wf_a87bd7a2-a73")

crops = json.load(open(os.path.join(V1, "crops.json")))
img2crop = {c["img"]: c for c in crops}          # local img path -> crop (1:1)

RANGE = re.compile(r"Process entries index (\d+)\.\.(\d+)")


def walk(o, imgs, box):
    if isinstance(o, dict):
        if o.get("type") == "tool_use":
            nm, inp = o.get("name"), o.get("input", {})
            if nm == "Read" and isinstance(inp, dict):
                fp = inp.get("file_path", "")
                if fp.endswith((".png", ".jpg", ".jpeg")):
                    imgs.append(fp)
            if nm in ("StructuredOutput", "structured_output") and isinstance(inp, dict) and "reads" in inp:
                box["reads"] = inp["reads"]
        for v in o.values():
            walk(v, imgs, box)
    elif isinstance(o, list):
        for v in o:
            walk(v, imgs, box)


def parse_agent(path):
    idxrange = None
    imgs = []
    box = {}
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if idxrange is None:
            m = RANGE.search(json.dumps(e.get("message", e)) if "message" in e else json.dumps(e))
            if m:
                idxrange = (int(m.group(1)), int(m.group(2)))
        walk(e, imgs, box)
    return idxrange, imgs, box.get("reads")


def main():
    out = {}           # true table_id -> read fields
    stats = defaultdict(int)
    n_agents = 0
    for path in glob.glob(os.path.join(WF, "agent-*.jsonl")):
        rng, imgs, reads = parse_agent(path)
        if not reads:
            stats["agent_no_reads"] += 1
            continue
        if rng is None:
            stats["agent_no_range"] += 1
            continue
        n_agents += 1
        lo, hi = rng
        span = hi - lo
        exact = len(reads) == span        # agent emitted one read per assigned crop, in order
        for k, rd in enumerate(reads):
            if lo + k >= len(crops):
                stats["out_of_range"] += 1
                continue
            crop = crops[lo + k]           # positional: disjoint ranges => unique crop per read
            stats["exact_pos" if exact else "ragged_pos"] += 1
            tid = crop["table_id"]
            # confidence check: did the k-th Read hit this crop's image?
            img_ok = k < len(imgs) and imgs[k] == crop["img"]
            cand = {"table_id": tid, "dochash": crop["dochash"], "seq": crop.get("seq", ""),
                    "family": crop.get("family", ""), "page_known": crop.get("page_known", ""),
                    "page_number": rd.get("page_number", ""), "page_visible": bool(rd.get("page_visible")),
                    "table_number": rd.get("table_number", ""), "section": rd.get("section", ""),
                    "confidence": rd.get("confidence", ""), "exact": exact, "img_ok": img_ok}
            prev = out.get(tid)
            if prev is None or (cand["page_visible"] and not prev["page_visible"]):
                out[tid] = cand
    reads_out = list(out.values())
    json.dump(reads_out, open(os.path.join(V1, "page_reads_fixed.json"), "w"))
    vis = sum(1 for r in reads_out if r["page_visible"])
    numeric = sum(1 for r in reads_out if str(r["page_number"]).isdigit())
    img_ok = sum(1 for r in reads_out if r.get("img_ok"))
    print(f"agents parsed: {n_agents}   mapping: {dict(stats)}")
    print(f"recovered reads (distinct true crops): {len(reads_out)} / {len(crops)} crops")
    print(f"  page_visible: {vis}   numeric page#: {numeric}")
    print(f"  img-path verified (k-th Read == k-th crop): {img_ok} / {len(reads_out)}")
    print("wrote page_reads_fixed.json")


if __name__ == "__main__":
    main()
