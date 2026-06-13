#!/usr/bin/env python3
"""Production place-split prep: for every multi-page chop group, list its member
pages in page order with their titles, so an AI can assign each member to its
geographic place (taluk/district), carrying the place across continuation pages
that don't repeat the header. Output: place_split_input.json (per group)."""
import os, re, csv, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def main():
    groups = json.load(open(os.path.join(V1, "chop_groups.json")))
    title = {}
    for r in csv.DictReader(open(os.path.join(V1, "provenance.csv"))):
        title[r["table_id"]] = (r.get("title", "") or "")[:110]
    reads = {r["table_id"]: r for r in json.load(open(os.path.join(V1, "page_reads_fixed.json")))}

    GEO_RE = re.compile(r"\(?\d{0,2}\)?\s*([A-Z][A-Za-z'’]+(?:[ -][A-Z][A-Za-z'’]+){0,2})\s+(?:Taluk|Taluq|District|Distt|State|Division|Tahsil|Zila|City|Municipality)", re.I)
    out = []
    single = 0
    for gi, g in enumerate(groups):
        mem = g.get("member_table_ids", [])
        if len(mem) < 2:
            continue
        seq = []
        for t in mem:
            pg = reads.get(t, {}).get("page_number", "")
            seq.append({"tid": t, "page": int(pg) if str(pg).isdigit() else 0, "title": title.get(t, "")})
        seq.sort(key=lambda x: x["page"])
        places = {GEO_RE.search(s["title"]).group(1).strip().title() for s in seq if GEO_RE.search(s["title"])}
        if len(places) < 2:        # single place (or none detected) -> no AI split needed
            single += 1
            continue
        out.append({"gid": gi, "dochash": g["dochash"], "subject": g.get("subject", "")[:80],
                    "members": [{"i": i, "page": s["page"], "title": s["title"], "tid": s["tid"]}
                                for i, s in enumerate(seq)]})
    json.dump(out, open(os.path.join(V1, "place_split_input.json"), "w"))
    print(f"single-place groups (no split): {single}")
    # split into per-group files for token-efficient agents
    dd = os.path.join(V1, "place_split_docs"); os.makedirs(dd, exist_ok=True)
    for d in out:
        json.dump(d, open(os.path.join(dd, f"g{d['gid']}.json"), "w"))
    print(f"multi-page groups to place-split: {len(out)}")
    print(f"  total member pages: {sum(len(d['members']) for d in out)}")


if __name__ == "__main__":
    main()
