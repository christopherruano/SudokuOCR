#!/usr/bin/env python3
"""TOC + page-number completeness for Table VII: per census volume, compare the
TOC-resolved Table VII page span (manifest queue_row.resolved_start/end_page)
against how many Table VII tables we've actually read. Classify each volume:
  complete / partial(missing N) / no_reads(must OCR) / no_toc_entry(investigate).
Output: tableVII_volume_gaps.csv  (the set to finish, ordered by gap size).
"""
import os, re, json, csv
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DOC = re.compile(r"([0-9a-f]{16})_t")
VII = re.compile(r"\bVII\b|age,?\s*sex|civil\s*condition", re.I)


def main():
    vols = {}
    toc = defaultdict(set)        # dochash -> set of TOC Table VII PDF pages
    for line in open(os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")):
        try:
            j = json.loads(line)
        except Exception:
            continue
        m = DOC.search(j.get("source_image", "") or j.get("result_json", "") or "")
        if not m:
            continue
        dh = m.group(1); qr = j.get("queue_row", {}) or {}
        vols.setdefault(dh, {"pdf": ((j.get("digitized_source_file") or qr.get("filename", "")) or "").split("/")[-1],
                             "year": qr.get("year") or j.get("year", ""), "prov": qr.get("province") or j.get("province", "")})
        title = (qr.get("title", "") or "") + " " + (qr.get("canonical_topic", "") or "")
        if VII.search(title) and isinstance(qr.get("resolved_start_page"), int) and isinstance(qr.get("resolved_end_page"), int):
            toc[dh].update(range(qr["resolved_start_page"], qr["resolved_end_page"] + 1))

    # Table VII reads we possess, per dochash (any status)
    reads = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(V1, "tableVII_tracker.csv"))):
        m = DOC.search(r["table_id"])
        if m:
            reads[m.group(1)] += 1

    rows = []
    for dh, info in vols.items():
        toc_pages = len(toc.get(dh, ()))
        got = reads.get(dh, 0)
        if toc_pages == 0:
            status = "no_toc_entry" if got == 0 else "read_no_toc"
            gap = 0
        elif got == 0:
            status = "missing_all"; gap = toc_pages
        elif got >= toc_pages * 0.8:
            status = "complete"; gap = max(0, toc_pages - got)
        else:
            status = "partial"; gap = toc_pages - got
        rows.append({"dochash": dh, "pdf": info["pdf"], "year": info["year"], "province": info["prov"],
                     "toc_VII_pages": toc_pages, "VII_reads": got, "gap": gap, "status": status})

    rows.sort(key=lambda r: -r["gap"])
    with open(os.path.join(V1, "tableVII_volume_gaps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dochash", "pdf", "year", "province", "toc_VII_pages", "VII_reads", "gap", "status"])
        w.writeheader(); [w.writerow(r) for r in rows]

    from collections import Counter
    st = Counter(r["status"] for r in rows)
    print(f"volumes: {len(rows)}")
    for k in ("complete", "partial", "missing_all", "no_toc_entry", "read_no_toc"):
        print(f"  {k:14} {st.get(k,0)}")
    print(f"\ntotal TOC Table VII pages: {sum(r['toc_VII_pages'] for r in rows)}")
    print(f"total Table VII reads we have: {sum(r['VII_reads'] for r in rows)}")
    print(f"total page-gap (partial+missing): {sum(r['gap'] for r in rows)}")
    print("\nbiggest gaps (volumes to finish):")
    for r in rows[:12]:
        print(f"  {r['year']!s:5} {r['province'][:16]:16} {r['pdf'][:22]:22} TOC {r['toc_VII_pages']:3}p, have {r['VII_reads']:3}, gap {r['gap']:3} [{r['status']}]")
    print("wrote tableVII_volume_gaps.csv")


if __name__ == "__main__":
    main()
