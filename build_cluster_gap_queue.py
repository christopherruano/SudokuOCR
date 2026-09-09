#!/usr/bin/env python3
"""Build the CLUSTER work-list of Table VII pages we HAVEN'T done yet — to run on the
cluster (render+OCR) before assembly. Anchored on our validated HAVE set so it keeps
real gaps and drops the CC detector's isolated non-tables.

Inputs (export_final/sendout_final/sendout_v1/):
  tableVII_inventory_cc.csv   the CC's detected Table VII pages (saoa, pdf_page,
                              cluster_path, status/tier, year, title, province)
  cc_volumes.csv              per-volume cluster_path + npages_total
  have_imghashes.txt          our crop hashes
  master_tracking.csv         imghash -> is_tableVII / validation
Outputs:
  cluster_gap_queue.jsonl     blocks to render+OCR (digitize_table_pages_task format)
  cluster_gap_queue.csv       same, human-readable, with trust tier
  gap_singletons_hold.csv     isolated low-confidence pages we are NOT queueing
"""
import csv, json, hashlib, re
from collections import defaultdict

V1 = "export_final/sendout_final/sendout_v1"
GAPBRIDGE = 3        # pages <=this apart are the same contiguous Table VII block


def dochash(cluster_path):
    return hashlib.sha1(cluster_path.encode()).hexdigest()[:16]


def main():
    have = set(open(f"{V1}/have_imghashes.txt").read().split())
    mt = {r["imghash"]: r for r in csv.DictReader(open(f"{V1}/master_tracking.csv"))}

    # 1) reverse HAVE -> the Table VII pages we already hold, by volume
    vols = {}                          # cluster_path -> npages
    meta = {}                          # saoa -> {cluster_path, year, province, title}
    for r in csv.DictReader(open(f"{V1}/cc_volumes.csv")):
        cp = r.get("cluster_path", "")
        if not cp:
            continue
        try:
            n = int(r["npages_total"])
        except Exception:
            n = 1600
        vols[cp] = max(vols.get(cp, 0), n)
        meta[r["saoa_id"]] = {"cluster_path": cp, "year": r.get("year", ""),
                              "province": r.get("province_or_volume", ""), "title": r.get("title", "")}
    have_vii = defaultdict(set)         # saoa -> {pdf_page we HAVE and is Table VII}
    for cp, n in vols.items():
        sa = (re.search(r"\.(\d{6,})\.pdf", cp) or [None, ""])[1]
        for p in range(1, min(n, 2000) + 1):
            ih = hashlib.sha1(f"{cp}:{p}".encode()).hexdigest()
            if ih in have and mt.get(ih, {}).get("is_tableVII") == "Y":
                have_vii[sa].add(p)

    # 2) CC detected pages, by volume (with tier); also collect provenance
    cc = defaultdict(dict)              # saoa -> {pdf_page: tier}
    for r in csv.DictReader(open(f"{V1}/tableVII_inventory_cc.csv")):
        cp = r["cluster_path"]; pg = r["pdf_page"]
        if not (cp and str(pg).isdigit()):
            continue
        sa = r["saoa_id"]
        cc[sa][int(pg)] = r["status"]
        meta.setdefault(sa, {"cluster_path": cp, "year": r.get("year", ""),
                             "province": r.get("province_or_volume", ""), "title": r.get("title", "")})

    # 3) per volume: union of (HAVE Table VII) + (CC detected); contiguous blocks
    queue, csv_rows, singletons = [], [], []
    tindex = 0
    for sa in sorted(set(have_vii) | set(cc)):
        pages = {}                      # page -> source ('have' / cc-tier)
        for p in have_vii.get(sa, ()):
            pages[p] = "have"
        for p, tier in cc.get(sa, {}).items():
            pages.setdefault(p, tier)   # prefer 'have' label if both
        if not pages:
            continue
        order = sorted(pages)
        # split into contiguous blocks
        blocks = [[order[0]]]
        for p in order[1:]:
            (blocks[-1].append(p) if p - blocks[-1][-1] <= GAPBRIDGE else blocks.append([p]))

        m = meta.get(sa, {})
        cp = m.get("cluster_path", "")
        for blk in blocks:
            lo, hi = blk[0], blk[-1]
            gap_pages = [p for p in range(lo, hi + 1) if p not in have_vii.get(sa, set())]
            if not gap_pages:
                continue               # we already have the whole block — nothing to do
            anchored = any(p in have_vii.get(sa, set()) for p in blk)
            confirmed = any(pages.get(p) == "CONFIRMED" for p in blk)
            if anchored or confirmed:
                trust = "trusted"
            elif len(blk) >= 3:
                trust = "candidate_new_volume"
            elif len(blk) == 2:
                trust = "weak_run"
            else:
                trust = "singleton"
            row = {"saoa_id": sa, "year": m.get("year", ""), "province": m.get("province", ""),
                   "title": m.get("title", ""), "cluster_path": cp,
                   "block_start": lo, "block_end": hi, "n_pages": hi - lo + 1,
                   "n_gap": len(gap_pages), "trust": trust}
            csv_rows.append(row)
            if trust == "singleton":
                singletons.append(row)
                continue
            tindex += 1
            queue.append({"file": cp, "filename": cp.split("/")[-1],
                          "year": m.get("year", ""), "province": m.get("province", ""),
                          "dochash": dochash(cp) if cp else "",
                          "title": f"Table VII (gap:{trust})",
                          "table_index": tindex,
                          "resolved_start_page": lo, "resolved_end_page": hi,
                          "n_gap_pages": len(gap_pages), "range_status": trust})

    # write outputs
    with open(f"{V1}/cluster_gap_queue.jsonl", "w") as f:
        for q in queue:
            f.write(json.dumps(q) + "\n")
    cols = ["saoa_id", "year", "province", "title", "cluster_path",
            "block_start", "block_end", "n_pages", "n_gap", "trust"]
    with open(f"{V1}/cluster_gap_queue.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in sorted(csv_rows, key=lambda r: (r["trust"], -r["n_gap"]))]
    with open(f"{V1}/gap_singletons_hold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in singletons]

    # summary
    from collections import Counter
    bt = Counter(r["trust"] for r in csv_rows)
    gp = Counter()
    for r in csv_rows:
        gp[r["trust"]] += r["n_gap"]
    print("CLUSTER GAP QUEUE — blocks containing pages we haven't done:")
    for t in ("trusted", "candidate_new_volume", "weak_run", "singleton"):
        print(f"  {t:<22} blocks={bt.get(t,0):4d}   gap pages={gp.get(t,0):5d}")
    queued_blocks = sum(1 for r in csv_rows if r["trust"] != "singleton")
    queued_gap = sum(r["n_gap"] for r in csv_rows if r["trust"] != "singleton")
    render_pages = sum(q["resolved_end_page"] - q["resolved_start_page"] + 1 for q in queue)
    print(f"\nQUEUED: {queued_blocks} blocks  |  {queued_gap} new gap pages  "
          f"(cluster renders {render_pages} pages incl. context)")
    print(f"HELD as singletons (not queued): {bt.get('singleton',0)}")
    print(f"\nwrote {V1}/cluster_gap_queue.jsonl  (+ .csv, gap_singletons_hold.csv)")


if __name__ == "__main__":
    main()
