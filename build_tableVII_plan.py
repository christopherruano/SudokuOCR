#!/usr/bin/env python3
"""PRIORITY #1: make every Table VII (Age, Sex & Civil Condition — general
population age structure) as complete and correct as possible. Census every
Table VII we possess across ALL statuses, classify the action each needs, and
emit Table-VII-FIRST cluster lists + a coverage tracker.

Outputs (sendout_v1/):
  tableVII_tracker.csv     one row per Table VII: year, province, status, action
  tableVII_reocr.txt       cluster re-OCR image list (flagged Table VIIs) — run FIRST
  tableVII_newocr.jsonl    digitization queue for missing Table VII pages — run FIRST
  tableVII_coverage.csv    year x province: on-site / needs-reocr / missing-pages
"""
import os, re, csv, json
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
VII = re.compile(r"\bVII\b|age,?\s*sex\s*(and|&)?\s*civil|civil\s*condition", re.I)
AGEFAM = {"age_civil_condition", "age_sex_civil_condition", "age_sex_population"}
GOOD = {"confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled"}
REOCR = {"keeper_reocr", "keeper_badread", "keeper_suspect", "keeper_rejected",
         "needs_reocr", "hold_reocr"}
REVIEW = {"excluded_or_unscoped", "needs_human_review", "excluded_malformed"}


def isVII(fam, title):
    return fam in AGEFAM and bool(VII.search(title or ""))


def main():
    # cluster source image per table (for re-OCR list)
    src = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
        src[r["table_id"]] = r.get("source_image", "")
    for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
        src.setdefault(r["table_id"], r.get("source_image", ""))

    rows = []
    for fn, source in (("catalog_tables.csv", "existing"), ("keeper_catalog.csv", "keeper")):
        for r in csv.DictReader(open(os.path.join(V1, fn))):
            if not isVII(r["table_family"], r["title"]):
                continue
            st = r["status"]
            action = ("on_site" if st in GOOD else "reocr" if st in REOCR else "review" if st in REVIEW else "other")
            rows.append({"table_id": r["table_id"], "source": source, "status": st, "action": action,
                         "year": r["year"], "region": r["broad_region"], "province": r["canonical_name"],
                         "title": (r["title"] or "")[:80], "source_image": src.get(r["table_id"], "")})

    # write tracker
    with open(os.path.join(V1, "tableVII_tracker.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "source", "status", "action", "year",
                                          "region", "province", "title", "source_image"])
        w.writeheader(); [w.writerow(r) for r in rows]

    # Table-VII re-OCR list (cluster images)
    imgs = [r["source_image"] for r in rows if r["action"] == "reocr" and r["source_image"]]
    open(os.path.join(V1, "tableVII_reocr.txt"), "w").write("\n".join(dict.fromkeys(imgs)) + "\n")

    # missing Table VII pages -> digitization queue (reuse the manifest queue_rows for VII docs)
    nd = {r["dochash"]: r for r in csv.DictReader(open(os.path.join(V1, "newocr_neil_pages.csv")))
          if "VII" in r["tables"] and r["source_pdf"].strip()}
    DOC = re.compile(r"([0-9a-f]{16})_t")
    qrows = []
    import glob
    for line in open(os.path.join(ROOT, "export/cluster_sync/organized_manifest.jsonl")):
        try:
            j = json.loads(line)
        except Exception:
            continue
        m = DOC.search(j.get("source_image", "") or j.get("result_json", "") or "")
        if not m or m.group(1) not in nd:
            continue
        qr = j.get("queue_row", {}) or {}
        t = (qr.get("title", "") or "") + " " + (qr.get("canonical_topic", "") or "")
        if qr.get("file") and isinstance(qr.get("resolved_start_page"), int) and VII.search(t):
            qrows.append(qr)
    # dedupe
    seen, uq = set(), []
    for q in qrows:
        k = (q["file"], q.get("resolved_start_page"), q.get("resolved_end_page"))
        if k not in seen:
            seen.add(k); uq.append(q)
    with open(os.path.join(V1, "tableVII_newocr.jsonl"), "w") as f:
        for q in uq:
            f.write(json.dumps(q) + "\n")

    # coverage matrix year x region
    cov = defaultdict(lambda: Counter())
    for r in rows:
        cov[(str(r["year"] or "?"), r["region"] or "(none)")][r["action"]] += 1
    with open(os.path.join(V1, "tableVII_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year", "region", "on_site", "reocr", "review", "other"])
        for (y, reg), c in sorted(cov.items()):
            w.writerow([y, reg, c["on_site"], c["reocr"], c["review"], c["other"]])

    ac = Counter(r["action"] for r in rows)
    print(f"Table VII tables: {len(rows)}   actions: {dict(ac)}")
    print(f"  tableVII_reocr.txt: {len(set(imgs))} cluster images to re-OCR (run FIRST)")
    print(f"  tableVII_newocr.jsonl: {len(uq)} page-windows to extract (missing Table VII pages)")
    print(f"  tableVII_tracker.csv + tableVII_coverage.csv written")


if __name__ == "__main__":
    main()
