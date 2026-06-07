#!/usr/bin/env python3
"""Build the keeper re-OCR queue from the QA verdicts. Tiers:
  T1 keeper_reocr   (234)  - Opus-confirmed real-table misreads (mostly dropped Persons cols)
  T2 keeper_badread/suspect(283) - flagged, no local scan to verify -> re-OCR to be safe
  T3 keeper_rejected (1284) - constraint failures
Outputs (sendout_v1/): reocr_keeper_images.txt (T1+T2 cluster image paths),
reocr_keeper_manifest.csv (all tiers w/ reason + cluster source_image).
"""
import csv, os
from collections import Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
TIER = {"keeper_reocr": "T1_opus_misread", "keeper_badread": "T2_unverified",
        "keeper_suspect": "T2_unverified", "keeper_rejected": "T3_constraint_fail"}


def main():
    src = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv")))}
    cat = list(csv.DictReader(open(os.path.join(V1, "keeper_catalog.csv"))))

    rows, imgs_immediate = [], []
    for r in cat:
        tier = TIER.get(r["status"])
        if not tier:
            continue
        sc = src.get(r["table_id"], {})
        cimg = sc.get("source_image", "")
        rows.append({"table_id": r["table_id"], "tier": tier, "status": r["status"],
                     "year": r["year"], "place": r["canonical_name"], "table_family": r["table_family"],
                     "source_image": cimg, "review_note": r.get("review_note", ""), "title": r["title"]})
        if tier.startswith(("T1", "T2")) and cimg:
            imgs_immediate.append(cimg)

    with open(os.path.join(V1, "reocr_keeper_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "tier", "status", "year", "place",
                                          "table_family", "source_image", "review_note", "title"])
        w.writeheader(); [w.writerow(r) for r in sorted(rows, key=lambda x: x["tier"])]
    with open(os.path.join(V1, "reocr_keeper_images.txt"), "w") as f:
        f.write("\n".join(dict.fromkeys(imgs_immediate)) + "\n")

    print(f"keeper re-OCR manifest: {len(rows)} tables   by tier: {dict(Counter(r['tier'] for r in rows))}")
    print(f"reocr_keeper_images.txt (T1+T2 immediate batch): {len(set(imgs_immediate))} images")
    print(f"  with resolvable cluster source_image: {sum(1 for r in rows if r['source_image'])}/{len(rows)}")


if __name__ == "__main__":
    main()
