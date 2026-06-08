#!/usr/bin/env python3
"""Sample chopped multi-page groups (stratified by confidence x page-contiguity)
and resolve member scans so Opus can verify each grouping against the images:
same subject? correct page order? continuation bound to the right anchor?
Writes spotverify_candidates.json (anchor + sample members' local scans)."""
import os, re, json, csv, glob, hashlib
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")

IMG = {}
for pat in ("export/never_ocrd_images/**/*.png", "export/digitized_tables_referenced/**/*.png",
            "export/**/*.png", "census_site/img/*.jpg"):
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
        IMG.setdefault(os.path.basename(p).rsplit(".", 1)[0], p)

# table_id -> source_image
SRC = {}
for r in csv.DictReader(open(os.path.join(ROOT, "export/keeper_inference/source_catalog.csv"))):
    SRC[r["table_id"]] = r["source_image"]
for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv"))):
    SRC.setdefault(r["table_id"], r["source_image"])


def resolve(tid):
    s = SRC.get(tid, "") or tid
    return IMG.get(os.path.basename(s).rsplit(".", 1)[0], "")


def deterministic_sample(items, n):
    # stable pseudo-random pick (no RNG): sort by md5 of identity
    keyed = sorted(items, key=lambda g: hashlib.md5(g["dochash"].encode() + str(g["pages"]).encode()).hexdigest())
    return keyed[:n]


def main():
    groups = json.load(open(os.path.join(V1, "chop_groups.json")))
    # contiguity flag
    for g in groups:
        pgs = g["pages"]
        g["contig"] = (max(pgs) - min(pgs) + 1) <= len(pgs) + 2 if pgs else False
    strata = {
        "high_contig": [g for g in groups if g["confidence"] == "high" and g["contig"]],
        "high_noncontig": [g for g in groups if g["confidence"] == "high" and not g["contig"]],
        "med_contig": [g for g in groups if g["confidence"] == "medium" and g["contig"]],
        "med_noncontig": [g for g in groups if g["confidence"] == "medium" and not g["contig"]],
        "low": [g for g in groups if g["confidence"] == "low"],
    }
    quota = {"high_contig": 12, "high_noncontig": 16, "med_contig": 10, "med_noncontig": 16, "low": 10}
    cands = []
    for k, gs in strata.items():
        for g in deterministic_sample(gs, quota[k]):
            mem = g["member_table_ids"]
            # anchor + up to 3 spread members
            picks = [mem[0]] + (mem[1:][:: max(1, len(mem) // 3)][:3] if len(mem) > 1 else [])
            imgs = [(t, resolve(t)) for t in dict.fromkeys(picks)]
            imgs = [(t, p) for t, p in imgs if p]
            if len(imgs) < 2:
                continue
            cands.append({"idx": len(cands), "stratum": k, "dochash": g["dochash"],
                          "subject": g["subject"], "confidence": g["confidence"],
                          "pages": g["pages"], "n_members": len(mem),
                          "member_images": [p for _, p in imgs]})
    json.dump(cands, open(os.path.join(V1, "spotverify_candidates.json"), "w"))
    from collections import Counter
    print(f"spot-verify candidates: {len(cands)}")
    print("  by stratum:", dict(Counter(c["stratum"] for c in cands)))


if __name__ == "__main__":
    main()
