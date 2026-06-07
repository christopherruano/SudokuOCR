#!/usr/bin/env python3
"""Apply red-flag verdicts to keeper_catalog.csv so the explorer reflects trust.
  HARD (full_zero/single_value/mostly_zero) -> status 'keeper_badread'  (exclude from trusted views)
  SOFT (constant_col/near_constant)         -> status 'keeper_suspect'   (surface in review)
  qc_flag-only                              -> unchanged (minor)
Writes a review_note with the detail so it shows in the /review queue.
"""
import csv, os

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
HARD = ("full_zero", "single_value", "mostly_zero")
SOFT = ("constant_col", "near_constant")


def verdict(reasons):
    if any(k in reasons for k in HARD):
        return "keeper_badread"
    if any(k in reasons for k in SOFT):
        return "keeper_suspect"
    return None   # qc_flag only -> leave alone


def main():
    flag = {}
    for r in csv.DictReader(open(os.path.join(V1, "keeper_bad_reads.csv"))):
        v = verdict(r["reasons"])
        if v:
            flag[r["table_id"]] = (v, r["reasons"], r["detail"])

    path = os.path.join(V1, "keeper_catalog.csv")
    rows = list(csv.DictReader(open(path)))
    cols = rows[0].keys()
    n = {"keeper_badread": 0, "keeper_suspect": 0}
    for r in rows:
        f = flag.get(r["table_id"])
        if f:
            r["status"] = f[0]
            r["review_note"] = f"{f[1]} | {f[2]}"[:160]
            n[f[0]] += 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"flagged keeper tables: {n}")
    print("  (kept in catalog/map; facts retained but status marks them for the review queue)")


if __name__ == "__main__":
    main()
