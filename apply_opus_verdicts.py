#!/usr/bin/env python3
"""Apply the Opus adjudication verdicts to keeper_catalog.csv:
  restore           -> keeper_accepted   (Opus confirmed faithful -> back on explorer)
  reocr             -> keeper_reocr       (real table misread -> cluster re-OCR queue)
  exclude_not_table -> keeper_not_table   (diagram/chart/prose, not data -> drop)
  hold              -> keeper_suspect      (uncertain -> stays off)
"""
import csv, json, os
from collections import Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
MAP = {"restore": "keeper_accepted", "reocr": "keeper_reocr",
       "exclude_not_table": "keeper_not_table", "hold": "keeper_suspect"}


def main():
    verdicts = {x["table_id"]: x for x in json.load(open(os.path.join(V1, "opus_verdicts.json"))) if x.get("table_id")}
    path = os.path.join(V1, "keeper_catalog.csv")
    rows = list(csv.DictReader(open(path)))
    cols = rows[0].keys()
    n = Counter()
    for r in rows:
        x = verdicts.get(r["table_id"])
        if not x:
            continue
        r["status"] = MAP[x["action"]]
        r["review_note"] = (f"opus:{x['verdict']}/{x['completeness']}/match={x['values_match']} {x.get('note','')}")[:160]
        n[r["status"]] += 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print("applied opus verdicts:", dict(n))


if __name__ == "__main__":
    main()
