#!/usr/bin/env python3
"""Merge all never-OCR'd triage results (gap-fillers + new-table) into one CSV
plus a keepers-only CSV. Run after the new-table workflow completes."""
import json, glob, csv, os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
TRI = os.path.join(ROOT, "export/never_ocrd_triage")
master = {m["hash"]: m for m in json.load(open(os.path.join(TRI, "master_worklist.json")))}

results = {}
for f in glob.glob(os.path.join(TRI, "batches/result_fill_*.json")):
    results.update({h: dict(v, cls="filler") for h, v in json.load(open(f)).items()})
nb = 0
for f in glob.glob(os.path.join(TRI, "newbatches/result_new_*.json")):
    try:
        results.update({h: dict(v, cls="new") for h, v in json.load(open(f)).items()})
        nb += 1
    except Exception as e:
        print("bad", f, e)

rows = []
for h, v in results.items():
    m = master.get(h, {})
    rows.append({
        "hash": h, "cls": v.get("cls"), "year": m.get("year"), "dir": m.get("dir"),
        "title": (m.get("title") or "")[:50],
        "on_subject": v.get("on_subject"), "doc_type": v.get("doc_type"),
        "topic": v.get("topic"), "table_identity": v.get("table_identity"),
        "quality": v.get("quality"), "page_number": v.get("page_number"),
        "note": v.get("note"), "img": m.get("img"),
    })

cols = ["hash", "cls", "year", "dir", "title", "on_subject", "doc_type", "topic",
        "table_identity", "quality", "page_number", "note", "img"]


def is_on(r):
    return r["on_subject"] in (True, "true", "True")


for name, sel in [("triage_all.csv", rows), ("keepers.csv", [r for r in rows if is_on(r)])]:
    with open(os.path.join(TRI, name), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader()
        for r in sel:
            w.writerow(r)

on = [r for r in rows if is_on(r)]
print(f"new-table result files merged: {nb}")
print(f"total pages triaged: {len(rows)} (fillers + new)")
print(f"ON-SUBJECT keepers: {len(on)}")
print("keepers by cls:", dict(Counter(r["cls"] for r in on)))
print("keepers by year:", dict(sorted(Counter(r["year"] for r in on).items(), key=lambda x: str(x[0]))))
print("keepers by topic:")
for k, v in Counter(r["topic"] for r in on).most_common(10):
    print(f"  {k:30} {v}")
print("keeper quality:", dict(Counter(r["quality"] for r in on)))
print(f"\nwrote {TRI}/triage_all.csv and {TRI}/keepers.csv")
