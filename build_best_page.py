#!/usr/bin/env python3
"""best.html v2: the BEST available age table for EVERY place-year we have anything for.
Two tiers merged: (1) the prime province-level best dataset (verified Imperial Table VII runs);
(2) the area-age series (districts/taluks/cities/states + provinces not in prime). Entries that are
not an absolute Imperial Table VII are explicitly marked '≈ nearest available' with what they are.
Run AFTER build_best_dataset.py (this overwrites best.html)."""
import csv, io, json, re, os, zipfile, html
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__))
IMPVII = re.compile(r"(imperial\s+table\s+(no\.?\s*)?vii|table\s+(no\.?\s*)?vii)", re.I)
OFFTARGET = re.compile(r"subsidiary|distribution|per\s*(1|10),?000|proportion|literacy|marriage|birth", re.I)

z = zipfile.ZipFile(os.path.join(ROOT, "census_site/data/tableVII_best.zip"))
prime = list(csv.DictReader(io.TextIOWrapper(z.open("catalog.csv"))))
area = json.load(open(os.path.join(ROOT, "census_site/data/area_ages.json")))

entries = defaultdict(dict)   # place -> year -> rec
for b in prime:
    entries[b["province"]][b["year"]] = {
        "tier": "prime", "quality": b["quality"], "kind": b["source_kind"],
        "cells": b.get("n_cells", ""), "href": b["site_page"], "title": "Imperial Table VII (assembled)",
        "on_target": True}
for a, years in area.items():
    for y, rec in years.items():
        if y in entries.get(a, {}):
            continue                       # prime wins
        t = rec.get("title", "")
        on = bool(IMPVII.search(t)) and not OFFTARGET.search(t)
        entries[a][y] = {"tier": "area", "quality": "", "kind": rec.get("level", ""),
                         "cells": str(len(rec.get("rows", []))) + " age rows",
                         "href": f"ages.html?a={html.escape(a)}", "title": t, "on_target": on}

QB = {"verified": ("✓ verified", "#0a5d1e"), "minor-issues": ("~ minor", "#b8860b"),
      "needs-review": ("⚠ review", "#9c0006"), "incomplete": ("… incomplete", "#777")}
n_place = len(entries)
n_py = sum(len(v) for v in entries.values())
n_on = sum(1 for v in entries.values() for r in v.values() if r["on_target"])
out = ["<!doctype html><meta charset=utf-8><title>Best age table per place-year</title>",
       "<style>body{font:14px system-ui;margin:0;background:#fbfdff;color:#1a2233}",
       "header{background:#1f3864;color:#fff;padding:12px 20px}header a{color:#ffd966}",
       ".sec{margin:14px 20px 4px;font-weight:700;color:#1f3864;border-bottom:2px solid #e3e9f5;font-size:16px}",
       ".grid{display:flex;flex-wrap:wrap;gap:8px;margin:8px 20px}",
       ".c{background:#fff;border:1px solid #e2e6ee;border-left:4px solid #1f3864;border-radius:7px;padding:8px 12px;",
       "text-decoration:none;color:inherit;width:260px}.c:hover{box-shadow:0 2px 8px rgba(0,0,0,.1)}",
       ".c.approx{border-left-color:#b8860b;background:#fffdf5}",
       ".b{font-size:11px;padding:0 6px;border-radius:8px;color:#fff}</style>",
       f"<header><a href='index.html'>← home</a> <b>Best age table — every place &amp; census year with data</b> · ",
       f"{n_place} places · {n_py} place-years ({n_on} Imperial Table VII, {n_py-n_on} ≈ nearest available) · ",
       "<a href='data/tableVII_best.zip'>Table VII zip</a> · <a href='data/area_ages.zip'>area-age zip</a> · ",
       "<a href='ages.html'>📊 pyramids</a></header>",
       "<div style='margin:8px 20px;font-size:13px;color:#555'>Blue cards = assembled Imperial Table VII. ",
       "<span style='background:#fffdf5;border:1px solid #b8860b;border-radius:4px;padding:0 5px'>Amber cards</span>",
       " = ≈ nearest available age data for that place-year (subsidiary / proportional / civil-condition series), honestly labeled.</div>"]
for place in sorted(entries):
    out.append(f"<div class=sec>{html.escape(place)}</div><div class=grid>")
    for y in sorted(entries[place]):
        r = entries[place][y]
        cls = "c" if r["on_target"] else "c approx"
        ql, qc = QB.get(r["quality"], ("", ""))
        badge = (f"<span class=b style='background:{qc}'>{ql}</span> " if ql else "")
        kind = f"<span class=b style='background:#456'>{html.escape(str(r['kind']))}</span>" if r["kind"] else ""
        mark = "" if r["on_target"] else "<span class=b style='background:#b8860b'>≈ nearest</span> "
        out.append(f"<a class='{cls}' href='{r['href']}'><b>{y}</b> {mark}{badge}{kind}<br>"
                   f"<span style='color:#789;font-size:12px'>{html.escape(r['title'][:52])} · {r['cells']}</span></a>")
    out.append("</div>")
open(os.path.join(ROOT, "census_site/best.html"), "w").write("\n".join(out))
print(f"best.html v2: {n_place} places · {n_py} place-years · {n_on} on-target VII · {n_py-n_on} nearest-available")
