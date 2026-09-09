#!/usr/bin/env python3
"""Quick population-pyramid charts from export/pyramids/pyramids_long.csv. Picks a diverse set of clean
province/state pyramids (standard bands, good totals, spread across years) and renders back-to-back
M/F pyramids -> export/pyramids/charts/. Also a combined grid.
  python3 graph_pyramids.py
"""
import os, csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(ROOT, "export/pyramids/pyramids_long.csv")
OUT = os.path.join(ROOT, "export/pyramids/charts")
PROV = ("british_province", "princely_state", "all_india", "modern_state")


def load():
    tabs = defaultdict(lambda: {"meta": None, "cells": defaultdict(dict)})
    for r in csv.DictReader(open(LONG)):
        t = tabs[r["table_id"]]
        if t["meta"] is None:
            t["meta"] = (r["place"], r["year"], r["entity_type"], r["geo_level"])
        lo = int(r["age_lower"]); hi = r["age_upper"]
        t["cells"][(lo, hi)][r["sex"]] = int(r["population"])
    return tabs


def draw(ax, place, year, cells, entity):
    bands = sorted(cells)
    labels = [f"{lo}-{hi}" if hi else f"{lo}+" for (lo, hi) in bands]
    m = [cells[b].get("Male", 0) for b in bands]
    f = [cells[b].get("Female", 0) for b in bands]
    y = range(len(bands))
    ax.barh(y, [-x for x in m], color="#3a6ea5", height=0.82, label="Male")
    ax.barh(y, f, color="#c44e6a", height=0.82, label="Female")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#333", lw=0.6)
    mx = max(max(m, default=1), max(f, default=1))
    ax.set_xlim(-mx * 1.1, mx * 1.1)
    ax.set_xticklabels([f"{abs(int(t)):,}" for t in ax.get_xticks()], fontsize=6)
    tot = sum(m) + sum(f)
    ax.set_title(f"{place[:26]} — {year}\n{tot:,} persons ({entity.replace('_',' ')})", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    tabs = load()
    # candidates: province/state, >=8 bands, standard start (0-5 or 0-1), good total, M&F present
    cand = []
    for tid, t in tabs.items():
        place, year, ent, gl = t["meta"]
        if ent not in PROV or not year:
            continue
        cells = t["cells"]
        bands = sorted(cells)
        if len(bands) < 8:
            continue
        tot = sum(c.get("Male", 0) + c.get("Female", 0) for c in cells.values())
        nz = sum(1 for c in cells.values() if c.get("Male") and c.get("Female"))
        if tot < 200000 or nz < len(bands) * 0.7:
            continue
        cand.append((tot, place, year, ent, cells, tid))
    # diversify: one per (place,year), spread across years, biggest first
    cand.sort(key=lambda x: -x[0])
    seen, picks = set(), []
    for c in cand:
        key = (c[1], c[2])
        yr = c[2]
        if key in seen:
            continue
        seen.add(key); picks.append(c)
        if len(picks) >= 6 and len({p[2] for p in picks}) >= 4:
            break
    picks = picks[:6]

    # grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (tot, place, year, ent, cells, tid) in zip(axes.flat, picks):
        draw(ax, place, year, cells, ent)
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    axes.flat[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle("Indian Census — Population Pyramids (Table VII: Age × Sex)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid = os.path.join(OUT, "pyramids_grid.png")
    fig.savefig(grid, dpi=130); plt.close(fig)
    # individuals
    for tot, place, year, ent, cells, tid in picks:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw(ax, place, year, cells, ent); ax.legend(loc="lower right", fontsize=7, frameon=False)
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in f"{place}_{year}")[:40]
        fig.savefig(os.path.join(OUT, safe + ".png"), dpi=120); plt.close(fig)
    print(f"rendered {len(picks)} pyramids -> {OUT}")
    for tot, place, year, ent, cells, tid in picks:
        print(f"  {place} {year}  ({tot:,} persons, {len(cells)} bands)")
    print(f"grid: {grid}")


if __name__ == "__main__":
    main()
