#!/usr/bin/env python3
"""Static audit of every table page in census_site/tables: scans exist and decode, matrix
present, CSV embedded, page labels consistent, no template leftovers. Prints a summary and
writes atlas/out/table_page_audit.csv for anything that fails."""
import os, re, glob, csv, collections, base64
from PIL import Image
SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "census_site")
probs = collections.Counter(); rows = []; sizes = []
img_ok = {}
for f in sorted(glob.glob(os.path.join(SITE, "tables", "*.html"))):
    s = open(f, errors="replace").read()
    if "<title>Table Withdrawn</title>" in s or "<title>Moved</title>" in s:
        continue   # withdrawal notices and forwarding stubs carry no scan or matrix by design
    name = os.path.basename(f)
    issues = []
    imgs = re.findall(r'<img[^>]+src="\.\./img/([^"]+)"', s)
    if not imgs:
        issues.append("no scan")
    for im in imgs:
        if im not in img_ok:
            p = os.path.join(SITE, "img", im)
            try:
                with Image.open(p) as I:
                    I.verify()
                with Image.open(p) as I:
                    img_ok[im] = I.size
            except Exception:
                img_ok[im] = None
        if img_ok[im] is None:
            issues.append(f"bad scan {im}")
        else:
            sizes.append(img_ok[im])
    cells = len(re.findall(r"<td(?! class=rh)[^>]*>[^<]+</td>", s))
    if "<table class=mx>" not in s:
        issues.append("no matrix")
    elif cells == 0:
        issues.append("empty matrix")
    m = re.search(r'<script type="text/plain" id="csv64">[A-Za-z0-9+/=]{20,}</script>', s)
    if not m and "dlcsv" in s:
        issues.append("csv missing")
    lbl = re.findall(r"page (\d+) of (\d+)", s)
    if lbl and any(int(b) != len(imgs) for a, b in lbl):
        issues.append("page-label mismatch")
    for leftover in ("{title}", "{matrices}", "__NAV__", "__DATA__", ">NaN<", ">undefined<"):
        if leftover in s:
            issues.append(f"leftover {leftover}")
    if s.count("<nav class=nav>") != 1:
        issues.append("nav count")
    for i in issues:
        probs[i.split(" ")[0] + " " + (i.split(" ")[1] if len(i.split()) > 1 else "")] += 1
    if issues:
        rows.append([name, "; ".join(issues)])
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "table_page_audit.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["page", "issues"]); w.writerows(rows)
n = len(glob.glob(os.path.join(SITE, "tables", "*.html")))
print(f"table pages: {n}; pages with issues: {len(rows)}; issue counts: {dict(probs)}")
ws = sorted(w for w, h in sizes)
print(f"scans: {len(img_ok)} distinct, unreadable {sum(1 for v in img_ok.values() if v is None)}; width median {ws[len(ws)//2] if ws else 0}px, min {ws[0] if ws else 0}px")
