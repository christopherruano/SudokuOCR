#!/usr/bin/env python3
"""Shrink census_site scans for web hosting: downscale to <=1400px and convert
PNG -> JPEG (q82). Updates the table pages' <img> refs from .png to .jpg.
Keeps the matrices/data pixel-exact; only the reference scan image is lighter."""
import os, glob, re
from PIL import Image

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census_site")
IMG = os.path.join(SITE, "img")
MAXW = 1400

pngs = glob.glob(os.path.join(IMG, "*.png"))
before = sum(os.path.getsize(p) for p in pngs)
done = 0
for p in pngs:
    try:
        im = Image.open(p).convert("RGB")
        if im.width > MAXW:
            im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
        im.save(p[:-4] + ".jpg", "JPEG", quality=82, optimize=True)
        os.remove(p)
        done += 1
    except Exception as e:
        print("skip", os.path.basename(p), e)
    if done % 500 == 0:
        print(f"  {done}/{len(pngs)}")
after = sum(os.path.getsize(f) for f in glob.glob(os.path.join(IMG, "*.jpg")))

# repoint <img> refs in table pages
pat = re.compile(r'(\.\./img/[0-9a-f]+)\.png')
for h in glob.glob(os.path.join(SITE, "tables", "*.html")):
    s = open(h).read()
    s2 = pat.sub(r"\1.jpg", s)
    if s2 != s:
        open(h, "w").write(s2)

print(f"converted {done} scans  {before/1e9:.2f}GB -> {after/1e6:.0f}MB")
