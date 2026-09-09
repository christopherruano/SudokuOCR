#!/usr/bin/env python3
"""Compress census_site/img scans PNG -> JPEG (grayscale, downscale, quality) to shrink the deploy
~10x so it uploads fast on a slow connection. Then updates every HTML reference img/<hash>.png ->
img/<hash>.jpg and removes the old PNGs. Run once on a frozen census_site before deploying.
  python3 compress_site_images.py
"""
import glob, os, re
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "census_site/img")
MAXPX = 2200
Q = 82


def main():
    pngs = glob.glob(os.path.join(IMG, "*.png"))
    done = err = saved = 0
    before = sum(os.path.getsize(p) for p in pngs)
    for p in pngs:
        jpg = p[:-4] + ".jpg"
        if os.path.exists(jpg):
            os.remove(p); continue
        try:
            im = Image.open(p).convert("L")
            w, h = im.size
            if max(w, h) > MAXPX:
                s = MAXPX / max(w, h)
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
            im.save(jpg, "JPEG", quality=Q, optimize=True)
            os.remove(p)
            done += 1
        except Exception:
            err += 1
        if done % 500 == 0 and done:
            print(f"  {done} converted...", flush=True)
    after = sum(os.path.getsize(f) for f in glob.glob(os.path.join(IMG, "*.jpg")))
    print(f"converted {done} images (errors {err}) | {before//1024//1024}MB -> {after//1024//1024}MB")

    # update every HTML reference img/<hash>.png -> .jpg (handles ../img/, img/, ?thumb, href)
    pat = re.compile(r"(img/[0-9a-f]{40})\.png")
    patched = 0
    for hf in glob.glob(os.path.join(ROOT, "census_site/**/*.html"), recursive=True):
        s = open(hf, encoding="utf-8", errors="ignore").read()
        if "img/" in s and ".png" in s:
            ns = pat.sub(r"\1.jpg", s)
            if ns != s:
                open(hf, "w", encoding="utf-8").write(ns); patched += 1
    print(f"patched {patched} HTML files (img refs .png -> .jpg)")
    print(f"census_site now: {sum(os.path.getsize(f) for f in glob.glob(os.path.join(ROOT,'census_site/**/*'),recursive=True) if os.path.isfile(f))//1024//1024}MB")


if __name__ == "__main__":
    main()
