#!/usr/bin/env python3
"""Preprocess the illegible/plateaued Table VII scans (upscale 2x + grayscale +
autocontrast + unsharp) — the documented method for bridging hard images — into
enhanced copies keyed by the SAME imghash, plus a list to re-OCR. The enhanced
re-OCR results then replace the originals ONLY if they parse better (gated downstream
by U+M+W=Population)."""
import os
from PIL import Image, ImageOps, ImageFilter

V1 = "export_final/sendout_final/sendout_v1"
OUT = os.path.join(V1, "enhanced")
import re
HEX = re.compile(r"([0-9a-f]{40})")


def main():
    os.makedirs(OUT, exist_ok=True)
    scans = [l.strip() for l in open(os.path.join(V1, "illegible_scans.txt")) if l.strip()]
    listf = []
    ok = err = 0
    for p in scans:
        ih = HEX.search(os.path.basename(p))
        if not ih:
            continue
        dst = os.path.join(OUT, f"{ih.group(1)}.png")
        try:
            im = Image.open(p).convert("L")
            w, h = im.size
            scale = 2 if max(w, h) < 2600 else 1
            if scale > 1:
                im = im.resize((w * scale, h * scale), Image.LANCZOS)
            im = ImageOps.autocontrast(im, cutoff=1)
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
            im.save(dst)
            listf.append(dst); ok += 1
        except Exception as e:
            err += 1
            print(f"  err {os.path.basename(p)[:12]}: {str(e)[:50]}")
    open(os.path.join(V1, "enhanced_list.txt"), "w").write("\n".join(listf) + "\n")
    print(f"enhanced {ok} scans (err {err}) -> {OUT}/  + enhanced_list.txt")


if __name__ == "__main__":
    main()
