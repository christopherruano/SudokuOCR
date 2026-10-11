#!/usr/bin/env python3
"""Count table columns from the printed vertical rules (independent of cell contents).

Binarize; for every pixel column take the longest run of dark pixels (allowing small gaps
from broken print); a rule is a column whose run is >= 18% of the page height. Adjacent
rule pixels merge (double rules, thick lines). Columns = rules - 1 across the table.
"""
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def rules(path, min_frac=0.18):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = 1400 / max(w, h)
    im = ImageOps.autocontrast(im.resize((int(w * s), int(h * s))))
    a = np.asarray(im) < 140
    # bridge print gaps: vertical dilation by 5 px
    a = np.asarray(Image.fromarray((a * 255).astype("uint8")).filter(ImageFilter.MaxFilter(5))) > 0
    H, W = a.shape
    best = np.zeros(W, int)
    run = np.zeros(W, int)
    for row in a:
        run = np.where(row, run + 1, 0)
        best = np.maximum(best, run)
    xs = np.where(best >= min_frac * H)[0]
    out, cur = [], []
    for x in xs:
        if cur and x - cur[-1] > 6:
            out.append(int(np.mean(cur))); cur = []
        cur.append(x)
    if cur:
        out.append(int(np.mean(cur)))
    # drop page-edge artefacts (scan borders, binding shadow)
    out = [x for x in out if 0.03 * W < x < 0.97 * W]
    return [round(x / W, 3) for x in out]
