#!/usr/bin/env python3
"""Count the numeric columns printed on a scan, independent of any reading of the digits.

Census figures are right-aligned, so the right edges of number boxes (Tesseract word boxes
containing digits or the '...' nil mark) stack into vertical bands, one per column. A band
counts as a column when it holds at least max(4, 30% of the busiest band) boxes.
scan_columns(path) -> (n_columns, band x-positions as fractions of page width, rotation)
"""
import re, collections
from PIL import Image, ImageOps
import pytesseract
import os
SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "census_site")

NUMLIKE = re.compile(r"^[\d,.·'•…]*\d[\d,.·'•…]*$|^\.{2,}$|^…$")


def bands(im):
    W = im.size[0]
    d = pytesseract.image_to_data(im, config="--psm 11", output_type=pytesseract.Output.DICT)
    xs = []
    for t, l, w, c in zip(d["text"], d["left"], d["width"], d["conf"]):
        t = t.strip()
        if t and NUMLIKE.match(t) and float(c) > 20:
            xs.append((l + w) / W)
    if not xs:
        return []
    xs.sort()
    tol = 0.012
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            groups.append(cur); cur = [x]
    groups.append(cur)
    big = max(len(g) for g in groups)
    bands.last_density = big          # boxes in the busiest band ~ number of data rows read
    return [round(sum(g) / len(g), 3) for g in groups if len(g) >= max(4, 0.3 * big)]


def scan_columns(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = 1800 / max(w, h)
    im = ImageOps.autocontrast(im.resize((int(w * s), int(h * s)), Image.LANCZOS))
    best = (0, [], 0, 0)
    for rot in (0, 90, 270):
        bands.last_density = 0
        b = bands(im.rotate(rot, expand=True))
        if len(b) > len(best[1]):
            best = (len(b), b, rot, bands.last_density)
        if rot == 0 and len(b) >= 3:
            break
    return best


def _work(img):
    import os
    try:
        n, b, rot, dens = scan_columns(os.path.join(SITE, "img", img))
        return dict(img=img, n=n, bands=b, rot=rot, density=dens)
    except Exception as e:
        return dict(img=img, n=None, error=str(e)[:100])


if __name__ == "__main__":
    import os, glob, json
    from concurrent.futures import ProcessPoolExecutor
    HERE = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.join(HERE, "..", "census_site")
    OUT = os.path.join(HERE, "out", "scan_columns.jsonl")
    imgs = set()
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        imgs.update(re.findall(r'src="\.\./img/([^"]+)"', open(f, errors="ignore").read()))
    done = {json.loads(l)["img"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = sorted(imgs - done)
    print(f"images {len(imgs)}, to do {len(todo)}", flush=True)
    with open(OUT, "a") as f, ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for k, r in enumerate(ex.map(_work, todo, chunksize=2)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 500 == 0:
                print(f"{k}/{len(todo)}", flush=True)
    print("COLUMNS_DONE", flush=True)
