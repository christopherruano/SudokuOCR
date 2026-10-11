#!/usr/bin/env python3
"""Number grounding: does every number in a digitized table actually appear on its page?

An independent, structure-free OCR (Tesseract, digits only) lists the numbers printed on the
scan; a table is grounded to the extent its digitized values are found there. This catches
the failure modes arithmetic checks miss — columns read from another page, synthesized
totals (e.g. "females" = unmarried + widowed), and hallucinated cells — without re-reading.

Pages are tried at 0/90/270 degrees (many census pages are printed sideways); the rotation
yielding the most numbers is used. Only values with >= 2 significant digits are scored
(single digits are too common to be evidence).

grounding(page_images, values) -> dict(recall, n_scored, n_found, missing_sample)
"""
import re, collections
from PIL import Image, ImageOps
import pytesseract

CFG = "--psm 11 -c tessedit_char_whitelist=0123456789.,"


def page_numbers(path, cache=None):
    if cache is not None and path in cache:
        return cache[path]
    im = Image.open(path).convert("L")
    w, h = im.size
    if max(w, h) < 2200:                          # tesseract needs ~3x the 1000 px site scans
        s = 2600 / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    im = ImageOps.autocontrast(im)
    best = collections.Counter()
    for rot in (0, 90, 270):
        if rot and sum(best.values()) >= 80:     # upright read is clearly a table page
            break
        txt = pytesseract.image_to_string(im.rotate(rot, expand=True), config=CFG)
        nums = collections.Counter()
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", txt):
            t = tok.replace(",", "").rstrip(".")
            if t:
                nums[t.lstrip("0") or "0"] += 1
        if sum(nums.values()) > sum(best.values()):
            best = nums
    if cache is not None:
        cache[path] = best
    return best


def norm_value(v):
    if v is None:
        return None
    if float(v).is_integer():
        return str(int(v))
    return ("%f" % v).rstrip("0").rstrip(".")


def grounding(paths, values, cache=None):
    found_pool = collections.Counter()
    for p in paths:
        found_pool += page_numbers(p, cache)
    scored = [norm_value(v) for v in values if v is not None and len(norm_value(v).replace(".", "").lstrip("0")) >= 2]
    if not scored:
        return dict(recall=None, n_scored=0, n_found=0, missing_sample=[])
    pool = set(found_pool)
    found = [v for v in scored if v in pool]
    missing = [v for v in scored if v not in pool]
    return dict(recall=len(found) / len(scored), n_scored=len(scored), n_found=len(found),
                missing_sample=missing[:12])
